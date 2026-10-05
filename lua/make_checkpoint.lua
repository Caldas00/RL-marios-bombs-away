-- make_checkpoint.lua -- creates the Game B start checkpoint (sta/gnw_mbaway/1.sta)
--
-- Only needs to run ONCE (1.sta already ships with the project). Run it again only
-- if you want to regenerate the checkpoint: WARNING, this overwrites 1.sta.
--
--   mame gnw_mbaway -autoboot_script lua/make_checkpoint.lua -autoboot_delay 3 -window
--
-- In MAME: press Game B and play normally. The script saves the state on the 3rd
-- falling edge of segment 1.0.2 (Mario received the bomb for the 3rd time, i.e.
-- the game is already running with a bomb in hand) and freezes: it won't save again.
-- After that it keeps auto-resetting on every death (useful for testing).
--
-- Logic:
--   * 1.0.2 turns off -> Mario received the bomb. On the 3rd edge it saves the
--                        checkpoint and freezes (won't save again).
--   * 0.10.0 turns on -> bomb delivered. Enters game_stopped.
--   * 1.0.2 turns off -> leaves game_stopped (new bomb in hand).
--   * m_2C -> 15      -> death. Loads the checkpoint.

local mach = manager.machine
local root = mach.devices[":"]
local cpu  = mach.devices[":maincpu"]
local ram  = cpu.spaces["data"] or cpu.spaces["program"]

------------------------------------------------------------------ config
local LOGFILE      = "make_checkpoint.log"   -- relative to the project folder
local SEG_PEGAR    = "1.0.2"    -- turns off when Mario receives the bomb
local SEG_ENTREGA  = "0.10.0"   -- turns on when the bomb is delivered
local SAVE_ON_EDGE = 3          -- edge on which it saves and freezes
local SLOT         = "1"
local ADDR_MORTE   = 0x2C
local VALOR_MORTE  = 15
local COOLDOWN     = 12

------------------------------------------------------------------ log
local logf = io.open(LOGFILE, "w")
logf:setvbuf("line")
local function log(s) logf:write(s .. "\n"); print(s) end

------------------------------------------------------------------ output proxies
local proxies = {}
local function get(name)
  local p = proxies[name]
  if p then return p:get() end
  local q = root:output(name)
  if q and q:exists() then
    proxies[name] = q
    log(">>> proxy ligado a " .. name)
    return q:get()
  end
  return nil
end

------------------------------------------------------------------ state
local frame        = 0
local prev_pegar   = nil
local prev_entrega = nil
local prev_ram     = -1
local cooldown     = 0
local edges        = 0
local congelado    = false
local tem_save     = false
local mortes       = 0
local entregas     = 0
local game_stopped = false

------------------------------------------------------------------ loop
local function tick()
  frame = frame + 1

  local rp, re = get(SEG_PEGAR), get(SEG_ENTREGA)
  if rp == nil or re == nil then return end

  local pegar   = (rp ~= 0)
  local entrega = (re ~= 0)
  local v       = ram:read_u8(ADDR_MORTE)

  if prev_pegar == nil then
    prev_pegar, prev_entrega, prev_ram = pegar, entrega, v
    return
  end

  if cooldown > 0 then
    cooldown = cooldown - 1
    prev_pegar, prev_entrega, prev_ram = pegar, entrega, v
    return
  end

  ---------------------------------------------------------------- death
  if tem_save and v == VALOR_MORTE and prev_ram ~= VALOR_MORTE then
    mortes = mortes + 1
    local ok, err = pcall(function() mach:load(SLOT) end)
    if ok then
      log(string.format(">>> MORTE #%d no frame %d -- reiniciado", mortes, frame))
    else
      log(">>> load falhou: " .. tostring(err))
    end
    game_stopped = false
    cooldown  = COOLDOWN
    prev_pegar, prev_entrega, prev_ram = pegar, entrega, v
    return
  end

  ---------------------------------------------------------------- delivery
  if entrega and not prev_entrega then
    entregas = entregas + 1
    game_stopped = true
    log(string.format("[entrega %d] frame %d -- game_stopped = true",
                      entregas, frame))
  end

  ---------------------------------------------------------------- 1.0.2 edge
  if prev_pegar and not pegar then
    edges = edges + 1

    if game_stopped then
      game_stopped = false
      log(string.format("[bomba nova] frame %d -- game_stopped = false", frame))
    end

    if not congelado then
      if edges >= SAVE_ON_EDGE then
        local ok, err = pcall(function() mach:save(SLOT) end)
        if ok then
          tem_save  = true
          congelado = true
          log(string.format("[checkpoint FIXO] flanco %d, frame %d -- gravado e congelado",
                            edges, frame))
        else
          log(">>> falhou a gravar: " .. tostring(err))
        end
        cooldown = COOLDOWN
      else
        log(string.format("[flanco %d] frame %d -- ignorado (grava no %d)",
                          edges, frame, SAVE_ON_EDGE))
      end
    end
  end

  prev_pegar, prev_entrega, prev_ram = pegar, entrega, v
end

_G.notifier = emu.add_machine_frame_notifier(tick)
log(">>> make_checkpoint.lua ativo | grava no flanco " .. SAVE_ON_EDGE .. " e congela")
log(">>> log em " .. LOGFILE)
