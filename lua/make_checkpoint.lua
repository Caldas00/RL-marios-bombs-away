-- make_checkpoint.lua -- cria o checkpoint do inicio do Game B (sta/gnw_mbaway/1.sta)
--
-- So e preciso correr UMA vez (o 1.sta ja vem no projeto). Volta a correr apenas
-- se quiseres regenerar o checkpoint: ATENCAO, isto reescreve o 1.sta.
--
--   mame gnw_mbaway -autoboot_script lua/make_checkpoint.lua -autoboot_delay 3 -window
--
-- No MAME: carrega em Game B e joga normalmente. O script grava o estado no 3o
-- flanco de descida do segmento 1.0.2 (o Mario recebeu a bomba pela 3a vez, ou
-- seja ja com o jogo a correr e uma bomba na mao) e congela: nao volta a gravar.
-- Depois disso fica a fazer reset automatico em cada morte (util para testar).
--
-- Logica:
--   * 1.0.2 desliga  -> Mario recebeu a bomba. No 3o flanco grava o checkpoint
--                       e congela (nao volta a gravar).
--   * 0.10.0 liga    -> bomba entregue. Entra em game_stopped.
--   * 1.0.2 desliga  -> sai de game_stopped (nova bomba na mao).
--   * m_2C -> 15     -> morte. Carrega o checkpoint.

local mach = manager.machine
local root = mach.devices[":"]
local cpu  = mach.devices[":maincpu"]
local ram  = cpu.spaces["data"] or cpu.spaces["program"]

------------------------------------------------------------------ config
local LOGFILE      = "make_checkpoint.log"   -- relativo a pasta do projeto
local SEG_PEGAR    = "1.0.2"    -- desliga quando o Mario recebe a bomba
local SEG_ENTREGA  = "0.10.0"   -- liga quando a bomba e entregue
local SAVE_ON_EDGE = 3          -- flanco em que grava e congela
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

------------------------------------------------------------------ estado
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

  ---------------------------------------------------------------- morte
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

  ---------------------------------------------------------------- entrega
  if entrega and not prev_entrega then
    entregas = entregas + 1
    game_stopped = true
    log(string.format("[entrega %d] frame %d -- game_stopped = true",
                      entregas, frame))
  end

  ---------------------------------------------------------------- flanco 1.0.2
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
