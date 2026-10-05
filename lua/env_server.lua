-- env_server.lua -- MAME <-> Python bridge for Mario's Bombs Away (Game B)
--
--   mame gnw_mbaway -autoboot_script lua/env_server.lua -autoboot_delay 0 \
--        -window -nothrottle -sound none
--
-- THE RESET HAPPENS HERE, in the same frame the death is detected.
-- Python is only told about it (death=1); it doesn't have to send anything back.
--
-- Protocol:
--   Python -> Lua :  "RESET" | "STEP <0-3>"
--   Lua -> Python :  "<bits>;<death>;<delivery>;<bug>;<level>;<phase>;<start>"
--
-- v3: <level> = RAM 0x0B (speed level), <phase> = RAM 0x0C (counter that goes
-- down from <level> to 0 between two ticks: tells how long until the next tick),
-- <start> = 1 if the last load was the start checkpoint (slot "1").
-- 0x0B only changes at the end of the delivery animation. The first time each
-- level (NIVEL_MIN .. NIVEL_INICIO-1) appears, the state is saved to "lv<N>".
-- On reset, the slot is picked at random between the start checkpoint (SLOT) and
-- the levels already saved. To regenerate the per-level checkpoints, delete lv*.sta.

local mach = manager.machine
local root = mach.devices[":"]
local cpu  = mach.devices[":maincpu"]
local ram  = cpu.spaces["data"] or cpu.spaces["program"]

------------------------------------------------------------------ config
local SEGNAMES    = "data/segnames.json"       -- relative to the project folder
local FIFO_IN     = "/tmp/mbaway_in"
local FIFO_OUT    = "/tmp/mbaway_out"
local SLOT        = "1"
local STA_DIR     = "sta/gnw_mbaway/"          -- MAME state_directory (relative)

local ADDR_NIVEL   = 0x0B     -- speed level (confirmed by poke)
local ADDR_FASE    = 0x0C     -- tick counter (goes down from ADDR_NIVEL to 0)
local ADDR_ANIM    = 0x5A     -- 0 when the delivery animation has finished
local NIVEL_INICIO = 5        -- level of the start checkpoint
local NIVEL_MIN    = 1        -- 0 is the value before the game starts: don't save
local P_INICIO     = 0.5      -- prob. that a reset goes to the start (if there are levels)

-- FIXED speed: if != nil, RAM 0x0B is forced to this value every frame
-- (the poke we confirmed controls the real speed). With this on there are no
-- per-level checkpoints and every reset goes to the start. Set to nil to disable.
local FORCE_NIVEL  = 3        -- set to nil to free the speed (per-level checkpoints)

local SEG_PEGAR   = "1.0.2"
local SEG_ENTREGA = "0.10.0"
local SEG_BUG     = "1.15.1"    -- lights up in the death cutscene: detection missed it
local ADDR_MORTE  = 0x2C
local VALOR_MORTE = 15

local ACTION_REPEAT = 8
local MAX_SKIP      = 300
local LOAD_FRAMES   = 12      -- same as the cooldown in make_checkpoint.lua

------------------------------------------------------------------ segments
local NAMES = {}
for line in io.lines(SEGNAMES) do
  local n = line:match('^"([%d%.]+)",?$')
  if n then NAMES[#NAMES+1] = n end
end

local SEGP, P_PEGAR, P_ENTREGA, P_BUG = nil, nil, nil, nil
local function build_proxies()
  local t = {}
  for i, n in ipairs(NAMES) do
    local p = root:output(n)
    if not (p and p:exists()) then return false end
    t[i] = p
  end
  SEGP      = t
  P_PEGAR   = root:output(SEG_PEGAR)
  P_ENTREGA = root:output(SEG_ENTREGA)
  P_BUG     = root:output(SEG_BUG)
  return true
end

------------------------------------------------------------------ inputs
local F = {}
for _, port in pairs(mach.ioport.ports) do
  for fname, field in pairs(port.fields) do
    if     fname:find("Left")     then F.left  = field
    elseif fname:find("Right")    then F.right = field
    elseif fname:find("Button 1") then F.btn   = field end
  end
end
assert(F.left and F.right and F.btn, "inputs nao encontrados")

local function clear_inputs() F.left:set_value(0); F.right:set_value(0); F.btn:set_value(0) end
local function apply(a)
  clear_inputs()
  if     a == 1 then F.left:set_value(1)
  elseif a == 2 then F.right:set_value(1)
  elseif a == 3 then F.btn:set_value(1) end
end

------------------------------------------------------------------ state
local prev_ram, prev_pegar, prev_entrega, prev_bug = -1, false, false, false
local game_stopped = false
local acc_morte, acc_entrega, acc_bug = false, false, false
local mortes, resets_falhados, bugs = 0, 0, 0

math.randomseed(os.time())

-- per-level checkpoints already on disk
local saved, tried, cargas = {}, {}, {}
for L = NIVEL_MIN, NIVEL_INICIO - 1 do
  local fh = io.open(STA_DIR .. "lv" .. L .. ".sta", "rb")
  if fh then fh:close(); saved[L] = true end
end

local function pick_slot()
  if FORCE_NIVEL then return SLOT end
  local avail = {}
  for L = NIVEL_MIN, NIVEL_INICIO - 1 do
    if saved[L] then avail[#avail + 1] = L end
  end
  if #avail == 0 or math.random() < P_INICIO then return SLOT end
  return "lv" .. avail[math.random(#avail)]
end

local ultimo_inicio = true    -- was the last load the start checkpoint?

local function do_load()
  local slot = pick_slot()
  ultimo_inicio = (slot == SLOT)
  cargas[slot] = (cargas[slot] or 0) + 1
  local ok, err = pcall(function() mach:load(slot) end)
  if not ok then
    resets_falhados = resets_falhados + 1
    print("!! load falhou (" .. resets_falhados .. "): " .. tostring(err))
  end
  return ok
end

-- returns true if it detected a death (and has already loaded the checkpoint)
local function sample()
  local pegar   = (P_PEGAR:get()   ~= 0)
  local entrega = (P_ENTREGA:get() ~= 0)
  local bug     = (P_BUG:get()     ~= 0)
  local v       = ram:read_u8(ADDR_MORTE)

  local morreu  = (v == VALOR_MORTE and prev_ram ~= VALOR_MORTE)
  local bugou   = (bug and not prev_bug)

  if entrega and not prev_entrega then
    acc_entrega  = true
    game_stopped = true
  end
  if prev_pegar and not pegar then game_stopped = false end

  prev_ram, prev_pegar, prev_entrega, prev_bug = v, pegar, entrega, bug

  -- the death cutscene was reached: the m_2C detection missed it.
  -- reset and mark the episode as invalid.
  if bugou then
    acc_bug = true
    bugs = bugs + 1
    do_load()
    print(string.format("!! BUG #%d -- %s acendeu, episodio descartado", bugs, SEG_BUG))
    return true
  end

  if morreu then
    acc_morte = true
    mortes = mortes + 1
    do_load()                      -- IMMEDIATE RESET, in the same frame
    if mortes % 50 == 0 then
      local partes = {}
      for k, n in pairs(cargas) do partes[#partes + 1] = k .. "=" .. n end
      table.sort(partes)
      print(string.format(">>> %d mortes | %d resets falhados | cargas: %s",
            mortes, resets_falhados, table.concat(partes, " ")))
    end
    return true
  end
  return false
end

-- saves the checkpoint the first time a new level appears.
-- returns true if it requested a save (MAME's save happens at the end of the frame).
local function maybe_save_level()
  if FORCE_NIVEL then return false end
  local v = ram:read_u8(ADDR_NIVEL)
  if v < NIVEL_MIN or v >= NIVEL_INICIO or saved[v] or tried[v] then return false end
  if game_stopped or ram:read_u8(ADDR_ANIM) ~= 0 then return false end
  if ram:read_u8(ADDR_MORTE) == VALOR_MORTE or (P_BUG:get() ~= 0) then return false end
  tried[v] = true
  local ok, err = pcall(function() mach:save("lv" .. v) end)
  if ok then
    saved[v] = true
    print(string.format(">>> checkpoint do nivel %d guardado (lv%d)", v, v))
  else
    print("!! save do nivel " .. v .. " falhou: " .. tostring(err))
  end
  return true
end

local function obs()
  local b = {}
  for i = 1, #SEGP do b[i] = (SEGP[i]:get() ~= 0) and "1" or "0" end
  return table.concat(b)
end

------------------------------------------------------------------ FIFO
os.execute("rm -f " .. FIFO_IN .. " " .. FIFO_OUT)
os.execute("mkfifo " .. FIFO_IN .. " " .. FIFO_OUT)
local fout = io.open(FIFO_OUT, "w"); fout:setvbuf("line")
local fin  = io.open(FIFO_IN,  "r")

------------------------------------------------------------------ phases
local IDLE, STEP, SKIP, SETTLE = 0, 1, 2, 3
local phase, left, skipped = IDLE, 0, 0

local function reply()
  fout:write(string.format("%s;%d;%d;%d;%d;%d;%d\n", obs(),
             acc_morte and 1 or 0, acc_entrega and 1 or 0, acc_bug and 1 or 0,
             ram:read_u8(ADDR_NIVEL), ram:read_u8(ADDR_FASE),
             ultimo_inicio and 1 or 0))
  acc_morte, acc_entrega, acc_bug = false, false, false
end

local function settle_after_load()
  phase, left = SETTLE, LOAD_FRAMES
end

local function on_frame()
  if not SEGP then
    if build_proxies() then
      -- initialize the prev_* with the REAL values, otherwise the first sample
      -- sees phantom edges and game_stopped gets stuck
      prev_ram     = ram:read_u8(ADDR_MORTE)
      prev_pegar   = (P_PEGAR:get()   ~= 0)
      prev_entrega = (P_ENTREGA:get() ~= 0)
      prev_bug     = (P_BUG:get()     ~= 0)
      game_stopped = false
      print(">>> proxies prontos: " .. #SEGP ..
            " | pegar=" .. tostring(prev_pegar) ..
            " entrega=" .. tostring(prev_entrega) ..
            " m2C=" .. prev_ram)
    end
    return
  end

  if FORCE_NIVEL then ram:write_u8(ADDR_NIVEL, FORCE_NIVEL) end

  ---------------------------------------------------------------- step
  if phase == STEP then
    local morreu = sample()
    left = left - 1
    if morreu then
      clear_inputs()
      settle_after_load()
    elseif left <= 0 then
      clear_inputs()
      if game_stopped then phase, skipped = SKIP, 0
      else phase = IDLE; reply() end
    end
    return
  end

  ---------------------------------------------------------------- skip
  if phase == SKIP then
    local morreu = sample()
    skipped = skipped + 1
    if morreu then
      settle_after_load()
    elseif (not game_stopped) or skipped >= MAX_SKIP then
      if skipped >= MAX_SKIP then
        print("!! MAX_SKIP atingido -- a libertar game_stopped")
        game_stopped = false        -- don't get stuck in a loop
      end
      phase = IDLE; reply()
    end
    return
  end

  ---------------------------------------------------------------- post-load
  if phase == SETTLE then
    left = left - 1
    if left <= 0 then
      prev_ram     = ram:read_u8(ADDR_MORTE)
      prev_pegar   = (P_PEGAR:get()   ~= 0)
      prev_entrega = (P_ENTREGA:get() ~= 0)
      prev_bug     = (P_BUG:get()     ~= 0)
      game_stopped = false
      phase = IDLE
      reply()                       -- acc_morte is still 1: Python ends the episode
    end
    return
  end

  ---------------------------------------------------------------- command
  if maybe_save_level() then return end   -- 1-frame gap after the save

  local cmd = fin:read("l")
  if not cmd then return end

  if cmd == "RESET" then
    do_load()
    settle_after_load()
  elseif cmd:sub(1, 4) == "STEP" then
    apply(tonumber(cmd:sub(6)) or 0)
    phase, left = STEP, ACTION_REPEAT
  end
end

_G.notifier = emu.add_machine_frame_notifier(on_frame)
mach.video.throttled = false
local ja = {}
for L = NIVEL_MIN, NIVEL_INICIO - 1 do if saved[L] then ja[#ja + 1] = "lv" .. L end end
if FORCE_NIVEL then print(">>> VELOCIDADE FIXA: 0x0B := " .. FORCE_NIVEL) end
print(">>> env_server pronto | segmentos=" .. #NAMES ..
      " | checkpoints de nivel: " .. (#ja > 0 and table.concat(ja, ",") or "nenhum"))
