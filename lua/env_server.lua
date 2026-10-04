-- env_server.lua -- ponte MAME <-> Python para Mario's Bombs Away (Game B)
--
--   mame gnw_mbaway -autoboot_script lua/env_server.lua -autoboot_delay 0 \
--        -window -nothrottle -sound none
--
-- O RESET E FEITO AQUI, no mesmo frame em que a morte e detetada.
-- O Python so e informado ("morte=1"); nao tem de mandar nada de volta.
--
-- Protocolo:
--   Python -> Lua :  "RESET" | "STEP <0-3>"
--   Lua -> Python :  "<bits>;<morte>;<entrega>;<bug>;<nivel>;<fase>;<inicio>"
--
-- v3: <nivel> = RAM 0x0B (nivel de velocidade), <fase> = RAM 0x0C (contador que
-- desce de <nivel> ate 0 entre dois ticks: diz quanto falta para o proximo tick),
-- <inicio> = 1 se o ultimo load foi o checkpoint do inicio (slot "1").
-- O 0x0B so muda no fim da animacao de entrega. A primeira vez que cada
-- nivel (NIVEL_MIN .. NIVEL_INICIO-1) aparece, o estado e guardado em "lv<N>".
-- Nos resets escolhe-se ao acaso entre o checkpoint do inicio (SLOT) e os niveis
-- ja guardados. Para regenerar os checkpoints por nivel, apaga os lv*.sta.

local mach = manager.machine
local root = mach.devices[":"]
local cpu  = mach.devices[":maincpu"]
local ram  = cpu.spaces["data"] or cpu.spaces["program"]

------------------------------------------------------------------ config
local SEGNAMES    = "data/segnames.json"       -- relativo a pasta do projeto
local FIFO_IN     = "/tmp/mbaway_in"
local FIFO_OUT    = "/tmp/mbaway_out"
local SLOT        = "1"
local STA_DIR     = "sta/gnw_mbaway/"          -- state_directory do MAME (relativo)

local ADDR_NIVEL   = 0x0B     -- nivel de velocidade (confirmado por poke)
local ADDR_FASE    = 0x0C     -- contador do tick (desce de ADDR_NIVEL ate 0)
local ADDR_ANIM    = 0x5A     -- 0 quando a animacao de entrega acabou
local NIVEL_INICIO = 5        -- nivel do checkpoint do inicio
local NIVEL_MIN    = 1        -- 0 e o valor antes do jogo arrancar: nao guardar
local P_INICIO     = 0.5      -- prob. de um reset ir para o inicio (se houver niveis)

-- Velocidade FIXA: se != nil, a RAM 0x0B e forcada a este valor a cada frame
-- (o poke que confirmamos que controla a velocidade real). Com isto ligado nao ha
-- checkpoints por nivel e todos os resets vao para o inicio. Poe nil para desligar.
local FORCE_NIVEL  = 3        -- poe nil para libertar a velocidade (checkpoints por nivel)

local SEG_PEGAR   = "1.0.2"
local SEG_ENTREGA = "0.10.0"
local SEG_BUG     = "1.15.1"    -- acende na cutscene de morte: deteccao escapou
local ADDR_MORTE  = 0x2C
local VALOR_MORTE = 15

local ACTION_REPEAT = 8
local MAX_SKIP      = 300
local LOAD_FRAMES   = 12      -- igual ao cooldown do make_checkpoint.lua

------------------------------------------------------------------ segmentos
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

------------------------------------------------------------------ estado
local prev_ram, prev_pegar, prev_entrega, prev_bug = -1, false, false, false
local game_stopped = false
local acc_morte, acc_entrega, acc_bug = false, false, false
local mortes, resets_falhados, bugs = 0, 0, 0

math.randomseed(os.time())

-- checkpoints por nivel ja existentes em disco
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

local ultimo_inicio = true    -- o ultimo load foi o checkpoint do inicio?

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

-- devolve true se detetou morte (e ja carregou o checkpoint)
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

  -- a cutscene de morte foi alcancada: a deteccao por m_2C escapou.
  -- reinicia e marca o episodio como invalido.
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
    do_load()                      -- RESET IMEDIATO, no proprio frame
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

-- guarda o checkpoint da primeira vez que um nivel novo aparece.
-- devolve true se pediu um save (o save do MAME acontece no fim do frame).
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

------------------------------------------------------------------ fases
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
      -- inicializa os prev_* com os valores REAIS, senao o primeiro sample
      -- ve flancos fantasma e o game_stopped fica preso
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

  ---------------------------------------------------------------- passo
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

  ---------------------------------------------------------------- salto
  if phase == SKIP then
    local morreu = sample()
    skipped = skipped + 1
    if morreu then
      settle_after_load()
    elseif (not game_stopped) or skipped >= MAX_SKIP then
      if skipped >= MAX_SKIP then
        print("!! MAX_SKIP atingido -- a libertar game_stopped")
        game_stopped = false        -- nao ficar preso em ciclo
      end
      phase = IDLE; reply()
    end
    return
  end

  ---------------------------------------------------------------- pos-load
  if phase == SETTLE then
    left = left - 1
    if left <= 0 then
      prev_ram     = ram:read_u8(ADDR_MORTE)
      prev_pegar   = (P_PEGAR:get()   ~= 0)
      prev_entrega = (P_ENTREGA:get() ~= 0)
      prev_bug     = (P_BUG:get()     ~= 0)
      game_stopped = false
      phase = IDLE
      reply()                       -- acc_morte segue a 1: o Python fecha o episodio
    end
    return
  end

  ---------------------------------------------------------------- comando
  if maybe_save_level() then return end   -- 1 frame de intervalo apos o save

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
