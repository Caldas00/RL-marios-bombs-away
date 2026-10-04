-- dump_ram.lua -- grava, a cada frame, os 131 segmentos do LCD + os 128 enderecos
-- de RAM do SM511. Foi com isto que se encontrou a velocidade (0x0B) e a fase do
-- tick (0x0C). Joga tu normalmente enquanto grava.
--
--   mame gnw_mbaway -autoboot_script lua/dump_ram.lua -autoboot_delay 0 -window
--
-- Escreve em research/data/ram_dump.csv  (colunas: frame, s_<segmento>..., m_00..m_7F)

local mach = manager.machine
local root = mach.devices[":"]
local cpu  = mach.devices[":maincpu"]
local ram  = cpu.spaces["data"] or cpu.spaces["program"]

local NAMES = {}
for line in io.lines("data/segnames.json") do
  local n = line:match('^"([%d%.]+)",?$')
  if n then NAMES[#NAMES+1] = n end
end

local f = io.open("research/data/ram_dump.csv", "w")
f:setvbuf("full")

local P, frame, started = {}, 0, false

local function start()
  for i, n in ipairs(NAMES) do
    local p = root:output(n)
    if not (p and p:exists()) then return false end
    P[i] = p
  end
  local h = { "frame" }
  for _, n in ipairs(NAMES) do h[#h+1] = "s_" .. n end
  for a = 0, 127 do h[#h+1] = string.format("m_%02X", a) end
  f:write(table.concat(h, ",") .. "\n")
  print(">>> speed dump ativo: " .. #P .. " segmentos + 128 RAM")
  return true
end

local function tick()
  if not started then started = start(); return end
  local row = { frame }
  for i = 1, #P do row[#row+1] = (P[i]:get() ~= 0) and 1 or 0 end
  for a = 0, 127 do row[#row+1] = ram:read_u8(a) end
  f:write(table.concat(row, ",") .. "\n")
  frame = frame + 1
  if frame % 600 == 0 then f:flush() end
end

_G.notifier = emu.add_machine_frame_notifier(tick)
pcall(function() emu.add_machine_stop_notifier(function() f:flush(); f:close() end) end)
