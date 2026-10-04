-- poke_speed.lua -- forca o nivel de velocidade (RAM 0x0B) e mede o tick m_09
--
--   mame gnw_mbaway -autoboot_script lua/poke_speed.lua -autoboot_delay 0 -window
--
-- Muda LEVEL abaixo (5 = normal do inicio, 4, 3, 2, 1, 0 = mais rapido?)
-- Joga normalmente (sem invencibilidade). Depois de FRAME_START a RAM 0x0B e
-- forcada a LEVEL a cada frame. No terminal aparece o intervalo medio do tick
-- m_09 (frames por tick): se o poke funciona, tem de baixar como previsto
-- (5 ~ 27, 4 ~ 22, 3 ~ 17.5) e o jogo tem de ficar visivelmente mais rapido.

local LEVEL       = 3
local FRAME_START = 300

local mach = manager.machine
local cpu  = mach.devices[":maincpu"]
local ram  = cpu.spaces["data"] or cpu.spaces["program"]

local frame, last_val, last_t, acc, n = 0, nil, nil, 0, 0

local function tick()
  frame = frame + 1
  if frame >= FRAME_START then ram:write_u8(0x0B, LEVEL) end
  local v = ram:read_u8(0x09)
  if last_val ~= nil and v ~= last_val and last_t then
    local d = frame - last_t
    if d < 100 then acc = acc + d; n = n + 1 end
    last_t = frame
    if n == 15 then
      print(string.format("m_0B=%d  intervalo m_09 medio (15 ticks) = %.1f frames",
            ram:read_u8(0x0B), acc / n))
      acc, n = 0, 0
    end
  elseif last_val ~= nil and v ~= last_val then
    last_t = frame
  end
  if last_t == nil and last_val ~= nil and v ~= last_val then last_t = frame end
  last_val = v
end

_G.notifier = emu.add_machine_frame_notifier(tick)
print(">>> poke_speed ativo: 0x0B := " .. LEVEL .. " a partir do frame " .. FRAME_START)
