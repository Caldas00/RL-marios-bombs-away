-- poke_speed.lua -- forces the speed level (RAM 0x0B) and measures the m_09 tick
--
--   mame gnw_mbaway -autoboot_script lua/poke_speed.lua -autoboot_delay 0 -window
--
-- Change LEVEL below (5 = normal at the start, 4, 3, 2, 1, 0 = faster?)
-- Play normally (no invincibility). After FRAME_START, RAM 0x0B is forced
-- to LEVEL every frame. The terminal shows the average interval of the m_09
-- tick (frames per tick): if the poke works, it should drop as predicted
-- (5 ~ 27, 4 ~ 22, 3 ~ 17.5) and the game should become visibly faster.

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
