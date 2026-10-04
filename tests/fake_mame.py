"""Testa o pipeline SEM o MAME: corre o lua/env_server.lua verdadeiro num
interpretador Lua (lupa) com um MAME falso (segmentos, RAM e inputs simulados),
e liga-o ao Python pelos mesmos FIFOs. Serve para verificar que o protocolo, os
resets, o treino e o play funcionam depois de mexer no codigo.

Corre a partir da pasta do projeto:
    pip install lupa
    python tests/fake_mame.py train      # treino curto (2000 passos)
    python tests/fake_mame.py play       # avalia models/mario_dqn_final.pt (3 ep.)

O jogo simulado nao tem nada a ver com o real (o Mario anda sozinho e morre
de 300 em 300 frames): os numeros que aparecem nao significam nada.
"""
import os
import sys
import threading
import time

from lupa import LuaRuntime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                          # o Lua usa caminhos relativos a raiz
sys.path.insert(0, os.path.join(ROOT, "src"))
MODE = sys.argv[1] if len(sys.argv) > 1 else "train"

MOCK = r'''
sim = {frame=0, level=5, m2c=0, saved={}, loads={}}
local MARIOIDX = {["1.1.1"]=0,["1.1.2"]=1,["0.3.1"]=2,["0.3.2"]=3,["1.4.1"]=4,
                  ["1.4.2"]=5,["0.6.1"]=6,["0.6.2"]=7,["0.8.1"]=8,["1.7.2"]=9}
local function proxy(n) return {exists=function() return true end, get=function()
  if n=="1.15.1" or n=="1.0.2" or n=="0.10.0" then return 0 end
  local mi = MARIOIDX[n]
  if mi then return ((math.floor(sim.frame/40) % 10) == mi) and 1 or 0 end
  return ((sim.frame + #n*7) % 5 == 0) and 1 or 0 end} end
local ram = {
  read_u8=function(self, a)
    if a==0x2C then return sim.m2c elseif a==0x0B then return sim.level
    elseif a==0x0C then return 3 - (sim.frame % 4) else return 0 end end,
  write_u8=function(self, a, v) if a==0x0B then sim.level=v end end}
local field = function() return {set_value=function() end} end
local mach = {devices={[":"]={output=function(self, n) return proxy(n) end},
                       [":maincpu"]={spaces={data=ram}}},
              ioport={ports={p={fields={["Left"]=field(), ["Right"]=field(),
                                        ["Button 1"]=field()}}}},
              video={throttled=true}}
function mach:load(slot) sim.loads[#sim.loads+1]=slot; sim.m2c=0 end
function mach:save(slot) sim.saved[slot]=true end
manager = {machine=mach}
emu = {add_machine_frame_notifier=function(f) NOTIF=f end}
'''

lua = LuaRuntime(unpack_returned_tuples=True)
lua.execute(MOCK)
G = lua.globals()


def run_lua():
    lua.execute(open("lua/env_server.lua").read())
    while True:                         # cada chamada = 1 frame do MAME
        G.sim.frame = G.sim.frame + 1
        G.sim.m2c = 15 if G.sim.frame % 300 == 0 else 0
        G.NOTIF()


threading.Thread(target=run_lua, daemon=True).start()
time.sleep(1.0)

if MODE == "train":
    os.environ.setdefault("MB_STEPS", "2000")
    os.environ.setdefault("MB_WARMUP", "300")
    os.environ.setdefault("MB_CKPT_DIR", "/tmp/mbaway_fake_ckpt")
    import train_dqn
    train_dqn.main()
else:
    os.environ.setdefault("MB_EP", "3")
    sys.argv = ["play.py"] + sys.argv[2:]
    exec(open("src/play.py").read(), {"__name__": "__main__", "__file__": "src/play.py"})
print(f"OK ({MODE}) | resets no Lua: {len(G.sim.loads)}")
os._exit(0)
