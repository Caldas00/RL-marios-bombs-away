"""Avalia um modelo em modo greedy (epsilon = 0) e imprime a media de entregas.

Corre a partir da pasta do projeto. Sem -nothrottle ves o jogo a velocidade real;
com -nothrottle a avaliacao e ~8x mais rapida.
    mame gnw_mbaway -autoboot_script lua/env_server.lua -autoboot_delay 0 -window
    MB_EP=30 python src/play.py [models/mario_dqn_final.pt]

MB_EP: numero de episodios (10). MB_MAX_STEPS: limite de passos por episodio (20000).
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from mbaway_env import MBAwayEnv
from segments import JOGO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGNAMES = os.environ.get("MB_SEGNAMES", os.path.join(ROOT, "data", "segnames.json"))
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "mario_dqn_final.pt")
EPISODIOS = int(os.environ.get("MB_EP", 10))


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, n_out))


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
env = MBAwayEnv(segnames=SEGNAMES, include=JOGO,
                frame_stack=ck.get("frame_stack", 1), shaping=0.0)
env.max_steps = int(os.environ.get("MB_MAX_STEPS", 20000))   # sem o corte dos 2000
assert env.mask.sum() == ck["mask"].sum() and (env.mask == ck["mask"]).all(), \
    "a mascara do checkpoint nao bate com a do ambiente"
assert env.observation_space.shape[0] == ck["n_in"], "n_in diferente do checkpoint"

q = mlp(ck["n_in"], env.action_space.n)
q.load_state_dict(ck["model"])
q.eval()
print(f"modelo: {CKPT} (step {ck['step']})")

todas = []
for ep in range(EPISODIOS):
    obs, info = env.reset()
    entregas, passos, done = 0, 0, False
    acoes = np.zeros(env.action_space.n, dtype=int)
    while not done:
        with torch.no_grad():
            a = int(q(torch.as_tensor(obs)).argmax())
        acoes[a] += 1
        obs, r, done, trunc, info = env.step(a)
        entregas += int(info["entrega"])
        passos += 1
        if trunc:
            break
    todas.append(entregas)
    print(f"ep {ep + 1}: comecou no nivel {env.start_nivel} | {entregas} entregas | "
          f"{passos} passos | nivel final {info['nivel']} | acoes nada/esq/dir/botao "
          f"{'/'.join(str(x) for x in acoes)}")
m = float(np.mean(todas))
print(f"\nMEDIA: {m:.2f} entregas/episodio em {len(todas)} episodios "
      f"(desvio {np.std(todas):.2f}, erro tipico {np.std(todas)/np.sqrt(len(todas)):.2f}) "
      f"-> p por bomba ~ {m/(1+m):.2f}")
env.close()
