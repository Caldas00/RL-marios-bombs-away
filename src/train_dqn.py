"""Double DQN para Mario's Bombs Away. Requer o lua/env_server.lua a correr no MAME.

Corre a partir da pasta do projeto (dois terminais):
    mame gnw_mbaway -autoboot_script lua/env_server.lua -autoboot_delay 0 \
         -window -nothrottle -sound none
    python src/train_dqn.py

Variaveis de ambiente (todas opcionais): MB_STEPS, MB_LR, MB_EPS_START, MB_EPS_END,
MB_EPS_DECAY, MB_WARMUP, MB_MAX_STEPS, MB_FRAME_STACK, MB_SHAPING, MB_CKPT_DIR,
MB_RESUME, MB_SEGNAMES. Ver o README.

Observacao: 36 segmentos x FRAME_STACK frames + nivel (RAM 0x0B) + fase do tick
(RAM 0x0C). Com FRAME_STACK = 1: 38 entradas. Com 4: 146.
Recompensa com shaping por potencial (posicao do Mario).
Com FORCE_NIVEL no Lua a velocidade e fixa; com nil, os resets sorteiam entre o
inicio e os checkpoints por nivel ja guardados.
"""

import os, random, collections
import numpy as np
import torch
import torch.nn as nn
from mbaway_env import MBAwayEnv
from segments import JOGO

# ---------------------------------------------------------------- config
BUFFER      = 100_000
BATCH       = 64
LR          = float(os.environ.get("MB_LR", 1e-4))
GAMMA       = 0.99
TARGET_SYNC = 1_000
WARMUP      = int(os.environ.get("MB_WARMUP", 5_000))
# Retomar: MB_RESUME=models/mario_dqn_final.pt  (usa MB_EPS_START/MB_EPS_END/MB_EPS_DECAY
# para um chao de exploracao mais baixo; o buffer recomeca vazio)
EPS_START = float(os.environ.get("MB_EPS_START", 1.0))
EPS_END   = float(os.environ.get("MB_EPS_END", 0.05))
EPS_DECAY = int(os.environ.get("MB_EPS_DECAY", 50_000))
RESUME    = os.environ.get("MB_RESUME", "")
MAX_STEPS = int(os.environ.get("MB_MAX_STEPS", 5000))   # limite de passos por episodio
TOTAL_STEPS = int(os.environ.get("MB_STEPS", 300_000))
CKPT_EVERY  = 10_000
FRAME_STACK = int(os.environ.get("MB_FRAME_STACK", 1))   # 1 = so a frame atual; 4 = como antes
SHAPING     = float(os.environ.get("MB_SHAPING", 0.05))   # 0 desliga o shaping
CKPT_DIR    = os.environ.get("MB_CKPT_DIR", "checkpoints")
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGNAMES    = os.environ.get("MB_SEGNAMES", os.path.join(ROOT, "data", "segnames.json"))

dev = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(n_in, n_out):
    return nn.Sequential(
        nn.Linear(n_in, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_out),
    )


def main():
    env = MBAwayEnv(segnames=SEGNAMES, include=JOGO, frame_stack=FRAME_STACK,
                    shaping=SHAPING, gamma=GAMMA)

    env.max_steps = MAX_STEPS
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n
    print(f"entrada da rede: {n_in} valores ({int(env.mask.sum())} segmentos x "
          f"{env.k} frame(s) + nivel + fase do tick) | shaping {SHAPING}")
    print("recompensa: +1 entrega | -1 morte | -0.01 por decisao | "
          f"+/-{SHAPING} por linha que o Mario avanca/recua")

    q = mlp(n_in, n_out).to(dev)
    tgt = mlp(n_in, n_out).to(dev)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=LR)

    if RESUME:
        ck = torch.load(RESUME, map_location=dev, weights_only=False)
        assert ck["n_in"] == n_in and (ck["mask"] == env.mask).all(), \
            "o checkpoint tem outra observacao (n_in/mascara): nao da para retomar"
        q.load_state_dict(ck["model"])
        tgt.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:
            g["lr"] = LR                     # LR pode ser mudado ao retomar
        print(f"retomado de {RESUME} (passo {ck['step']}) | lr {LR} | "
              f"eps {EPS_START} -> {EPS_END} em {EPS_DECAY} passos | buffer vazio")

    os.makedirs(CKPT_DIR, exist_ok=True)

    buf = collections.deque(maxlen=BUFFER)
    obs, info0 = env.reset()
    start_nivel, start_inicio = info0["start_nivel"], info0["start_inicio"]
    ep_entregas, ep_ret, ep = 0, 0.0, 0
    hist_inicio = collections.deque(maxlen=50)   # so episodios que comecam no inicio
    hist_niv = collections.defaultdict(lambda: collections.deque(maxlen=50))
    melhor = -1.0
    descartados = 0
    truncados = 0        # episodios cortados pelo limite de passos (guardados, sem fim)
    ep_buf = []          # transicoes deste episodio, so commit se for valido
    acoes = np.zeros(n_out, dtype=np.int64)   # contagem de acoes desde o ultimo log
    NOMES_ACAO = ["nada", "esq", "dir", "botao"]

    def guardar(nome, step):
        torch.save({"model": q.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "mask": env.mask,
                    "n_in": n_in,
                    "frame_stack": FRAME_STACK}, os.path.join(CKPT_DIR, nome))

    def novo_episodio():
        nonlocal obs, start_nivel, start_inicio, ep_entregas, ep_ret
        obs, i = env.reset()
        start_nivel, start_inicio = i["start_nivel"], i["start_inicio"]
        ep_entregas, ep_ret = 0, 0.0

    for step in range(TOTAL_STEPS):
        eps = max(EPS_END, EPS_START - step * (EPS_START - EPS_END) / EPS_DECAY)

        if random.random() < eps:
            a = random.randrange(n_out)
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(obs, device=dev)).argmax())

        nobs, r, done, trunc, info = env.step(a)
        acoes[a] += 1
        if (step + 1) % 5000 == 0:
            pct = 100 * acoes / max(1, acoes.sum())
            print(f"   [acoes, ultimos 5000 passos | eps {eps:.2f}] " +
                  " ".join(f"{NOMES_ACAO[i]} {pct[i]:.0f}%" for i in range(n_out)))
            acoes[:] = 0
        ep_buf.append((obs, a, r, nobs, done))
        obs = nobs
        ep_ret += r
        ep_entregas += int(info["entrega"])

        # ------------------------------------------------ episodio invalido
        if info["bug"]:
            descartados += 1
            print(f"   [DESCARTADO #{descartados}] step {step} | "
                  f"{len(ep_buf)} transicoes deitadas fora | "
                  f"{ep_entregas} entregas perdidas")
            ep_buf.clear()
            novo_episodio()
            continue

        if trunc and not done:
            # Limite de passos, NAO e morte: o Mario ainda estava vivo. Guarda as
            # transicoes (a ultima tem done=False, por isso a rede continua a
            # fazer bootstrap do estado seguinte) e comeca outro episodio.
            truncados += 1
            buf.extend(ep_buf)
            hist_niv[start_nivel].append(ep_entregas)
            if start_inicio:
                hist_inicio.append(ep_entregas)
            print(f"   [TRUNCADO #{truncados}] step {step} | {len(ep_buf)} transicoes "
                  f"guardadas | {ep_entregas} entregas (limite {MAX_STEPS} passos)")
            ep_buf.clear()
            novo_episodio()
            continue

        if done:
            buf.extend(ep_buf)                 # so agora entra no buffer
            ep_buf.clear()
            hist_niv[start_nivel].append(ep_entregas)
            if start_inicio:                   # episodio que comecou no checkpoint do inicio
                hist_inicio.append(ep_entregas)
            ep += 1

            # best.pt: medido so em episodios que comecam no inicio, para ser
            # comparavel com o baseline (os checkpoints de nivel sao mais dificeis)
            if len(hist_inicio) == hist_inicio.maxlen:
                media = float(np.mean(hist_inicio))
                if media > melhor:
                    melhor = media
                    guardar("best.pt", step)

            if ep % 10 == 0:
                por_niv = " ".join(f"n{k}:{np.mean(v):.2f}({len(v)})"
                                   for k, v in sorted(hist_niv.items(), reverse=True))
                m_ini = np.mean(hist_inicio) if hist_inicio else 0.0
                print(f"ep {ep:5d} | step {step:7d} | eps {eps:.3f} | "
                      f"entregas inicio(50) {m_ini:.2f} | melhor {melhor:.2f} | "
                      f"por nivel [{por_niv}] | retorno {ep_ret:.2f} | "
                      f"descartados {descartados} | truncados {truncados}")
            novo_episodio()

        # -------------------------------------------------------- treino
        if len(buf) < WARMUP:
            continue

        batch = random.sample(buf, BATCH)
        o, a_, r_, no, d = map(np.array, zip(*batch))
        o  = torch.as_tensor(o,  device=dev)
        no = torch.as_tensor(no, device=dev)
        a_ = torch.as_tensor(a_, device=dev, dtype=torch.int64).unsqueeze(1)
        r_ = torch.as_tensor(r_, device=dev, dtype=torch.float32)
        d  = torch.as_tensor(d,  device=dev, dtype=torch.float32)

        qv = q(o).gather(1, a_).squeeze(1)
        with torch.no_grad():
            # Double DQN: a rede online escolhe a acao, a target avalia-a
            a_next = q(no).argmax(1, keepdim=True)
            tv = r_ + GAMMA * (1 - d) * tgt(no).gather(1, a_next).squeeze(1)

        loss = nn.functional.smooth_l1_loss(qv, tv)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()

        if step % TARGET_SYNC == 0:
            tgt.load_state_dict(q.state_dict())

        if step % CKPT_EVERY == 0 and step > 0:
            guardar(f"ckpt_{step}.pt", step)
            guardar("latest.pt", step)
            print(f"   [guardado] step {step}")

    guardar("final.pt", TOTAL_STEPS)
    print(f"terminado. melhor media de entregas (inicio): {melhor:.2f}")
    print(f"modelos em {CKPT_DIR}/  (best.pt, latest.pt, final.pt)")


if __name__ == "__main__":
    main()
