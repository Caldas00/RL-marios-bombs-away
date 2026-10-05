"""Gymnasium environment for Mario's Bombs Away (Game & Watch, 1983) via MAME.

Observation (v3): whitelisted segments x frame_stack  +  2 scalars:
  - speed level (RAM 0x0B, normalized)
  - tick phase (RAM 0x0C / 5): how long until the next tick
With the 36 GAME segments and frame_stack=1 that's 36 + 2 = 38 values.

Reward: +1 delivery, -1 death, -0.01 per step, plus potential-based shaping
    F = gamma * phi(s') - phi(s),   phi = shaping * (Mario's row, 0..4)
which does NOT change the optimal policy (Ng, Harada, Russell 1999): it only speeds up the start.
"""

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from segments import MARIO

FIFO_IN, FIFO_OUT = "/tmp/mbaway_in", "/tmp/mbaway_out"

# speed level (RAM 0x0B): 5 = slow (start), lower = faster.
NIVEL_LENTO, NIVEL_RAPIDO = 5, 1
FASE_MAX = 5.0        # 0x0C goes down from 0x0B (<=5) to 0


def nivel_norm(n):
    n = min(max(int(n), NIVEL_RAPIDO), NIVEL_LENTO)
    return (NIVEL_LENTO - n) / (NIVEL_LENTO - NIVEL_RAPIDO)


class MBAwayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame_stack=1, noop_max=30, include=None, segnames=None,
                 shaping=0.05, gamma=0.99):
        """include: list of segments that GO INTO the network (everything else is left out).
        segnames: path to segnames.json.
        shaping: weight of the potential (0 disables it). gamma: the one used for training.

        The death, delivery and bug detectors and the checkpoints live in
        env_server.lua and don't go through the network."""
        super().__init__()
        self.fin = open(FIFO_OUT, "r")
        self.fout = open(FIFO_IN, "w")
        self.k = frame_stack
        self.noop_max = noop_max
        self.max_steps = 2000
        self.t = 0
        self.shaping, self.gamma = shaping, gamma
        self.nivel, self.fase, self.inicio = NIVEL_LENTO, 0, True
        self.start_nivel, self.start_inicio = NIVEL_LENTO, True
        self.phi = 0.0

        self.names = json.load(open(segnames)) if segnames else None

        raw = self._cmd("STEP 0")[0]
        self.n_raw = len(raw)
        self.mask = np.ones(self.n_raw, dtype=bool)
        self._apply_selection(include)
        self.mario_idx = ([self.names.index(s) for s in MARIO]
                          if self.names else None)

        self.action_space = spaces.Discrete(4)        # 0 nothing 1 left 2 right 3 button
        n = int(self.mask.sum()) * self.k + 2
        self.observation_space = spaces.Box(0.0, 1.0, (n,), np.float32)

    # ------------------------------------------------------------ bridge
    def _cmd(self, s):
        self.fout.write(s + "\n")
        self.fout.flush()
        campos = self.fin.readline().strip().split(";")
        if len(campos) != 7:
            raise RuntimeError(
                f"resposta com {len(campos)} campos (esperava 7): o env_server.lua "
                f"tem de ser a versao v3 e o MAME tem de estar a correr")
        bits, morte, entrega, bug, nivel, fase, inicio = campos
        raw = np.frombuffer(bits.encode(), dtype=np.uint8) - ord("0")
        self.nivel, self.fase, self.inicio = int(nivel), int(fase), inicio == "1"
        return raw.astype(np.float32), morte == "1", entrega == "1", bug == "1"

    def _apply_selection(self, include):
        if not include:
            return
        if self.names is None:
            raise ValueError("include precisa de segnames=<caminho do json>")
        if len(self.names) != self.n_raw:
            raise ValueError(f"segnames tem {len(self.names)}, o servidor manda {self.n_raw}")
        keep = set(include)
        desconhecidos = keep - set(self.names)
        if desconhecidos:
            raise ValueError(f"nomes inexistentes: {sorted(desconhecidos)}")
        self.mask = np.array([nm in keep for nm in self.names], dtype=bool)
        print(f"lista branca: {int(self.mask.sum())} segmentos na rede (de {self.n_raw})")

    # ------------------------------------------------------------ potential
    def _phi(self, raw):
        """Mario's row (0..4) x shaping. Mario is a single lit segment, in 2
        columns (.1/.2) x 5 rows: index 0..9, row = index // 2. If there isn't
        exactly one lit (blinking, cutscene) it keeps the last value."""
        if not self.shaping or self.mario_idx is None:
            return 0.0
        lit = raw[self.mario_idx]
        if lit.sum() != 1:
            return self.phi
        return self.shaping * (int(lit.argmax()) // 2)

    # ------------------------------------------------------------ API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw, *_ = self._cmd("RESET")
        self.start_nivel, self.start_inicio = self.nivel, self.inicio
        for _ in range(int(self.np_random.integers(0, self.noop_max))):
            raw, *_ = self._cmd("STEP 0")     # shifts the chip's counters out of phase
        self.frames = [raw[self.mask]] * self.k
        self.phi = 0.0
        self.phi = self._phi(raw)
        self.t = 0
        return self._obs(), {"nivel": self.nivel, "start_nivel": self.start_nivel,
                             "start_inicio": self.start_inicio}

    def step(self, action):
        raw, morte, entrega, bug = self._cmd(f"STEP {int(action)}")
        self.frames.pop(0)
        self.frames.append(raw[self.mask])
        self.t += 1

        reward = -0.01 + (1.0 if entrega else 0.0) - (1.0 if morte else 0.0)

        # potential-based shaping; at the end of the episode phi(s') = 0
        phi_new = 0.0 if (morte or bug) else self._phi(raw)
        reward += self.gamma * phi_new - self.phi
        self.phi = phi_new

        truncated = bug or self.t >= self.max_steps
        return (self._obs(), reward, morte, truncated,
                {"entrega": entrega, "bug": bug, "nivel": self.nivel})

    def _obs(self):
        extras = np.array([nivel_norm(self.nivel),
                           min(self.fase / FASE_MAX, 1.0)], dtype=np.float32)
        return np.concatenate(self.frames + [extras]).astype(np.float32)

    def close(self):
        self.fin.close()
        self.fout.close()
