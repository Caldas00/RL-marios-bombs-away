"""Looks for the RAM address that controls the speed.

  python research/analyze_speed.py research/data/speed_session.csv

Idea: the speed is, by definition, the interval between torch movements.
That interval is measured from the segments (no RAM), and then we look for
the address that "explains" the interval: for each value of the address, the
interval should be nearly constant.
"""
import sys
import numpy as np
import pandas as pd

TOCHAS = ["1.1.3","0.2.3","0.2.0","0.3.3","1.3.3","1.3.0","1.4.3","0.5.3","0.5.0",
          "0.6.3","1.6.3","1.6.0","1.7.3","0.8.3","1.7.0"]

d = pd.read_csv(sys.argv[1])
cols = [f"s_{t}" for t in TOCHAS if f"s_{t}" in d.columns]
T = d[cols].values
mov = np.where((np.diff(T, axis=0) != 0).any(axis=1))[0] + 1     # frames with movement
print(f"{len(d)} frames, {len(mov)} movimentos de tocha")

iv = np.diff(mov)
ok = iv < 200                      # discard pauses (cutscenes, deliveries)
t_iv, iv = mov[1:][ok], iv[ok]     # interval measured at the frame of the next movement
print(f"{len(iv)} intervalos validos | min {iv.min()} | mediana {int(np.median(iv))} | max {iv.max()}")

# how the interval evolves over the session
print("\nintervalo medio por bloco de 2000 frames:")
for a in range(0, len(d), 2000):
    m = (t_iv >= a) & (t_iv < a + 2000)
    if m.sum() > 3:
        print(f"  {a:6d}-{a+2000:6d}: {iv[m].mean():6.1f} frames  (n={m.sum()})")

# candidates: each address, its value at the time of the previous movement
rams = [c for c in d.columns if c.startswith("m_")]
res = []
for c in rams:
    v = d[c].values[mov[:-1][ok]]  # address value right after the previous movement
    if len(set(v.tolist())) < 2:
        continue
    # variance of the interval explained by the value (eta^2)
    tot = ((iv - iv.mean()) ** 2).sum()
    within = sum(((iv[v == u] - iv[v == u].mean()) ** 2).sum() for u in set(v.tolist()))
    eta = 1 - within / tot if tot > 0 else 0
    res.append((eta, c, len(set(v.tolist()))))

res.sort(reverse=True)
print("\ncandidatos a velocidade (eta2 = quanto do intervalo o endereco explica):")
for eta, c, n in res[:12]:
    print(f"  {c}: eta2 = {eta:.3f}  ({n} valores distintos)")

if res:
    best = res[0][1]
    v = d[best].values[mov[:-1][ok]]
    print(f"\nintervalo medio por valor de {best}:")
    for u in sorted(set(v.tolist())):
        print(f"  {best} = {u:2d}: {iv[v == u].mean():6.1f} frames  (n={(v == u).sum()})")
