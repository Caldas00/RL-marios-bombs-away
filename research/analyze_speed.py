"""Procura o endereco de RAM que controla a velocidade.

  python research/analyze_speed.py research/data/speed_session.csv

Ideia: a velocidade e, por definicao, o intervalo entre movimentos das tochas.
Mede-se esse intervalo a partir dos segmentos (sem RAM), e depois procura-se
qual endereco "explica" o intervalo: para cada valor do endereco, o intervalo
deve ser quase constante.
"""
import sys
import numpy as np
import pandas as pd

TOCHAS = ["1.1.3","0.2.3","0.2.0","0.3.3","1.3.3","1.3.0","1.4.3","0.5.3","0.5.0",
          "0.6.3","1.6.3","1.6.0","1.7.3","0.8.3","1.7.0"]

d = pd.read_csv(sys.argv[1])
cols = [f"s_{t}" for t in TOCHAS if f"s_{t}" in d.columns]
T = d[cols].values
mov = np.where((np.diff(T, axis=0) != 0).any(axis=1))[0] + 1     # frames com movimento
print(f"{len(d)} frames, {len(mov)} movimentos de tocha")

iv = np.diff(mov)
ok = iv < 200                      # descarta pausas (cutscenes, entregas)
t_iv, iv = mov[1:][ok], iv[ok]     # intervalo medido no frame do movimento seguinte
print(f"{len(iv)} intervalos validos | min {iv.min()} | mediana {int(np.median(iv))} | max {iv.max()}")

# evolucao do intervalo ao longo da sessao
print("\nintervalo medio por bloco de 2000 frames:")
for a in range(0, len(d), 2000):
    m = (t_iv >= a) & (t_iv < a + 2000)
    if m.sum() > 3:
        print(f"  {a:6d}-{a+2000:6d}: {iv[m].mean():6.1f} frames  (n={m.sum()})")

# candidatos: cada endereco, valor no instante do movimento anterior
rams = [c for c in d.columns if c.startswith("m_")]
res = []
for c in rams:
    v = d[c].values[mov[:-1][ok]]  # valor do endereco logo apos o movimento anterior
    if len(set(v.tolist())) < 2:
        continue
    # variancia do intervalo explicada pelo valor (eta^2)
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
