"""Looks for the speed RAM address.
Idea: the speed only changes with progress (bombs delivered) and stays
constant between deliveries. Per life (reset on death), for each address:
  - how many times it changes value (should be rarely)
  - how many of those changes happen near a delivery (+-WIN frames)
  - correlation (Spearman) between the value and the number of deliveries in the life
Usage: python research/analyze_speed2.py research/data/speed_session.csv
"""
import sys
import numpy as np
import pandas as pd

WIN = 40  # tolerance in frames around a delivery
df = pd.read_csv(sys.argv[1])
n = len(df)
mem = [c for c in df.columns if c.startswith("m_")]
print(f"{n} frames")

# deliveries: rising edge of (1.8.0 & 0.8.1) lit, 30-frame debounce
d = ((df["s_1.8.0"] > 0) & (df["s_0.8.1"] > 0)).values.astype(int)
deliv = []
last = -999
for i in range(1, n):
    if d[i] and not d[i - 1] and i - last > 30:
        deliv.append(i)
        last = i
# deaths: m_2C rises to 15
m2c = df["m_2C"].values
deaths = [i for i in range(1, n) if m2c[i] == 15 and m2c[i - 1] != 15]
print(f"entregas: {len(deliv)} | mortes: {len(deaths)}")
if len(deliv) < 3:
    print("Poucas entregas: grava uma sessao mais longa (varias bombas por vida).")

# life and cumulative number of deliveries per frame
life = np.zeros(n, int)
for dpos in deaths:
    life[dpos:] += 1
kdel = np.zeros(n, int)
cnt = 0
di = 0
for i in range(n):
    while di < len(deliv) and deliv[di] == i:
        cnt += 1
        di += 1
    if i > 0 and life[i] != life[i - 1]:
        cnt = 0
    kdel[i] = cnt
print("entregas maximas numa vida:", kdel.max())

near = np.zeros(n, bool)
for x in deliv:
    near[max(0, x - WIN):x + WIN] = True

rows = []
for c in mem:
    v = df[c].values
    ch = np.where(v[1:] != v[:-1])[0] + 1
    nch = len(ch)
    if nch == 0 or len(np.unique(v)) > 16:
        continue
    frac_near = near[ch].mean()
    # correlation of value vs number of deliveries (Spearman via ranks)
    if len(np.unique(v)) > 1 and len(np.unique(kdel)) > 1:
        r = np.corrcoef(pd.Series(v).rank(), pd.Series(kdel).rank())[0, 1]
    else:
        r = 0.0
    rows.append((c, nch, frac_near, r, len(np.unique(v))))

res = pd.DataFrame(rows, columns=["addr", "mudancas", "frac_perto_entrega", "corr_k", "n_vals"])
# good candidate: few changes, almost all near deliveries, high |corr|
res["score"] = res["corr_k"].abs() * res["frac_perto_entrega"] / np.log1p(res["mudancas"])
print("\nCandidatos (ordenados por score):")
print(res.sort_values("score", ascending=False).head(15).to_string(index=False))

best = res.sort_values("score", ascending=False).head(3)["addr"].tolist()
for c in best:
    g = pd.DataFrame({"v": df[c].values, "k": kdel})
    print(f"\nvalor medio de {c} por n de entregas na vida:")
    print(g.groupby("k")["v"].agg(["mean", "min", "max", "count"]).head(20).to_string())
