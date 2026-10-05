"""LCD segment map for Mario's Bombs Away (gnw_mbaway)."""

TOCHAS = [
    "1.1.3", "0.2.3", "0.2.0",
    "0.3.3", "1.3.3", "1.3.0",
    "1.4.3", "0.5.3", "0.5.0",
    "0.6.3", "1.6.3", "1.6.0",
    "1.7.3", "0.8.3", "1.7.0",
]

HOMEM_FINAL = ["1.8.0", "1.9.0"]

MARIO = [
    "1.1.1", "1.1.2",
    "0.3.1", "0.3.2",
    "1.4.1", "1.4.2",
    "0.6.1", "0.6.2",
    "0.8.1", "1.7.2",
]

HOMEM_CHARUTO = ["1.9.3", "0.9.3", "1.8.3", "1.8.2"]

FOGOS_CHAO = ["0.8.2", "1.6.2", "0.5.2", "1.3.2", "0.2.2"]

# 3 seven-segment digits
PONTUACAO = [
    "1.10.2", "1.10.1", "0.11.2", "0.11.1", "0.11.0", "1.10.0", "0.11.3",
    "1.11.2", "0.12.2", "1.11.1", "0.12.1", "0.12.0", "1.11.0", "0.12.3",
    "1.12.2", "0.13.2", "1.12.1", "0.13.1", "0.13.0", "1.12.0", "0.13.3",
]

# ---------------------------------------------------------------- aggregates
JOGO = TOCHAS + HOMEM_FINAL + MARIO + HOMEM_CHARUTO + FOGOS_CHAO
TODOS = JOGO + PONTUACAO

# signals used by the detectors
SEG_ENTREGA = ("1.8.0", "0.8.1")   # Mario in the last position + the man at the end

if __name__ == "__main__":
    for nome, g in [("tochas", TOCHAS), ("homem final", HOMEM_FINAL),
                    ("mario", MARIO), ("homem charuto", HOMEM_CHARUTO),
                    ("fogos do chao", FOGOS_CHAO), ("pontuacao", PONTUACAO)]:
        print(f"{nome:15s} {len(g):3d}")
    print(f"{'JOGO':15s} {len(JOGO):3d}")
    print(f"{'TOTAL mapeado':15s} {len(TODOS):3d}  (de 131)")
    dup = [x for x in TODOS if TODOS.count(x) > 1]
    print("duplicados:", sorted(set(dup)) or "nenhum")
