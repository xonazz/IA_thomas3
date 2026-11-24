"""Estatísticas das sessões de elaboração de regras (esforço humano).
Preencha logs/cronometro_regras.csv com as SUAS sessões (mín. 3 por problema).
Uso: python -m src.cronometro
"""
import pandas as pd
from .config import LOGS, TABLES

df = pd.read_csv(f"{LOGS}/cronometro_regras.csv")
if df.empty:
    raise SystemExit("Preencha logs/cronometro_regras.csv com as sessões reais.")
out = []
for prob, g in df.groupby("problema"):
    out.append({"problema": prob, "n_sessoes": len(g), "tempo_total_min": g.duracao_min.sum(),
                "tempo_medio_sessao_min": g.duracao_min.mean(), "desvio_sessao_min": g.duracao_min.std(ddof=0),
                "n_regras": g.n_regras_criadas.sum(),
                "tempo_medio_por_regra_min": g.duracao_min.sum() / g.n_regras_criadas.sum()})
pd.DataFrame(out).to_csv(f"{TABLES}/esforco_humano.csv", index=False)
print(pd.DataFrame(out).round(2).to_string(index=False))
