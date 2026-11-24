"""Gera a base de regras de regressão (diagnósticos Hypertension x Diabetes).

As condições vêm do conhecimento de domínio (a pressão sistólica depende do diagnóstico de hipertensão);
o valor de cada consequente é a média da pressão sistólica das amostras de TREINO cobertas pela regra.
"""
from .config import RULES


def build(Xtr, ytr, path=f"{RULES}/regras_regressao.txt"):
    linhas = ["# Base de conhecimento - regressão (target: Blood_Pressure_Systolic, mmHg)",
              "# Consequentes = média da pressão sistólica no TREINO para cada combinação de diagnósticos.",
              "# ID | tipo | prioridade | regra"]
    geral = float(ytr.mean()); k = 1
    for hip in ("Yes", "No"):
        for dia in ("No", "Yes"):
            m = (Xtr.Hypertension == hip) & (Xtr.Diabetes == dia)
            val = float(ytr[m].mean()) if m.sum() >= 5 else geral
            linhas.append(f"R{k} | relação | prio={k} | SE Hypertension == {hip} E Diabetes == {dia} ENTÃO PA = {val:.2f}")
            k += 1
    linhas.append(f"R{k} | estratégia | prio={k} | SE VERDADEIRO ENTÃO PA = {geral:.2f}")
    open(path, "w", encoding="utf-8").write("\n".join(linhas) + "\n")
    return path
