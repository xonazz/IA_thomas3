"""Atividade de Redes Neurais (MLP em PyTorch): executa tudo, na ordem.

    python run_rn.py                                   # estudo + final + relatório, para os dois problemas
    python run_rn.py --etapa estudo --problema clf     # só uma etapa / um problema
    RN_RAPIDO=1 python run_rn.py                       # teste de fumaça (~2 min), em pastas separadas
    RN_SEEDS=5 python run_rn.py                        # 5 sementes em vez de 3

Pré-requisito: python -m src.data_prep (gera data/processed com a divisão 80/20 da atividade anterior).
O estudo grava cada execução em logs/hiperparametros.csv; se for interrompido, rode de novo que ele continua.
"""
import argparse
import time

import torch

from src import rn_config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=["estudo", "final", "relatorio", "tudo"], default="tudo")
    ap.add_argument("--problema", choices=["clf", "reg", "ambos"], default="ambos")
    a = ap.parse_args()
    probs = ["clf", "reg"] if a.problema == "ambos" else [a.problema]
    print(f"PyTorch {torch.__version__} | threads={torch.get_num_threads()} | sementes={C.SEEDS} | rápido={C.RAPIDO}")
    t0 = time.perf_counter()
    if a.etapa in ("estudo", "tudo"):
        from src import rn_study
        for p in probs:
            rn_study.run(p)
    if a.etapa in ("final", "tudo"):
        from src import rn_final
        for p in probs:
            print(f"[{p}] modelo final")
            rn_final.run(p)
    if a.etapa in ("relatorio", "tudo"):
        from src import rn_report
        rn_report.main()
    print(f"total: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
