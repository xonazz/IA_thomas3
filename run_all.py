"""Executa todo o experimento: python run_all.py  (após python -m src.data_prep)"""
import time
from src.experiments import run
for prob in ("clf", "reg"):
    t0 = time.perf_counter(); run(prob); print(prob, "ok", round(time.perf_counter() - t0, 1), "s")
