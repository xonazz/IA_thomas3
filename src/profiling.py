"""Medição de tempo, memória, tamanho em disco, complexidade e energia."""
import os
import tempfile
import time
import tracemalloc

import joblib
import numpy as np

from .config import FATOR_CO2_G_KWH, N_REP_INFERENCIA, POTENCIA_ASSUMIDA_W


def energia(tempo_s, potencia_w=POTENCIA_ASSUMIDA_W):
    """Estimativa: kWh = P(W) x t(s) / 3,6e6; CO2(g) = kWh x 45,8 g/kWh."""
    kwh = potencia_w * tempo_s / 3.6e6
    return kwh, kwh * FATOR_CO2_G_KWH


def medir_inferencia(predict_fn, X, n_rep=N_REP_INFERENCIA):
    predict_fn(X)  # aquecimento (fora da medição)
    tempos = []
    tracemalloc.start()
    for _ in range(n_rep):
        t0 = time.perf_counter(); predict_fn(X); tempos.append(time.perf_counter() - t0)
    _, pico = tracemalloc.get_traced_memory(); tracemalloc.stop()
    m, s = float(np.mean(tempos)), float(np.std(tempos))
    kwh, co2 = energia(sum(tempos))
    return {"inf_tempo_medio_s": m, "inf_tempo_desvio_s": s,
            "inf_tempo_por_amostra_ms": 1000 * m / len(X), "inf_vazao_amostras_s": len(X) / m,
            "inf_pico_memoria_kb": pico / 1024,
            f"inf_energia_kWh_{n_rep}rep": kwh, f"inf_CO2_g_{n_rep}rep": co2}


def medir_treino(make_and_fit, n_rep=5):
    """make_and_fit() deve criar o modelo e treiná-lo. Tempo: média de n_rep;
    memória: uma execução separada com tracemalloc (que distorce o tempo)."""
    tempos = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); make_and_fit(); tempos.append(time.perf_counter() - t0)
    tracemalloc.start(); make_and_fit()
    _, pico = tracemalloc.get_traced_memory(); tracemalloc.stop()
    m = float(np.mean(tempos)); kwh, co2 = energia(m)
    return {"treino_tempo_medio_s": m, "treino_tempo_desvio_s": float(np.std(tempos)),
            "treino_pico_memoria_kb": pico / 1024, "treino_energia_kWh": kwh, "treino_CO2_g": co2}


def tamanho_disco_kb(modelo):
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    joblib.dump(modelo, path)
    kb = os.path.getsize(path) / 1024; os.remove(path)
    return kb


def complexidade(pipe_fitted):
    m = pipe_fitted.named_steps["model"]
    if hasattr(m, "tree_"):
        return {"profundidade": m.get_depth(), "nos_totais": m.tree_.node_count,
                "folhas": m.get_n_leaves(), "n_arvores": 1}
    ests = np.array(m.estimators_).ravel()
    return {"profundidade": max(e.get_depth() for e in ests),
            "nos_totais": int(sum(e.tree_.node_count for e in ests)),
            "folhas": int(sum(e.get_n_leaves() for e in ests)), "n_arvores": len(ests)}
