"""Modelo final (Seções 4 e 5): treino, limiar, checkpoint .pt, carregamento, fidelidade, métricas,
generalização (gap, curvas, K-fold) e eficiência computacional.

Uso: python run_rn.py --etapa final --problema clf|reg|ambos   (depois da etapa "estudo")
O conjunto de TESTE só é usado aqui, uma vez, com o modelo já escolhido e carregado do arquivo.
"""
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (f1_score, precision_recall_curve, precision_score, recall_score,  # noqa: E402
                             roc_curve)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split  # noqa: E402

from . import metrics as M  # noqa: E402
from . import rn_config as C  # noqa: E402
from .config import CLASSES, ROOT, TARGETS  # noqa: E402
from .inference import Preditor  # noqa: E402
from .rn_models import macs, n_params  # noqa: E402
from .rn_prep import preparar, split_val  # noqa: E402
from .rn_study import f1_constante, normalizar  # noqa: E402
from .rn_train import avaliar, rodar, saida, sigmoid  # noqa: E402

CLS = CLASSES["clf"]
NOME_ARQ = {"clf": "classificacao", "reg": "regressao"}


def _py(x):
    """Converte para tipos simples (exigência do torch.load(..., weights_only=True))."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def ambiente():
    return {"so": platform.platform(), "python": platform.python_version(),
            "cpu": C.CPU_NOME, "cpu_detectada": platform.processor() or platform.machine(), "nucleos_logicos": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "threads_torch": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma (treino em CPU)",
            "potencia_assumida_w": C.POTENCIA_ASSUMIDA_W}


# ---------------------------------------------------------------- limiar (só validação)
def melhor_limiar(y, prob):
    """Limiar que maximiza o F1 macro na VALIDAÇÃO; devolve também a curva precisão/recall x limiar."""
    lab = np.where(np.asarray(y) == 1, CLS[1], CLS[0])
    linhas = []
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = np.where(prob >= t, CLS[1], CLS[0])
        linhas.append({"limiar": t, "precision_pos": precision_score(lab, pred, pos_label=CLS[1], zero_division=0),
                       "recall_pos": recall_score(lab, pred, pos_label=CLS[1], zero_division=0),
                       "F1_macro": f1_score(lab, pred, average="macro", zero_division=0)})
    curva = pd.DataFrame(linhas)
    return float(curva.loc[curva.F1_macro.idxmax(), "limiar"]), curva


# ---------------------------------------------------------------- checkpoint
def checkpoint(problem, modelo, extra, enc, cfg, limiar, info):
    return {
        "state_dict": modelo.state_dict(),                        # pesos e vieses
        "arch": _py(extra["arch"]),                               # para recriar a arquitetura
        "config_treino": _py({k: v for k, v in cfg.items()}),
        "x_media": torch.tensor(enc.mean, dtype=torch.float32),   # StandardScaler (ajustado no treino)
        "x_desvio": torch.tensor(enc.std, dtype=torch.float32),
        "features": list(enc.feature_names),                      # ordem das colunas na entrada da rede
        "encoder": _py(enc.to_dict()),                            # imputação, one-hot e ordinal (do treino)
        "problema": problem, "target": TARGETS[problem],
        "y_media": float(extra["y_media"]), "y_desvio": float(extra["y_desvio"]),   # target padronizado (reg)
        "classes": list(CLS), "limiar": float(limiar),
        "info": _py(info), "torch_version": str(torch.__version__),
    }


# ---------------------------------------------------------------- medições
def medir_inferencia(modelo, X, n_rep=None, n_aquecimento=5):
    """Mede tempo e memória apenas da etapa de inferência (lote inteiro)."""
    n_rep = n_rep or C.N_REP_INFERENCIA
    modelo.eval()
    proc = psutil.Process()
    with torch.inference_mode():
        for _ in range(n_aquecimento):           # warm-up
            modelo(X)
        mem_antes = proc.memory_info().rss
        tempos = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            modelo(X)
            tempos.append(time.perf_counter() - t0)
        mem_depois = proc.memory_info().rss
    m, s = float(np.mean(tempos)), float(np.std(tempos))
    return {"tempo_medio_s": m, "tempo_desvio_s": s, "tempo_por_amostra_ms": 1000 * m / len(X),
            "vazao_amostras_s": len(X) / m, "delta_memoria_kb": (mem_depois - mem_antes) / 1024,
            "rss_mb": mem_depois / 2**20, "tempo_total_s": float(np.sum(tempos))}


def medir_individual(modelo, X, n=200, n_rep=None):
    """Latência com UMA amostra por vez (n amostras, n_rep repetições)."""
    n_rep = n_rep or C.N_REP_INFERENCIA
    X = X[:n]
    modelo.eval()
    with torch.inference_mode():
        for i in range(10):
            modelo(X[i:i + 1])
        tempos = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            for i in range(len(X)):
                modelo(X[i:i + 1])
            tempos.append((time.perf_counter() - t0) / len(X))
    m = float(np.mean(tempos))
    return {"tempo_por_amostra_ms": 1000 * m, "desvio_ms": 1000 * float(np.std(tempos)), "vazao_amostras_s": 1 / m}


def energia(tempo_s):
    """Estimativa: kWh = P(W) x t(s) / 3,6e6; CO2(g) = kWh x 45,8 g/kWh (fator do Brasil, Aula 08)."""
    kwh = C.POTENCIA_ASSUMIDA_W * tempo_s / 3.6e6
    return kwh, kwh * C.FATOR_CO2_G_KWH


def kb(path):
    return os.path.getsize(path) / 1024


# ---------------------------------------------------------------- gráficos
def fig_limiar(problem, curva, limiar):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curva.limiar, curva.precision_pos, label="Precision (Yes)")
    ax.plot(curva.limiar, curva.recall_pos, label="Recall (Yes)")
    ax.plot(curva.limiar, curva.F1_macro, label="F1 macro")
    ax.axvline(limiar, color="k", ls="--", label=f"limiar escolhido = {limiar:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="0,5")
    ax.set_xlabel("limiar de decisão"); ax.set_ylabel("métrica (validação)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Efeito do limiar na validação (modelo final)")
    fig.tight_layout(); fig.savefig(f"{C.FIGS}/limiar_{problem}.png", dpi=130); plt.close(fig)


def fig_curvas_aprendizado(problem, hists):
    fig, axs = plt.subplots(1, len(hists), figsize=(5.5 * len(hists), 3.8))
    for ax, (nome, (h, ep)) in zip(np.atleast_1d(axs), hists.items()):
        e = np.arange(1, len(h["treino"]) + 1)
        ax.plot(e, h["treino"], label="treino"); ax.plot(e, h["val"], label="validação")
        ax.axvline(ep, color="k", ls="--", label=f"melhor época = {ep}")
        ax.set_xlabel("época"); ax.set_ylabel("perda"); ax.set_title(nome); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{C.FIGS}/curvas_aprendizado_{problem}.png", dpi=130); plt.close(fig)


def fig_clf(y, probs, preds, limiares):
    fig, axs = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 3.6))
    for ax, (nome, pred) in zip(np.atleast_1d(axs), preds.items()):
        cm = M.cm_frame(np.where(y == 1, CLS[1], CLS[0]), pred, CLS).values
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="k" if cm[i, j] < cm.max() / 2 else "w")
        ax.set_xticks([0, 1], CLS); ax.set_yticks([0, 1], CLS); ax.set_xlabel("previsto"); ax.set_ylabel("real")
        ax.set_title(f"{nome} (limiar {limiares[nome]:.2f})", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{C.FIGS}/matrizes_confusao_clf.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for nome, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=nome); ax[1].plot(rc, pr, label=nome)
    ax[0].plot([0, 1], [0, 1], "k:", label="aleatório")
    ax[1].axhline(y.mean(), color="k", ls=":", label=f"prevalência = {y.mean():.3f}")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR (Recall)"); ax[0].set_title("ROC (teste)")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall (teste)")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{C.FIGS}/roc_pr_clf.png", dpi=130); plt.close(fig)


def fig_reg(y, preds):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for nome, p in preds.items():
        axs[0].scatter(y, p, s=8, alpha=.5, label=nome)
        axs[1].scatter(p, y - p, s=8, alpha=.5, label=nome)
    lo, hi = float(min(y.min(), min(p.min() for p in preds.values()))), float(max(y.max(), max(p.max() for p in preds.values())))
    axs[0].plot([lo, hi], [lo, hi], "k--", label="y = ŷ")
    axs[0].set_xlabel("real (mmHg)"); axs[0].set_ylabel("previsto (mmHg)"); axs[0].set_title("Real x previsto (teste)")
    axs[1].axhline(0, color="k", ls="--"); axs[1].set_xlabel("previsto (mmHg)"); axs[1].set_ylabel("resíduo y - ŷ (mmHg)")
    axs[1].set_title("Resíduos (teste)")
    for a in axs:
        a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{C.FIGS}/real_vs_previsto_residuos_reg.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- etapas
def kfold(problem, cfg, Xa, ya, Xv, yv):
    """Validação cruzada K-fold da configuração final no conjunto de treino (80%).
    Em cada fold: pré-processamento ajustado no treino do fold; 20% dele para early stopping (e limiar)."""
    clf = problem == "clf"
    X = pd.concat([Xa, Xv], ignore_index=True); y = pd.concat([ya, yv], ignore_index=True)
    kf = (StratifiedKFold if clf else KFold)(C.K_FOLDS, shuffle=True, random_state=C.SEED)
    linhas = []
    for i, (itr, iva) in enumerate(kf.split(X, y)):
        Xf, yf = X.iloc[itr], y.iloc[itr]
        Xi, Xe, yi, ye = train_test_split(Xf, yf, test_size=C.VAL_SIZE, random_state=C.SEED, stratify=yf if clf else None)
        fr = tuple(d.reset_index(drop=True) for d in (Xi, yi, Xe, ye, X.iloc[iva], y.iloc[iva]))
        d = preparar(problem, True, fr)
        _, ex = rodar(problem, cfg, C.SEED, d, devolver_modelo=True)
        lim = melhor_limiar(d["yva"], sigmoid(saida(ex["modelo"], d["Xva"])))[0] if clf else 0.5
        m = avaliar(problem, d["yte"], saida(ex["modelo"], d["Xte"]), ex["y_media"], ex["y_desvio"], lim)
        linhas.append({"fold": i + 1, **{k: m.get(k, np.nan) for k in
                                          (["F1_macro", "ROC_AUC", "PR_AUC"] if clf else ["RMSE", "MAE", "R2"])}})
        print(f"    fold {i + 1}: {C.PRIM[problem]} = {m.get(C.PRIM[problem], np.nan):.4f}")
    return pd.DataFrame(linhas)


def demo_eval(problem, cfg, data, modelo_final, extra_final):
    """O que acontece se esquecer model.eval() numa rede com dropout (medido na VALIDAÇÃO)."""
    if cfg["dropout"] > 0:
        modelo, ex = modelo_final, extra_final
        origem = "modelo final"
    else:
        _, ex = rodar(problem, {**cfg, "dropout": 0.3}, C.SEED, data, devolver_modelo=True)
        modelo = ex["modelo"]
        origem = "configuração final com dropout = 0,3 (o modelo final não tem dropout)"
    p = C.PRIM[problem]
    m_eval = avaliar(problem, data["yva"], saida(modelo, data["Xva"]), ex["y_media"], ex["y_desvio"]).get(p, np.nan)
    X = torch.tensor(data["Xva"], dtype=torch.float32)
    saidas, mets = [], []
    modelo.train()                               # ERRO proposital: dropout continua ativo
    with torch.no_grad():
        for _ in range(5):
            o = modelo(X).squeeze(1).numpy().astype(np.float64)
            saidas.append(o)
            mets.append(avaliar(problem, data["yva"], o, ex["y_media"], ex["y_desvio"]).get(p, np.nan))
    modelo.eval()
    S = np.stack(saidas)
    return {"origem": origem, "metrica": p, "val_com_eval": m_eval, "val_sem_eval_media": float(np.mean(mets)),
            "val_sem_eval_desvio": float(np.std(mets)), "dif_max_entre_passadas": float(np.abs(S[0] - S[1]).max())}


def persistencia_inteiro(modelo, tmp):
    """Compara com salvar o modelo inteiro (pickle)."""
    p = os.path.join(tmp, "modelo_inteiro.pt")
    torch.save(modelo, p)
    r = {"tamanho_inteiro_kb": kb(p)}
    try:
        torch.load(p, map_location="cpu", weights_only=True)
        r["weights_only_true"] = "carregou"
    except Exception as e:  # esperado: o arquivo contém a classe MLP serializada por pickle
        r["weights_only_true"] = f"falhou ({type(e).__name__})"
    m2 = torch.load(p, map_location="cpu", weights_only=False)
    r["inteiro_carrega_com_weights_only_false"] = isinstance(m2, nn.Module)
    return r


def extras(modelo, X_te, y_te, problem, ex, tmp, lim, m_ref):
    """Desafio opcional: TorchScript (trace) e quantização dinâmica int8."""
    r = {}
    p = C.PRIM[problem]
    try:
        ts = torch.jit.trace(modelo.eval(), X_te[:1])
        path = os.path.join(tmp, "modelo_ts.pt"); ts.save(path)
        with torch.inference_mode():
            o = ts(X_te).squeeze(1).numpy().astype(np.float64)
        r["torchscript"] = {"tamanho_kb": kb(path), **{k: v for k, v in medir_inferencia(ts, X_te).items() if k in ("tempo_por_amostra_ms", "vazao_amostras_s")},
                            p: avaliar(problem, y_te, o, ex["y_media"], ex["y_desvio"], lim).get(p, np.nan)}
    except Exception as e:
        r["torchscript"] = {"erro": f"{type(e).__name__}: {str(e)[:120]}"}
    try:
        q = torch.ao.quantization.quantize_dynamic(modelo.eval(), {nn.Linear}, dtype=torch.qint8)
        path = os.path.join(tmp, "modelo_q.pt"); torch.save(q.state_dict(), path)
        with torch.inference_mode():
            o = q(X_te).squeeze(1).numpy().astype(np.float64)
        r["quantizado_int8"] = {"tamanho_kb": kb(path), **{k: v for k, v in medir_inferencia(q, X_te).items() if k in ("tempo_por_amostra_ms", "vazao_amostras_s")},
                                p: avaliar(problem, y_te, o, ex["y_media"], ex["y_desvio"], lim).get(p, np.nan)}
    except Exception as e:
        r["quantizado_int8"] = {"erro": f"{type(e).__name__}: {str(e)[:120]}"}
    r["original"] = {"tamanho_kb": None, **{k: v for k, v in medir_inferencia(modelo, X_te).items() if k in ("tempo_por_amostra_ms", "vazao_amostras_s")}, p: m_ref}
    return r


def run(problem):
    t_ini = time.perf_counter()
    clf = problem == "clf"
    p = C.PRIM[problem]
    fr = split_val(problem)
    Xa, ya, Xv, yv, Xte_raw, yte_raw = fr
    data = preparar(problem, True, fr)
    enc = data["enc"]
    cf = json.load(open(f"{C.TABLES}/config_final_{problem}.json"))
    cfgs = {"Baseline RN": normalizar(problem, C.BASELINE), "RN final": normalizar(problem, cf["config"])}
    out = {"ambiente": ambiente(), "config_final": cfgs["RN final"], "baseline": cfgs["Baseline RN"],
           "busca_val_media": cf["val_media"], "busca_val_desvio": cf["val_desvio"], "sementes": C.SEEDS}
    tmp = tempfile.mkdtemp()

    # ---------- treino (semente principal), limiar na validação e checkpoint ----------
    res, linhas_m, linhas_g, probs_te, preds_te, lims, fidel = {}, [], [], {}, {}, {}, []
    for nome, cfg in cfgs.items():
        print(f"  treinando {nome} ...")
        rss0 = psutil.Process().memory_info().rss
        linha, ex = rodar(problem, cfg, C.SEED, data, devolver_modelo=True)
        modelo = ex["modelo"]
        if clf:
            lim, curva = melhor_limiar(data["yva"], sigmoid(saida(modelo, data["Xva"])))
            curva.to_csv(f"{C.TABLES}/limiar_{problem}_{'final' if nome == 'RN final' else 'baseline'}.csv", index=False)
            if nome == "RN final":
                fig_limiar(problem, curva, lim)
        else:
            lim = 0.5
        lims[nome] = lim
        # saídas do modelo EM MEMÓRIA, antes de salvar
        out_mem = saida(modelo, data["Xte"])
        m_antes = avaliar(problem, data["yte"], out_mem, ex["y_media"], ex["y_desvio"], lim)
        arq = "modelo" if nome == "RN final" else "baseline"
        path = os.path.join(C.MODELS, f"{arq}_{NOME_ARQ[problem]}{C.SUFIXO_MODELO}.pt")
        torch.save(checkpoint(problem, modelo, ex, enc, cfg, lim,
                              {"val_" + p: linha[f"val_{p}"], "semente": C.SEED, "epoca_melhor": linha["epoca_melhor"]}), path)

        # ---------- carregamento e fidelidade ----------
        pred = Preditor(path)
        out_load = pred.saida_bruta(Xte_raw).numpy().astype(np.float64)
        m_depois = avaliar(problem, data["yte"], out_load, ex["y_media"], ex["y_desvio"], lim)
        fidel.append({"modelo": nome, "arquivo": os.path.relpath(path, ROOT),
                      "allclose": bool(torch.allclose(torch.tensor(out_mem), torch.tensor(out_load), rtol=1e-5, atol=1e-6)),
                      "dif_max_abs": float(np.abs(out_mem - out_load).max()),
                      f"{p}_teste_antes_de_salvar": m_antes[p], f"{p}_teste_modelo_carregado": m_depois[p],
                      "metricas_identicas": bool(m_antes[p] == m_depois[p])})

        # ---------- métricas (modelo CARREGADO) em treino, validação e teste ----------
        mets = {}
        for conj, Xr, yr in (("treino", Xa, data["ytr"]), ("validacao", Xv, data["yva"]), ("teste", Xte_raw, data["yte"])):
            o = pred.saida_bruta(Xr).numpy().astype(np.float64)
            mets[conj] = avaliar(problem, yr, o, ex["y_media"], ex["y_desvio"], lim)
            linhas_m.append({"modelo": nome, "conjunto": conj, **mets[conj]})
            if conj == "teste":
                if clf:
                    probs_te[nome] = sigmoid(o)
                    preds_te[nome] = np.where(probs_te[nome] >= lim, CLS[1], CLS[0])
                else:
                    preds_te[nome] = o * ex["y_desvio"] + ex["y_media"]
        base_tr = {"F1_macro": f1_constante(problem)} if clf else {}
        gap, rel, diag = M.gap_and_diagnosis(problem, mets["treino"], mets["teste"], base_tr)
        linhas_g.append({"modelo": nome, f"{p}_treino": mets["treino"][p], f"{p}_validacao": mets["validacao"][p],
                         f"{p}_teste": mets["teste"][p], "gap_treino_teste": gap, "gap_relativo": rel, "diagnostico": diag,
                         "epoca_melhor": linha["epoca_melhor"], "epocas_rodadas": linha["epocas_rodadas"]})
        res[nome] = {"modelo": modelo, "ex": ex, "linha": linha, "path": path, "pred": pred, "rss0": rss0}

    pd.DataFrame(linhas_m).to_csv(f"{C.TABLES}/metricas_{problem}.csv", index=False)
    pd.DataFrame(linhas_g).to_csv(f"{C.TABLES}/generalizacao_{problem}.csv", index=False)
    pd.DataFrame(fidel).to_csv(f"{C.TABLES}/fidelidade_{problem}.csv", index=False)
    if clf:
        fig_clf(data["yte"], probs_te, preds_te, lims)
    else:
        fig_reg(data["yte"], preds_te)
    fig_curvas_aprendizado(problem, {n: (r["ex"]["hist"], r["linha"]["epoca_melhor"]) for n, r in res.items()})
    for n, r in res.items():
        json.dump(r["ex"]["hist"], open(f"{C.TABLES}/hist_{problem}_{'final' if n == 'RN final' else 'baseline'}.json", "w"))

    # ---------- inferência em ambiente limpo (processo Python separado) ----------
    novas = os.path.join(ROOT, "data", f"novas_amostras_{problem}.csv")
    Xte_raw.head(20).to_csv(novas, index=False)
    saida_csv = os.path.join(tmp, "prev.csv")
    final = res["RN final"]
    cmd = [sys.executable, os.path.join("src", "inference.py"), "--modelo", os.path.relpath(final["path"], ROOT),
           "--entrada", os.path.relpath(novas, ROOT), "--saida", saida_csv]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    limpo = {"comando": " ".join(["python", "src/inference.py", "--modelo", os.path.relpath(final["path"], ROOT),
                                  "--entrada", os.path.relpath(novas, ROOT)]),
             "codigo_saida": proc.returncode, "stdout": proc.stdout[-2500:], "stderr": proc.stderr[-800:]}
    if proc.returncode == 0:
        prev = pd.read_csv(saida_csv)
        o = saida(final["modelo"], data["Xte"][:20])
        ref = sigmoid(o) if clf else o * final["ex"]["y_desvio"] + final["ex"]["y_media"]
        limpo["dif_max_abs_vs_memoria"] = float(np.abs(prev.iloc[:, 0].to_numpy() - ref).max())
    out["inferencia_ambiente_limpo"] = limpo

    # ---------- persistência: state_dict x modelo inteiro; map_location ----------
    out["persistencia"] = {"state_dict_kb": kb(final["path"]), **persistencia_inteiro(final["modelo"], tmp)}
    a = final["ex"]["arch"]
    out["persistencia"].update({"n_params": n_params(final["modelo"]), "bytes_float32_teorico_kb": n_params(final["modelo"]) * 4 / 1024,
                                "baseline_kb": kb(res["Baseline RN"]["path"]), "baseline_n_params": n_params(res["Baseline RN"]["modelo"])})

    # ---------- validação cruzada K-fold (configuração final) ----------
    print(f"  validação cruzada {C.K_FOLDS}-fold ...")
    cv = kfold(problem, cfgs["RN final"], Xa, ya, Xv, yv)
    cv.to_csv(f"{C.TABLES}/cv_{problem}.csv", index=False)
    out["cv"] = {"media": float(cv[p].mean()), "desvio": float(cv[p].std())}

    # ---------- eficiência computacional ----------
    X_te = torch.tensor(data["Xte"], dtype=torch.float32)
    linhas_e = []
    for nome, r in res.items():
        a = r["ex"]["arch"]
        lote = medir_inferencia(r["modelo"], X_te)
        indiv = medir_individual(r["modelo"], X_te)
        t0 = time.perf_counter()
        for _ in range(C.N_REP_INFERENCIA):
            r["pred"].prever(Xte_raw)
        ponta = 1000 * (time.perf_counter() - t0) / C.N_REP_INFERENCIA / len(Xte_raw)
        kwh_tr, co2_tr = energia(r["linha"]["tempo_treino_s"])
        kwh_inf, co2_inf = energia(lote["tempo_total_s"])
        linhas_e.append({
            "modelo": nome, "tempo_treino_s": r["linha"]["tempo_treino_s"], "epocas_rodadas": r["linha"]["epocas_rodadas"],
            "pico_rss_treino_mb": r["linha"]["pico_rss_mb"],
            "inf_lote_ms_por_amostra": lote["tempo_por_amostra_ms"], "inf_lote_desvio_s": lote["tempo_desvio_s"],
            "inf_lote_vazao_s": lote["vazao_amostras_s"], "inf_individual_ms": indiv["tempo_por_amostra_ms"],
            "inf_individual_vazao_s": indiv["vazao_amostras_s"], "inf_com_preproc_ms_por_amostra": ponta,
            "rss_inferencia_mb": lote["rss_mb"], "delta_mem_inferencia_kb": lote["delta_memoria_kb"],
            "n_params": n_params(r["modelo"]), "camadas_ocultas": len(a["hidden"]), "MACs_por_amostra": macs(a),
            "tamanho_disco_kb": kb(r["path"]), "energia_treino_kWh": kwh_tr, "CO2_treino_g": co2_tr,
            f"energia_inferencia_{C.N_REP_INFERENCIA}rep_kWh": kwh_inf, f"CO2_inferencia_{C.N_REP_INFERENCIA}rep_g": co2_inf})
    pd.DataFrame(linhas_e).to_csv(f"{C.TABLES}/eficiencia_{problem}.csv", index=False)

    # ---------- eval() esquecido, extras ----------
    out["demo_eval"] = demo_eval(problem, cfgs["RN final"], data, final["modelo"], final["ex"])
    m_ref = [r for r in linhas_m if r["modelo"] == "RN final" and r["conjunto"] == "teste"][0][p]
    out["extras"] = extras(final["modelo"], X_te, data["yte"], problem, final["ex"], tmp, lims["RN final"], m_ref)
    out["limiar"] = lims

    # ---------- comparação opcional com a atividade anterior (mesmos treino/teste) ----------
    prev_path = os.path.join(ROOT, "reports", "tables", "resumo_final.csv")
    if os.path.exists(prev_path):
        ant = pd.read_csv(prev_path)
        ant = ant[(ant.problema == problem) & ant.modelo.isin(["Árvore (ajustada)", "Random Forest", "Gradient Boosting"])]
        cols = ["F1_macro", "ROC_AUC", "PR_AUC", "Accuracy"] if clf else ["RMSE", "MAE", "R2", "MAPE_%"]
        comp = ant[["modelo"] + cols + ["treino_tempo_medio_s", "tamanho_disco_kb"]].copy()
        teste = pd.DataFrame([r for r in linhas_m if r["conjunto"] == "teste"])
        ef = pd.DataFrame(linhas_e).set_index("modelo")
        rn = teste[["modelo"] + cols].copy()
        rn["treino_tempo_medio_s"] = [ef.loc[n, "tempo_treino_s"] for n in rn.modelo]
        rn["tamanho_disco_kb"] = [ef.loc[n, "tamanho_disco_kb"] for n in rn.modelo]
        pd.concat([comp, rn], ignore_index=True).to_csv(f"{C.TABLES}/comparacao_{problem}.csv", index=False)

    json.dump(_py(out), open(f"{C.TABLES}/final_{problem}.json", "w"), indent=2, default=str)
    print(f"[{problem}] final concluído em {(time.perf_counter() - t_ini) / 60:.1f} min")
