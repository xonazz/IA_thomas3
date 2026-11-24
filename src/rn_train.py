"""Treinamento (laço com autograd, DataLoader, early stopping), sementes e avaliação."""
import copy
import math
import random
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import metrics as M
from .config import CLASSES
from .rn_config import FULL_BATCH
from .rn_models import arch, build, n_params

CLS = CLASSES["clf"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(name, params, lr, weight_decay):
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "SGD_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "AdamW":   # decaimento desacoplado; com weight_decay=0 usa o padrão do PyTorch (0,01)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay if weight_decay > 0 else 1e-2)
    raise ValueError(name)


def make_loss(problem, name, pos_weight):
    if problem == "reg":
        return {"padrao": nn.MSELoss, "MSE": nn.MSELoss, "MAE": nn.L1Loss, "Huber": nn.HuberLoss}[name]()
    if name == "BCE":
        return nn.BCEWithLogitsLoss()
    # "padrao" e "BCE_pos_weight": peso n_neg/n_pos na classe positiva (desbalanceamento 83% x 17%)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))


def treinar(modelo, dl_tr, Xva, yva, loss_fn, otimizador, n_epocas=200, paciencia=20):
    """Laço de treinamento com validação e early stopping (guarda o melhor estado).
    Interrompe se a perda ficar NaN/infinita (divergência, ex.: taxa de aprendizado alta)."""
    hist = {"treino": [], "val": []}
    melhor, espera, melhor_estado, melhor_ep = float("inf"), 0, None, 0
    proc, pico_rss, divergiu = psutil.Process(), 0, False
    for ep in range(n_epocas):
        modelo.train()
        soma = 0.0
        for xb, yb in dl_tr:
            otimizador.zero_grad()
            perda = loss_fn(modelo(xb), yb)
            perda.backward()                    # autograd: calcula dL/dθ
            otimizador.step()                   # θ <- θ - η * (regra do otimizador)
            soma += perda.item() * len(xb)
        hist["treino"].append(soma / len(dl_tr.dataset))

        modelo.eval()
        with torch.no_grad():
            val = loss_fn(modelo(Xva), yva).item()
        hist["val"].append(val)
        pico_rss = max(pico_rss, proc.memory_info().rss)

        if not (math.isfinite(val) and math.isfinite(hist["treino"][-1])):
            divergiu = True
            break
        if val < melhor - 1e-6:
            melhor, espera, melhor_ep = val, 0, ep + 1
            melhor_estado = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
            if espera >= paciencia:
                break
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)   # restaura o melhor ponto
    return {"hist": hist, "epoca_melhor": melhor_ep, "epocas_rodadas": len(hist["val"]),
            "divergiu": divergiu, "pico_rss_mb": pico_rss / 2**20}


def saida(modelo, X):
    """Saída bruta da rede (logit ou valor padronizado) em modo de avaliação."""
    modelo.eval()
    with torch.inference_mode():
        return modelo(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy().astype(np.float64)


def _min_finito(v):
    v = np.asarray(v, dtype=float)
    return float(v[np.isfinite(v)].min()) if np.isfinite(v).any() else np.nan


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def avaliar(problem, y, out, y_media=0.0, y_desvio=1.0, limiar=0.5):
    """Métricas da Aula 08 a partir da saída da rede. y: rótulos 0/1 (clf) ou valores originais (reg)."""
    if not np.all(np.isfinite(out)):
        return {}
    if problem == "reg":
        return M.regression_metrics(y, out * y_desvio + y_media)     # volta à escala original (mmHg)
    prob = sigmoid(out)
    lab = np.where(np.asarray(y) == 1, CLS[1], CLS[0])
    pred = np.where(prob >= limiar, CLS[1], CLS[0])
    return M.classification_metrics(lab, pred, np.column_stack([1 - prob, prob]), CLS)


CHAVES = {"clf": ["F1_macro", "ROC_AUC", "PR_AUC", "Accuracy", "Recall_pos", "Precision_pos"],
          "reg": ["RMSE", "MAE", "R2", "MedAE"]}


def rodar(problem, cfg, seed, data, devolver_modelo=False):
    """Treina UMA configuração com UMA semente e devolve as métricas de treino e validação."""
    set_seed(seed)
    Xtr, ytr, Xva, yva = data["Xtr"], data["ytr"], data["Xva"], data["yva"]
    if problem == "reg":   # padroniza o target com média/desvio do treino
        y_media, y_desvio = float(ytr.mean()), float(ytr.std())
    else:
        y_media, y_desvio = 0.0, 1.0
    t_tr = torch.tensor((ytr - y_media) / y_desvio, dtype=torch.float32).unsqueeze(1)
    t_va = torch.tensor((yva - y_media) / y_desvio, dtype=torch.float32).unsqueeze(1)
    X_tr = torch.tensor(Xtr, dtype=torch.float32)
    X_va = torch.tensor(Xva, dtype=torch.float32)

    batch = len(Xtr) if cfg["batch"] == FULL_BATCH else int(cfg["batch"])
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(X_tr, t_tr), batch_size=batch, shuffle=True, generator=g)

    a = arch(cfg, Xtr.shape[1])
    modelo = build(a)
    pos_weight = float((1 - ytr.mean()) / ytr.mean()) if problem == "clf" else 1.0
    loss_fn = make_loss(problem, cfg["loss"], pos_weight)
    opt = make_optimizer(cfg["optimizer"], modelo.parameters(), float(cfg["lr"]), float(cfg["weight_decay"]))

    t0 = time.perf_counter()
    r = treinar(modelo, dl, X_va, t_va, loss_fn, opt, int(cfg["epochs"]), int(cfg["patience"]))
    t_treino = time.perf_counter() - t0

    m_tr = avaliar(problem, ytr, saida(modelo, Xtr), y_media, y_desvio)
    m_va = avaliar(problem, yva, saida(modelo, Xva), y_media, y_desvio)
    linha = {"seed": seed, "n_params": n_params(modelo), "epoca_melhor": r["epoca_melhor"],
             "epocas_rodadas": r["epocas_rodadas"], "tempo_treino_s": t_treino, "divergiu": r["divergiu"],
             "pico_rss_mb": r["pico_rss_mb"], "perda_val_min": _min_finito(r["hist"]["val"])}
    for k in CHAVES[problem]:
        linha[f"treino_{k}"] = m_tr.get(k, np.nan)
        linha[f"val_{k}"] = m_va.get(k, np.nan)
    extra = {"hist": r["hist"], "arch": a, "y_media": y_media, "y_desvio": y_desvio, "pos_weight": pos_weight}
    if devolver_modelo:
        extra["modelo"] = modelo
    return linha, extra
