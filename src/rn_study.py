"""Estudo de hiperparâmetros (Seção 3): um fator por vez, mapa de calor (lr x batch) e busca aleatória.

Uso: python run_rn.py --etapa estudo --problema clf|reg|ambos
Todas as decisões usam a VALIDAÇÃO; o teste não é consultado aqui.
Cada execução (configuração x semente) é gravada em logs/hiperparametros.csv assim que termina, então
o estudo pode ser interrompido e retomado (o que já foi feito é pulado).
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from . import rn_config as C  # noqa: E402
from .rn_prep import preparar, split_val  # noqa: E402
from .rn_train import CHAVES, rodar  # noqa: E402

HP_COLS = ["neurons", "layers", "activation", "optimizer", "lr", "batch", "epochs", "patience",
           "weight_decay", "dropout", "loss", "scale"]


LOG_COLS = (["problema", "estudo", "param", "valor", "cfg_key"] + HP_COLS
            + ["seed", "n_params", "epoca_melhor", "epocas_rodadas", "tempo_treino_s", "divergiu", "pico_rss_mb", "perda_val_min"]
            + [f"{c}_{k}" for p in ("clf", "reg") for k in CHAVES[p] for c in ("treino", "val")])


def normalizar(problem, cfg):
    c = dict(cfg)
    if c["loss"] == "padrao":
        c["loss"] = "MSE" if problem == "reg" else "BCE_pos_weight"
    return c


def cfg_key(problem, cfg):
    c = normalizar(problem, cfg)
    return json.dumps({k: (str(c[k]) if k in ("batch", "lr", "weight_decay") else c[k]) for k in HP_COLS},
                      sort_keys=True)


def vstr(v):
    return str(v)


class Registro:
    """Lê/grava logs/hiperparametros.csv e reaproveita execuções idênticas (mesma config e semente)."""

    def __init__(self):
        self.df = pd.read_csv(C.LOG_HP, dtype={"valor": str, "cfg_key": str}) if os.path.exists(C.LOG_HP) else pd.DataFrame()
        self.hist_path = os.path.join(C.LOGS, "historicos_rapido.json" if C.RAPIDO else "historicos.json")
        self.hist = json.load(open(self.hist_path)) if os.path.exists(self.hist_path) else {}

    def feito(self, problem, estudo, param, valor, seed):
        if self.df.empty:
            return False
        d = self.df
        return bool(((d.problema == problem) & (d.estudo == estudo) & (d.param == param)
                     & (d.valor.astype(str) == vstr(valor)) & (d.seed == seed)).any())

    def cache(self, problem, key, seed):
        if self.df.empty:
            return None
        d = self.df[(self.df.problema == problem) & (self.df.cfg_key == key) & (self.df.seed == seed)]
        return None if d.empty else d.iloc[0].to_dict()

    def add(self, linha):
        novo = pd.DataFrame([linha]).reindex(columns=LOG_COLS)   # mesmas colunas para clf e reg
        novo.to_csv(C.LOG_HP, mode="a", header=not os.path.exists(C.LOG_HP), index=False)
        self.df = pd.concat([self.df, novo], ignore_index=True)

    def salvar_hist(self):
        json.dump(self.hist, open(self.hist_path, "w"))


def executar(reg, problem, estudo, param, valor, cfg, datas):
    """Treina cfg com todas as sementes (ou reaproveita), registrando cada execução."""
    key = cfg_key(problem, cfg)
    for seed in C.SEEDS:
        if reg.feito(problem, estudo, param, valor, seed):
            continue
        base = {"problema": problem, "estudo": estudo, "param": param, "valor": vstr(valor), "cfg_key": key}
        antigo = reg.cache(problem, key, seed)
        if antigo is not None:          # mesma configuração já treinada (ex.: o baseline aparece em todo estudo)
            antigo.update(base)
            reg.add(antigo)
            continue
        linha, extra = rodar(problem, normalizar(problem, cfg), seed, datas[bool(cfg["scale"])])
        reg.add({**base, **{k: cfg[k] for k in HP_COLS}, **linha})
        if seed == C.SEEDS[0]:
            reg.hist[f"{problem}|{key}"] = extra["hist"]


# ---------------------------------------------------------------- agregação
def resumir(df, problem):
    """Média e desvio entre sementes por (estudo, param, valor) + gap e diagnóstico."""
    p = C.PRIM[problem]
    g = df.groupby(["estudo", "param", "valor"], sort=False)
    out = g.agg(n_params=("n_params", "first"),
                treino_media=(f"treino_{p}", "mean"), treino_desvio=(f"treino_{p}", "std"),
                val_media=(f"val_{p}", "mean"), val_desvio=(f"val_{p}", "std"),
                epoca_parada=("epoca_melhor", "mean"), epocas_rodadas=("epocas_rodadas", "mean"),
                tempo_s=("tempo_treino_s", "mean"), divergiu=("divergiu", "sum"), n_seeds=("seed", "count"),
                cfg_key=("cfg_key", "first")).reset_index()
    extra = "ROC_AUC" if problem == "clf" else "R2"
    out[f"treino_{extra}"] = g[f"treino_{extra}"].mean().values
    out[f"val_{extra}_media"] = g[f"val_{extra}"].mean().values
    out[f"val_{extra}_desvio"] = g[f"val_{extra}"].std().values
    # gap = desempenho_treino - desempenho_validação (para RMSE: val - treino); > 0 => piora fora do treino
    out["gap"] = (out.treino_media - out.val_media) if problem == "clf" else (out.val_media - out.treino_media)
    out["diagnostico"] = [diagnostico(problem, r) for r in out.itertuples()]
    return out


def f1_constante(problem):
    """F1 macro do classificador constante (sempre 'No') no treino: referência de underfitting."""
    y = preparar_cache(problem)["ytr"]
    p_no = 1 - y.mean()
    return (2 * p_no / (1 + p_no)) / 2


_CACHE = {}


def preparar_cache(problem):
    if problem not in _CACHE:
        fr = split_val(problem)
        _CACHE[problem] = {True: preparar(problem, True, fr), False: preparar(problem, False, fr)}
    return _CACHE[problem][True]


def diagnostico(problem, r):
    """Mesmos limiares da atividade anterior: overfitting se o gap relativo > 10%;
    underfitting se o treino fica a menos de 0,05 do modelo constante (F1) ou R² de treino <= 0,05."""
    if not np.isfinite(r.val_media):
        return "divergiu"
    rel = r.gap / max(abs(r.treino_media), 1e-9)
    if problem == "clf":
        fraco = r.treino_media <= f1_constante(problem) + 0.05
    else:
        fraco = getattr(r, "treino_R2") <= 0.05
    return "overfitting" if rel > 0.10 else ("underfitting" if fraco else "bem ajustado")


def sensibilidade(res, problem):
    """Ordena os hiperparâmetros pelo impacto na métrica de validação (melhor - pior valor testado).
    O efeito é considerado real se a diferença for maior que a soma dos desvios (entre sementes) do melhor e do pior."""
    maior = C.MAIOR_MELHOR[problem]
    linhas = []
    for param, d in res[res.estudo == "ofat"].groupby("param", sort=False):
        f = d[np.isfinite(d.val_media)]
        if f.empty:
            continue
        b = f.loc[f.val_media.idxmax() if maior else f.val_media.idxmin()]
        w = f.loc[f.val_media.idxmin() if maior else f.val_media.idxmax()]
        amp = abs(b.val_media - w.val_media)
        ruido = np.nan_to_num(b.val_desvio) + np.nan_to_num(w.val_desvio)
        linhas.append({"hiperparametro": C.NOMES[param], "param": param, "melhor_valor": b.valor,
                       "melhor_val": b.val_media, "pior_valor": w.valor, "pior_val": w.val_media,
                       "amplitude": amp, "soma_desvios": ruido, "efeito_real": bool(amp > ruido),
                       "n_divergiu": int((~np.isfinite(d.val_media)).sum())})
    return pd.DataFrame(linhas).sort_values("amplitude", ascending=False)


# ---------------------------------------------------------------- gráficos
def _rotulo(v):
    try:
        x = float(v)
        return f"{x:g}"
    except ValueError:
        return str(v)


def fig_param(problem, param, d, hist):
    p = C.PRIM[problem]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(d))
    ax[0].errorbar(x - 0.05, d.treino_media, yerr=d.treino_desvio.fillna(0), fmt="o-", capsize=4, label="treino")
    ax[0].errorbar(x + 0.05, d.val_media, yerr=d.val_desvio.fillna(0), fmt="s-", capsize=4, label="validação")
    ax[0].set_xticks(x, [_rotulo(v) for v in d.valor], rotation=0)
    ax[0].set_xlabel(C.NOMES[param]); ax[0].set_ylabel(f"{p} (média ± desvio, {len(C.SEEDS)} sementes)")
    for i, r in enumerate(d.itertuples()):
        if not np.isfinite(r.val_media):
            ax[0].annotate("divergiu", (i, ax[0].get_ylim()[0]), ha="center", fontsize=8, color="red")
    ax[0].legend(); ax[0].grid(alpha=.3)
    for r in d.itertuples():
        h = hist.get(f"{problem}|{r.cfg_key}")
        if h:
            v = np.asarray(h["val"], dtype=float)
            v[~np.isfinite(v)] = np.nan
            ax[1].plot(np.arange(1, len(v) + 1), v, label=_rotulo(r.valor))
    ax[1].set_yscale("log"); ax[1].set_xlabel("época"); ax[1].set_ylabel("perda de validação (semente 1)")
    ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=.3)
    fig.suptitle(f"{'Classificação' if problem == 'clf' else 'Regressão'} - {C.NOMES[param]}")
    fig.tight_layout(); fig.savefig(f"{C.FIGS}/hp_{problem}_{param}.png", dpi=130); plt.close(fig)


def fig_heatmap(problem, d):
    p = C.PRIM[problem]
    lrs, bs = C.HEATMAP["lr"], C.HEATMAP["batch"]
    M = np.full((len(lrs), len(bs)), np.nan); S = M.copy()
    for r in d.itertuples():
        lr, b = r.valor.split("|")
        M[lrs.index(float(lr)), bs.index(int(b))] = r.val_media
        S[lrs.index(float(lr)), bs.index(int(b))] = r.val_desvio
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = "viridis" if C.MAIOR_MELHOR[problem] else "viridis_r"
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    for i in range(len(lrs)):
        for j in range(len(bs)):
            t = "div." if not np.isfinite(M[i, j]) else f"{M[i, j]:.3f}\n±{np.nan_to_num(S[i, j]):.3f}"
            ax.text(j, i, t, ha="center", va="center", fontsize=8, color="w")
    ax.set_xticks(range(len(bs)), bs); ax.set_yticks(range(len(lrs)), [f"{v:g}" for v in lrs])
    ax.set_xlabel("tamanho do batch"); ax.set_ylabel("taxa de aprendizado")
    fig.colorbar(im, label=f"{p} na validação")
    ax.set_title(f"{'Classificação' if problem == 'clf' else 'Regressão'}: taxa de aprendizado x batch")
    fig.tight_layout(); fig.savefig(f"{C.FIGS}/heatmap_{problem}.png", dpi=130); plt.close(fig)
    pd.DataFrame(M, index=lrs, columns=bs).to_csv(f"{C.TABLES}/heatmap_{problem}_media.csv")
    pd.DataFrame(S, index=lrs, columns=bs).to_csv(f"{C.TABLES}/heatmap_{problem}_desvio.csv")


# ---------------------------------------------------------------- busca aleatória
def amostrar(n):
    rng = np.random.default_rng(C.SEED)
    sp = C.RANDOM_SPACE
    cfgs = []
    for _ in range(n):
        c = dict(C.BASELINE)
        for k, v in sp.items():
            if isinstance(v, tuple):
                c[k] = float(f"{10 ** rng.uniform(np.log10(v[0]), np.log10(v[1])):.2e}")
            else:
                c[k] = v[rng.integers(len(v))]
        c = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in c.items()}
        cfgs.append(c)
    return cfgs


# ---------------------------------------------------------------- principal
def run(problem):
    t0 = time.perf_counter()
    print(f"[{problem}] sementes={C.SEEDS} threads={torch.get_num_threads()} log={C.LOG_HP}")
    preparar_cache(problem)
    datas = _CACHE[problem]
    reg = Registro()

    for param, valores in C.STUDY.items():
        valores = valores[problem] if isinstance(valores, dict) else valores
        t1 = time.perf_counter()
        for v in valores:
            executar(reg, problem, "ofat", param, v, {**C.BASELINE, param: v}, datas)
        reg.salvar_hist()
        print(f"  {param:13s} ok ({time.perf_counter() - t1:6.1f} s)")

    for lr in C.HEATMAP["lr"]:
        for b in C.HEATMAP["batch"]:
            executar(reg, problem, "heatmap", "lr x batch", f"{lr:g}|{b}", {**C.BASELINE, "lr": lr, "batch": b}, datas)
    reg.salvar_hist()
    print("  heatmap ok")

    cfgs = amostrar(C.N_RANDOM)
    for i, c in enumerate(cfgs):
        executar(reg, problem, "random", "cfg", i, c, datas)
    reg.salvar_hist()
    print("  busca aleatória ok")

    df = reg.df[reg.df.problema == problem].copy()
    res = resumir(df, problem)
    res.to_csv(f"{C.TABLES}/estudo_{problem}.csv", index=False)
    for param in C.STUDY:
        d = res[(res.estudo == "ofat") & (res.param == param)]
        d.drop(columns="cfg_key").to_csv(f"{C.TABLES}/estudo_{problem}_{param}.csv", index=False)
        fig_param(problem, param, d.reset_index(drop=True), reg.hist)
    fig_heatmap(problem, res[res.estudo == "heatmap"])
    sensibilidade(res, problem).to_csv(f"{C.TABLES}/sensibilidade_{problem}.csv", index=False)

    rs = res[res.estudo == "random"].copy()
    rs["cfg"] = [cfgs[int(v)] for v in rs.valor]
    for k in ["neurons", "layers", "activation", "optimizer", "lr", "batch", "weight_decay", "dropout"]:
        rs[k] = [c[k] for c in rs.cfg]
    asc = not C.MAIOR_MELHOR[problem]
    rs = rs[np.isfinite(rs.val_media)].sort_values("val_media", ascending=asc)
    rs.drop(columns=["cfg", "cfg_key"]).to_csv(f"{C.TABLES}/busca_aleatoria_{problem}.csv", index=False)
    melhor = rs.iloc[0]
    final = dict(melhor.cfg)
    json.dump({"config": final, "val_media": float(melhor.val_media), "val_desvio": float(melhor.val_desvio),
               "treino_media": float(melhor.treino_media), "id_busca": int(melhor.valor)},
              open(f"{C.TABLES}/config_final_{problem}.json", "w"), indent=2)
    print(f"[{problem}] configuração final: {final}")
    print(f"[{problem}] estudo concluído em {(time.perf_counter() - t0) / 60:.1f} min")
