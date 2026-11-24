"""Pré-processamento para as redes neurais (sem PyTorch e sem pickle).

Faz o mesmo tratamento da atividade anterior (imputação pela mediana/moda, one-hot nas categóricas,
códigos ordinais em Stress_Level/Physical_Activity/Alcohol) e acrescenta a padronização (StandardScaler).
Tudo é ajustado SOMENTE no conjunto de treino e pode ser exportado para um dicionário de tipos simples,
que vai dentro do checkpoint .pt (compatível com torch.load(..., weights_only=True)).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CAT, CLASSES, NUM, ORD_COLS, ORD_ORDER, SEED, TARGETS
from .data import load
from .rn_config import VAL_SIZE


class Encoder:
    def __init__(self, problem=None, scale=True):
        self.problem, self.scale = problem, scale

    # ---------- ajuste (apenas treino) ----------
    def fit(self, X):
        p = self.problem
        self.num = list(NUM)
        self.cat = list(CAT[p])
        self.ord = list(ORD_COLS)
        self.ord_order = {c: list(o) for c, o in zip(ORD_COLS, ORD_ORDER)}
        self.medians = {c: float(X[c].astype(float).median()) for c in self.num}
        self.modes = {c: str(X[c].dropna().astype(str).mode().iloc[0]) for c in self.cat + self.ord}
        self.categories = {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in self.cat}
        Z = self._encode(X)
        if self.scale:
            self.mean = Z.mean(axis=0)
            sd = Z.std(axis=0)
            self.std = np.where(sd > 0, sd, 1.0)
        else:                               # estudo "sem StandardScaler"
            self.mean = np.zeros(Z.shape[1])
            self.std = np.ones(Z.shape[1])
        return self

    @property
    def feature_names(self):
        return self.num + [f"{c}={v}" for c in self.cat for v in self.categories[c]] + self.ord

    @property
    def raw_columns(self):
        return self.num + self.cat + self.ord

    def _col(self, X, c):
        s = X[c].astype(object)
        return s.where(s.notna(), self.modes[c]).astype(str)

    def _encode(self, X):
        """Imputação + one-hot + ordinal (SEM padronização)."""
        partes = [X[c].astype(float).fillna(self.medians[c]).to_numpy()[:, None] for c in self.num]
        for c in self.cat:
            s = self._col(X, c).to_numpy()
            partes.append(np.stack([(s == v) for v in self.categories[c]], axis=1).astype(float))
        for c in self.ord:
            s = self._col(X, c)
            m = {v: i for i, v in enumerate(self.ord_order[c])}
            partes.append(s.map(m).fillna(m[self.modes[c]]).to_numpy(dtype=float)[:, None])
        return np.hstack(partes)

    def encode(self, X):
        """Matriz numérica ainda não padronizada (a padronização x' = (x - média) / desvio é aplicada depois)."""
        return self._encode(X).astype(np.float32)

    def transform(self, X):
        # tudo em float32, exatamente como na inferência com tensores (src/inference.py)
        return (self.encode(X) - self.mean.astype(np.float32)) / self.std.astype(np.float32)

    # ---------- exportação para o checkpoint ----------
    def to_dict(self):
        return {"problem": self.problem, "scale": bool(self.scale), "num": list(self.num), "cat": list(self.cat),
                "ord": list(self.ord), "ord_order": {k: list(v) for k, v in self.ord_order.items()},
                "medians": {k: float(v) for k, v in self.medians.items()},
                "modes": {k: str(v) for k, v in self.modes.items()},
                "categories": {k: [str(x) for x in v] for k, v in self.categories.items()}}

    @classmethod
    def from_dict(cls, d, mean=None, std=None):
        e = cls(d["problem"], d["scale"])
        e.num, e.cat, e.ord = list(d["num"]), list(d["cat"]), list(d["ord"])
        e.ord_order = {k: list(v) for k, v in d["ord_order"].items()}
        e.medians, e.modes = dict(d["medians"]), dict(d["modes"])
        e.categories = {k: list(v) for k, v in d["categories"].items()}
        if mean is not None:
            e.mean, e.std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        return e


def target_array(problem, y):
    """Classificação: 1 = 'Yes' (classe positiva), 0 = 'No'. Regressão: float."""
    if problem == "clf":
        return (np.asarray(y) == CLASSES["clf"][1]).astype(np.float32)
    return np.asarray(y, dtype=np.float32)


def split_val(problem):
    """Mesma divisão treino/teste da atividade anterior (80/20, random_state=42) e, dentro do treino,
    20% para validação (estratificada na classificação)."""
    Xtr, ytr, Xte, yte = load(problem)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=VAL_SIZE, random_state=SEED,
                                      stratify=ytr if problem == "clf" else None)
    return Xa.reset_index(drop=True), ya.reset_index(drop=True), Xv.reset_index(drop=True), \
        yv.reset_index(drop=True), Xte.reset_index(drop=True), yte.reset_index(drop=True)


def preparar(problem, scale=True, frames=None):
    """Retorna dicionário com matrizes padronizadas (ajuste no treino) e targets numéricos."""
    Xa, ya, Xv, yv, Xte, yte = frames if frames is not None else split_val(problem)
    enc = Encoder(problem, scale).fit(Xa)
    return {"enc": enc, "Xtr": enc.transform(Xa), "ytr": target_array(problem, ya),
            "Xva": enc.transform(Xv), "yva": target_array(problem, yv),
            "Xte": enc.transform(Xte), "yte": target_array(problem, yte),
            "raw_te": Xte, "raw_va": Xv, "raw_tr": Xa, "target": TARGETS[problem]}
