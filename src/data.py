import pandas as pd
from .config import DATA, TARGETS, FEATURES


def load(problem):
    """Retorna Xtr, ytr, Xte, yte (splits fixos gerados por data_prep)."""
    t = TARGETS[problem]
    tr = pd.read_csv(f"{DATA}/{problem}_train.csv")
    te = pd.read_csv(f"{DATA}/{problem}_test.csv")
    return tr[FEATURES[problem]], tr[t], te[FEATURES[problem]], te[t]
