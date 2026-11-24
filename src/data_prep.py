"""Preparo dos dados: gera dataset_regression/classification e splits fixos.

Execução (a partir da raiz do projeto): python -m src.data_prep
Imputação e codificação NÃO são feitas aqui (evita data leakage): ficam no Pipeline do scikit-learn,
ajustado apenas no treino. Os ausentes (150 em 5 colunas de features) permanecem como NaN.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, TARGETS

SEED = 42
TEST_SIZE = 0.20
RAW = "data/raw/digital_twin_profile.csv"
OUT = "data/processed"


def main():
    df = pd.read_csv(RAW)
    assert df.Digital_ID.is_unique
    for p in ("reg", "clf"):
        t = TARGETS[p]
        d = df.dropna(subset=[t])[["Digital_ID"] + FEATURES[p] + [t]]
        d.to_csv(f"{OUT}/dataset_{'regression' if p == 'reg' else 'classification'}.csv", index=False)
        tr, te = train_test_split(d, test_size=TEST_SIZE, random_state=SEED,
                                  stratify=d[t] if p == "clf" else None)
        tr.to_csv(f"{OUT}/{p}_train.csv", index=False)
        te.to_csv(f"{OUT}/{p}_test.csv", index=False)
        print(p, d.shape, "treino", tr.shape, "teste", te.shape)


if __name__ == "__main__":
    main()
