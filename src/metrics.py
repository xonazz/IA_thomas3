import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def regression_metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"MAE": mean_absolute_error(y, p), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAPE_%": 100 * mean_absolute_percentage_error(y, p),
            "R2": r2_score(y, p), "MedAE": median_absolute_error(y, p)}


def classification_metrics(y, pred, proba, classes):
    """Problema binário: classe positiva = classes[1]. Médias macro e weighted + métricas da classe positiva.
    ROC-AUC e PR-AUC usam a probabilidade da classe positiva."""
    pos = classes[1]
    yb = (np.asarray(y) == pos).astype(int)
    d = {"Accuracy": accuracy_score(y, pred)}
    for avg in ("macro", "weighted"):
        d[f"Precision_{avg}"] = precision_score(y, pred, average=avg, zero_division=0)
        d[f"Recall_{avg}"] = recall_score(y, pred, average=avg, zero_division=0)
        d[f"F1_{avg}"] = f1_score(y, pred, average=avg, zero_division=0)
    d["Precision_pos"] = precision_score(y, pred, pos_label=pos, zero_division=0)
    d["Recall_pos"] = recall_score(y, pred, pos_label=pos, zero_division=0)
    d["F1_pos"] = f1_score(y, pred, pos_label=pos, zero_division=0)
    d["ROC_AUC"] = roc_auc_score(yb, proba[:, 1])
    d["PR_AUC"] = average_precision_score(yb, proba[:, 1])
    return d


def cm_frame(y, pred, classes):
    return pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)


def gap_and_diagnosis(problem, treino, teste, base_treino):
    """Gap = desempenho_treino - desempenho_teste (clf: F1 macro, maior=melhor);
    para erro (reg: RMSE) usa-se teste - treino. Em ambos, gap > 0 => piora no teste.
    Diagnóstico (limiares explícitos): overfitting se gap relativo > 10%;
    underfitting se o desempenho no TREINO fica a menos de 0,05 do baseline (modelo constante):
    F1 macro <= baseline + 0,05 (clf) ou R2 <= 0,05 (reg); caso contrário, bem ajustado."""
    if problem == "clf":
        gap = treino["F1_macro"] - teste["F1_macro"]; rel = gap / max(treino["F1_macro"], 1e-9)
        fraco = treino["F1_macro"] <= base_treino["F1_macro"] + 0.05
    else:
        gap = teste["RMSE"] - treino["RMSE"]; rel = gap / max(treino["RMSE"], 1e-9)
        fraco = treino["R2"] <= 0.05
    diag = "overfitting" if rel > 0.10 else ("underfitting" if fraco else "bem ajustado")
    return gap, rel, diag
