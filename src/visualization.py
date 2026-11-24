import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.tree import plot_tree

from .config import FIGS


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIGS}/{name}.png", dpi=130); plt.close(fig)


def confusion_grid(cms, name):
    fig, axs = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 4))
    for ax, (t, cm) in zip(np.atleast_1d(axs), cms.items()):
        ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=30)
        ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index); ax.set_title(t, fontsize=9)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm.values[i, j], ha="center", va="center", fontsize=8)
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    save(fig, name)


def roc_pr_curves(y, probas, classes, name):
    """Binário: curvas da classe positiva (classes[1])."""
    yb = (np.asarray(y) == classes[1]).astype(int)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for n, p in probas.items():
        fpr, tpr, _ = roc_curve(yb, p[:, 1]); axs[0].plot(fpr, tpr, label=n)
        pr, rc, _ = precision_recall_curve(yb, p[:, 1]); axs[1].plot(rc, pr, label=n)
    axs[0].plot([0, 1], [0, 1], "k--", lw=.7); axs[0].set_title(f"ROC (classe {classes[1]})")
    axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].legend(fontsize=7)
    axs[1].axhline(yb.mean(), color="k", ls="--", lw=.7)
    axs[1].set_title(f"Precision-Recall (classe {classes[1]}; linha = prevalência)")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].legend(fontsize=7)
    save(fig, name)


def real_vs_pred(y, preds, name):
    n = len(preds)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 7))
    for j, (t, p) in enumerate(preds.items()):
        axs[0, j].scatter(y, p, s=6, alpha=.4); lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        axs[0, j].plot(lim, lim, "r--", lw=.8); axs[0, j].set_title(t, fontsize=9)
        axs[0, j].set_xlabel("Real"); axs[0, j].set_ylabel("Previsto")
        axs[1, j].scatter(p, y - p, s=6, alpha=.4); axs[1, j].axhline(0, color="r", lw=.8)
        axs[1, j].set_xlabel("Previsto"); axs[1, j].set_ylabel("Resíduo")
    save(fig, name)


def tree_figure(pipe_fitted, feature_names, class_names, name, max_depth=3):
    fig = plt.figure(figsize=(16, 7))
    plot_tree(pipe_fitted.named_steps["model"], feature_names=feature_names, class_names=class_names,
              max_depth=max_depth, filled=True, impurity=False, fontsize=7)
    save(fig, name)


def importances(imps, name):
    fig, axs = plt.subplots(1, len(imps), figsize=(5 * len(imps), 4.5))
    for ax, (t, s) in zip(np.atleast_1d(axs), imps.items()):
        s = s.sort_values().tail(10); ax.barh(s.index, s.values); ax.set_title(t, fontsize=9)
    save(fig, name)


def rf_curve(df, ylabel, name):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df.n_arvores, df.erro_treino, label="treino", marker="o")
    ax.errorbar(df.n_arvores, df.erro_cv, yerr=df.erro_cv_desvio, label="validação cruzada", marker="s")
    ax.set_xlabel("Nº de árvores"); ax.set_ylabel(ylabel); ax.legend()
    save(fig, name)
