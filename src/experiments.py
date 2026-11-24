"""Orquestra tudo para um problema ('clf' ou 'reg'), salvando tabelas/figuras."""
import json
import os
import platform

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from . import metrics as M, models as MD, profiling as P, visualization as V
from . import regression_rules
from .config import CLASSES as CLS, FIGS, MODELS, RULES, SEED, TABLES
from .data import load
from .expert_system import ForwardChainingEngine

CLASSES = CLS["clf"]


def env_info():
    return {"so": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(), "nucleos": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)}


def run(problem):
    Xtr, ytr, Xte, yte = load(problem)
    clf = problem == "clf"
    prim = "F1_macro" if clf else "RMSE"
    base_model = MD.baseline(problem).fit(Xtr, ytr)
    log = {}

    # ---------- Sistema especialista ----------
    rpath = f"{RULES}/{'regras_classificacao' if clf else 'regras_regressao'}.txt"
    if not clf and not os.path.exists(rpath):   # não sobrescreve regras editadas à mão
        regression_rules.build(Xtr, ytr)
    es = ForwardChainingEngine(rpath, CLASSES if clf else None)
    if clf:
        es.fit_rule_stats(Xtr, ytr)
        es.save_stats(f"{MODELS}/es_stats.json")

    # ---------- Árvores ----------
    crit = ["gini", "entropy"] if clf else ["squared_error", "absolute_error"]
    trees_full = {c: MD.unrestricted_tree(problem, Xtr, ytr, c) for c in crit}
    s_tree = MD.tune(problem, "tree", Xtr, ytr)
    s_rf = MD.tune(problem, "rf", Xtr, ytr, n_iter=10)
    s_gb = MD.tune(problem, "gb", Xtr, ytr, n_iter=8)
    best = {"Árvore (ajustada)": s_tree.best_estimator_, "Random Forest": s_rf.best_estimator_,
            "Gradient Boosting": s_gb.best_estimator_}
    all_models = {"Baseline (constante)": base_model, "Sist. Especialista": es,
                  f"Árvore sem poda ({crit[0]})": trees_full[crit[0]],
                  f"Árvore sem poda ({crit[1]})": trees_full[crit[1]], **best}
    for n, s in [("tree", s_tree), ("rf", s_rf), ("gb", s_gb)]:
        pd.DataFrame(s.cv_results_).sort_values("rank_test_score").to_csv(f"{TABLES}/cv_{problem}_{n}.csv", index=False)
        log[n] = {"best_params": {k.replace("model__", ""): (v if not isinstance(v, np.generic) else v.item())
                                   for k, v in s.best_params_.items()}, "best_cv": float(s.best_score_)}
    # critério de impureza: melhor CV por critério (grade da árvore)
    cvr = pd.DataFrame(s_tree.cv_results_)
    by_crit = cvr.groupby("param_model__criterion")["mean_test_score"].max()
    by_crit.to_csv(f"{TABLES}/tree_criterio_{problem}.csv")

    # ---------- Predições e métricas (treino e teste) ----------
    rows, cms, probas, preds, gaps = [], {}, {}, {}, []
    for n, m in all_models.items():
        res = {}
        for tag, X, y in (("treino", Xtr, ytr), ("teste", Xte, yte)):
            p = m.predict(X)
            if clf:
                pr = m.predict_proba(X)
                res[tag] = M.classification_metrics(y, p, pr, CLASSES)
                if tag == "teste":
                    cms[n] = M.cm_frame(y, p, CLASSES); probas[n] = pr
            else:
                res[tag] = M.regression_metrics(y, p)
                if tag == "teste":
                    preds[n] = p
        if n.startswith("Baseline"):
            base_tr = res["treino"]
        gap, rel, diag = M.gap_and_diagnosis(problem, res["treino"], res["teste"], base_tr)
        for tag in ("treino", "teste"):
            rows.append({"modelo": n, "conjunto": tag, **res[tag]})
        gaps.append({"modelo": n, f"{prim}_treino": res["treino"][prim], f"{prim}_teste": res["teste"][prim],
                     "gap": gap, "gap_relativo": rel, "diagnostico": diag})
    pd.DataFrame(rows).to_csv(f"{TABLES}/metricas_{problem}.csv", index=False)
    pd.DataFrame(gaps).to_csv(f"{TABLES}/generalizacao_{problem}.csv", index=False)

    # ---------- Validação cruzada (árvore e ensembles) ----------
    cvrows = []
    for n, m in best.items():
        sc = cross_val_score(clone(m), Xtr, ytr, cv=MD.cv(problem), scoring=MD.SCORING[problem])
        sc = sc if clf else -sc
        cvrows.append({"modelo": n, f"{prim}_cv_media": sc.mean(), f"{prim}_cv_desvio": sc.std()})
    pd.DataFrame(cvrows).to_csv(f"{TABLES}/cv_final_{problem}.csv", index=False)

    # ---------- Curva Random Forest x nº de árvores ----------
    base_rf = clone(s_rf.best_estimator_); crv = []
    for k in (1, 10, 50, 100, 200):
        est = clone(base_rf).set_params(model__n_estimators=k)
        sc = cross_val_score(est, Xtr, ytr, cv=MD.cv(problem), scoring=MD.SCORING[problem])
        err_cv = (1 - sc) if clf else -sc
        pt = est.fit(Xtr, ytr).predict(Xtr)
        err_tr = (1 - M.classification_metrics(ytr, pt, est.predict_proba(Xtr), CLASSES)["F1_macro"]) if clf \
            else M.regression_metrics(ytr, pt)["RMSE"]
        crv.append({"n_arvores": k, "erro_treino": err_tr, "erro_cv": err_cv.mean(), "erro_cv_desvio": err_cv.std()})
    crv = pd.DataFrame(crv); crv.to_csv(f"{TABLES}/rf_curva_{problem}.csv", index=False)
    V.rf_curve(crv, "1 - F1 macro" if clf else "RMSE", f"rf_curva_{problem}")

    # ---------- Figuras ----------
    if clf:
        V.confusion_grid(cms, "matrizes_confusao"); V.roc_pr_curves(yte, {k: v for k, v in probas.items() if not k.startswith("Baseline")}, CLASSES, "roc_pr")
    else:
        V.real_vs_pred(yte, {k: v for k, v in preds.items() if not k.startswith("Baseline")}, "real_vs_previsto_residuos")
    fn = best["Random Forest"].named_steps["prep"].get_feature_names_out()
    fn = [f.split("__")[-1] for f in fn]
    V.tree_figure(best["Árvore (ajustada)"], fn, CLASSES if clf else None, f"arvore_{problem}")
    from sklearn.tree import export_text
    open(f"{TABLES}/arvore_regras_{problem}.txt", "w").write(
        export_text(best["Árvore (ajustada)"].named_steps["model"], feature_names=fn, max_depth=4))
    imps = {n: pd.Series(best[n].named_steps["model"].feature_importances_, index=fn)
            for n in ("Árvore (ajustada)", "Random Forest", "Gradient Boosting")}
    pd.DataFrame(imps).to_csv(f"{TABLES}/importancias_{problem}.csv")
    V.importances(imps, f"importancias_{problem}")

    # ---------- Eficiência ----------
    eff = []
    ef = P.medir_inferencia(es.predict, Xte)
    eff.append({"modelo": "Sist. Especialista", **ef, "tamanho_disco_kb": os.path.getsize(rpath) / 1024,
                "n_regras": len(es.rules)})
    for n, m in best.items():
        key = {"Árvore (ajustada)": "tree", "Random Forest": "rf", "Gradient Boosting": "gb"}[n]
        params = {k: v for k, v in s_tree.best_params_.items()} if key == "tree" else \
            (s_rf.best_params_ if key == "rf" else s_gb.best_params_)
        mk = lambda: clone(m).set_params(**{k: v for k, v in params.items()}).fit(Xtr, ytr)
        tr_ = P.medir_treino(mk, n_rep=3)
        inf = P.medir_inferencia(m.predict, Xte)
        eff.append({"modelo": n, **tr_, **inf, "tamanho_disco_kb": P.tamanho_disco_kb(m),
                    **P.complexidade(m)})
        joblib.dump(m, f"{MODELS}/{problem}_{key}.joblib")
    pd.DataFrame(eff).to_csv(f"{TABLES}/eficiencia_{problem}.csv", index=False)
    json.dump({**log, "ambiente": env_info()}, open(f"{TABLES}/log_{problem}.json", "w"), indent=2,
              ensure_ascii=False, default=str)
    return log
