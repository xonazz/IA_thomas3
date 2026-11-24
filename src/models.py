"""Pipelines e busca de hiperparâmetros (CV apenas no treino)."""
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.dummy import DummyClassifier, DummyRegressor
from .config import CAT, NUM, ORD_COLS, ORD_ORDER, SEED, N_SPLITS


def preprocessor(problem):
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT[problem]),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oe", OrdinalEncoder(categories=ORD_ORDER))]), ORD_COLS),
    ])


def pipe(estimator, problem):
    return Pipeline([("prep", preprocessor(problem)), ("model", estimator)])


def baseline(problem):
    est = DummyClassifier(strategy="prior") if problem == "clf" else DummyRegressor(strategy="mean")
    return pipe(est, problem)


def cv(problem):
    if problem == "clf":
        return StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    return KFold(N_SPLITS, shuffle=True, random_state=SEED)


SCORING = {"clf": "f1_macro", "reg": "neg_root_mean_squared_error"}


def base_estimators(problem):
    if problem == "clf":
        return dict(tree=DecisionTreeClassifier(random_state=SEED),
                    rf=RandomForestClassifier(random_state=SEED, n_jobs=1),
                    gb=GradientBoostingClassifier(random_state=SEED))
    return dict(tree=DecisionTreeRegressor(random_state=SEED),
                rf=RandomForestRegressor(random_state=SEED, n_jobs=1),
                gb=GradientBoostingRegressor(random_state=SEED))


def grids(problem):
    crit = ["gini", "entropy"] if problem == "clf" else ["squared_error", "absolute_error"]
    tree = {"model__criterion": crit, "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_leaf": [1, 5, 20], "model__ccp_alpha": [0.0, 0.001, 0.005]}
    if problem == "clf":
        tree["model__class_weight"] = [None, "balanced"]
    rf = {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 6, 10, 16],
          "model__max_features": ["sqrt", 0.5, 1.0], "model__min_samples_leaf": [1, 3, 5]}
    if problem == "clf":
        rf["model__class_weight"] = [None, "balanced"]
    gb = {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1],
          "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 1.0]}
    return dict(tree=tree, rf=rf, gb=gb)


def tune(problem, name, Xtr, ytr, n_iter=10):
    est = pipe(base_estimators(problem)[name], problem)
    grid = grids(problem)[name]
    kw = dict(scoring=SCORING[problem], cv=cv(problem), n_jobs=1, refit=True, return_train_score=False)
    if name == "tree":
        s = GridSearchCV(est, grid, **kw)
    else:
        s = RandomizedSearchCV(est, grid, n_iter=n_iter, random_state=SEED, **kw)
    return s.fit(Xtr, ytr)


def unrestricted_tree(problem, Xtr, ytr, criterion):
    est = pipe(base_estimators(problem)["tree"].set_params(criterion=criterion), problem)
    return est.fit(Xtr, ytr)
