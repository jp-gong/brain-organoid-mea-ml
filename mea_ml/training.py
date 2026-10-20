"""Grouped cross-validation and Bayesian model search used in the study."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

RANDOM_STATE = 0
OUTER_N_SPLITS = 5
N_REPEATS = 3
OUTER_SEEDS = [0, 1, 2]
BAYES_N_INIT = 8
BAYES_N_ITER = 16
CORR_THRESHOLD = 0.95
PRESERVE_CORRELATED_FEATURES = {
    "Mean::Mean Firing Rate (Hz)", "Delta::Mean Firing Rate (Hz)"
}

def make_outer_folds(pos_groups: List[str], neg_groups: List[str], n_splits: int = OUTER_N_SPLITS) -> List[Tuple[List[str], List[str]]]:
    if n_splits < 2 or min(len(pos_groups), len(neg_groups)) < n_splits:
        raise ValueError("Each class must have at least n_splits groups.")
    pos_parts = np.array_split(np.asarray(pos_groups, dtype=str), n_splits)
    neg_parts = np.array_split(np.asarray(neg_groups, dtype=str), n_splits)
    folds = []
    for pos_part, neg_part in zip(pos_parts, neg_parts):
        test_groups = neg_part.tolist() + pos_part.tolist()
        train_groups = [g for g in neg_groups + pos_groups if g not in test_groups]
        folds.append((train_groups, test_groups))
    return folds

def make_inner_folds(train_groups: List[str], y_by_group: Dict[str, int], seed: int) -> List[Tuple[List[str], List[str]]]:
    pos = [g for g in train_groups if y_by_group[g] == 1]
    neg = [g for g in train_groups if y_by_group[g] == 0]
    rng_local = np.random.default_rng(seed)
    pos = list(rng_local.permutation(pos))
    neg = list(rng_local.permutation(neg))
    n_folds = min(len(pos), len(neg))
    return make_outer_folds(pos, neg, n_splits=n_folds)

class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.columns_ = list(X_df.columns)
        self.mean_ = X_df.mean(axis=0)
        self.std_ = X_df.std(axis=0).replace(0, 1.0)
        self.std_ = self.std_.fillna(1.0)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df = X_df[self.columns_]
        Z = (X_df - self.mean_) / self.std_
        return Z

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95, preserve=None):
        self.threshold = threshold
        self.preserve = set(preserve or [])

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.input_columns_ = list(X_df.columns)
        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        kept = []
        for col in X_df.columns:
            if col in to_drop:
                continue
            kept.append(col)
            correlated = upper.index[upper[col] > self.threshold].tolist()
            for other in correlated:
                if other in self.preserve and col not in self.preserve:
                    to_drop.add(col)
                elif col in self.preserve and other not in self.preserve:
                    to_drop.add(other)
                else:
                    if other not in self.preserve:
                        to_drop.add(other)
        self.selected_columns_ = [c for c in X_df.columns if c not in to_drop]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df[self.selected_columns_]

def need_scaling(model_name: str) -> bool:
    return model_name in {"Logistic Regression", "SVM (RBF)", "KNN"}

def build_preprocessor(model_name: str) -> List[Tuple[str, Any]]:
    steps = []
    steps.append(('corr', CorrelationFilter(threshold=CORR_THRESHOLD, preserve=PRESERVE_CORRELATED_FEATURES)))
    if need_scaling(model_name):
        steps.append(('scaler', DataFrameStandardScaler()))
    return steps

@dataclass
class ParamSpec:
    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    categories: Optional[List[Any]] = None
    dtype: str = "float"

def _sample_categorical_preserve_type(categories, rng):
    idx = int(rng.integers(0, len(categories)))
    return categories[idx]

def _coerce_value_to_spec(value, spec: ParamSpec):
    if spec.kind == "categorical":
        return value
    if spec.dtype == "int":
        return int(round(float(value)))
    return float(value)

def sample_param(spec: ParamSpec, rng):
    if spec.kind == "categorical":
        return _sample_categorical_preserve_type(spec.categories, rng)
    if spec.log:
        val = np.exp(rng.uniform(np.log(spec.low), np.log(spec.high)))
    else:
        val = rng.uniform(spec.low, spec.high)
    return _coerce_value_to_spec(val, spec)

def canonicalize_params(params: Dict[str, Any], space: Dict[str, ParamSpec]) -> Dict[str, Any]:
    out = {}
    for k, spec in space.items():
        out[k] = _coerce_value_to_spec(params[k], spec)
    return out

def encode_params(params: Dict[str, Any], space: Dict[str, ParamSpec]) -> np.ndarray:
    vec = []
    for k, spec in space.items():
        v = params[k]
        if spec.kind == "categorical":
            vec.append(float(spec.categories.index(v)))
        else:
            vv = float(v)
            if spec.log:
                vv = np.log(vv)
            vec.append(vv)
    return np.array(vec, dtype=float)

def expected_improvement(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best - xi) / sigma
    from scipy.stats import norm
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

def subset_rows(X_df: pd.DataFrame, groups_arr: np.ndarray, group_subset: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    mask = np.isin(groups_arr, np.array(group_subset, dtype=str))
    return X_df.loc[mask].reset_index(drop=True), mask

def fit_predict_one_split(
    model_name: str,
    estimator,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    steps = build_preprocessor(model_name)
    clf = clone(estimator)
    clf.set_params(**params)
    pipe = Pipeline(steps + [("clf", clf)])
    pipe.fit(X_train, y_train)

    if hasattr(pipe, "predict_proba"):

        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        score = pipe.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-score))
    y_pred = (y_proba >= 0.5).astype(int)
    return y_pred, y_proba, pipe

def cv_score_grouped(
    model_name: str,
    estimator,
    params: Dict[str, Any],
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    groups_arr: np.ndarray,
    inner_folds: List[Tuple[List[str], List[str]]],
) -> float:
    scores = []
    for tr_groups, va_groups in inner_folds:
        X_tr, tr_mask = subset_rows(X_df, groups_arr, tr_groups)
        X_va, va_mask = subset_rows(X_df, groups_arr, va_groups)
        y_tr = y_arr[tr_mask]
        y_va = y_arr[va_mask]
        if len(np.unique(y_va)) < 2:
            continue
        _, y_proba, _ = fit_predict_one_split(model_name, estimator, params, X_tr, y_tr, X_va)
        scores.append(roc_auc_score(y_va, y_proba))
    if not scores:
        return -np.inf
    return float(np.mean(scores))

def bayes_optimize_model(
    model_name: str,
    estimator,
    space: Dict[str, ParamSpec],
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    groups_arr: np.ndarray,
    inner_folds: List[Tuple[List[str], List[str]]],
    seed: int,
    n_init: int = 8,
    n_iter: int = 16,
):
    rng = np.random.default_rng(seed)
    obs_params = []
    obs_scores = []

    for _ in range(n_init):
        p = {k: sample_param(spec, rng) for k, spec in space.items()}
        p = canonicalize_params(p, space)
        s = cv_score_grouped(model_name, estimator, p, X_df, y_arr, groups_arr, inner_folds)
        obs_params.append(p)
        obs_scores.append(s)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
    for _ in range(n_iter):
        X_obs = np.vstack([encode_params(p, space) for p in obs_params])
        y_obs = np.array(obs_scores, dtype=float)

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
                module=r"sklearn\.gaussian_process\.kernels",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*noise_level is close to the specified lower bound.*",
                category=ConvergenceWarning,
            )
            gpr.fit(X_obs, y_obs)

        candidates = []
        candidate_params = []
        for _ in range(256):
            p = {k: sample_param(spec, rng) for k, spec in space.items()}
            p = canonicalize_params(p, space)
            candidate_params.append(p)
            candidates.append(encode_params(p, space))
        X_cand = np.vstack(candidates)
        mu, sigma = gpr.predict(X_cand, return_std=True)
        ei = expected_improvement(mu, sigma, np.max(y_obs))
        best_idx = int(np.argmax(ei))
        p_best = candidate_params[best_idx]

        s_best = cv_score_grouped(model_name, estimator, p_best, X_df, y_arr, groups_arr, inner_folds)
        obs_params.append(p_best)
        obs_scores.append(s_best)

    best_idx = int(np.argmax(obs_scores))
    return obs_params[best_idx], float(obs_scores[best_idx]), pd.DataFrame(
        [
            {"iter": i + 1, "score": obs_scores[i], **obs_params[i]}
            for i in range(len(obs_params))
        ]
    )

def make_model_specs():
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    specs = []

    specs.append({
        "model_name": "Logistic Regression",
        "estimator": LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "space": {
            "penalty": ParamSpec(kind="categorical", categories=["l1", "l2"]),
            "C": ParamSpec(kind="float", low=1e-3, high=1e3, log=True, dtype="float"),
        },
        "simplicity_rank": 0,
    })

    specs.append({
        "model_name": "SVM (RBF)",
        "estimator": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "space": {
            "C": ParamSpec(kind="float", low=1e-3, high=1e3, log=True, dtype="float"),
            "gamma": ParamSpec(kind="float", low=1e-4, high=1e1, log=True, dtype="float"),
        },
        "simplicity_rank": 1,
    })

    specs.append({
        "model_name": "Random Forest",
        "estimator": RandomForestClassifier(
            random_state=RANDOM_STATE,
        ),
        "space": {
            "n_estimators": ParamSpec(kind="float", low=100, high=600, dtype="int"),
            "max_depth": ParamSpec(kind="categorical", categories=[None, 2, 3, 4, 5, 6, 8]),
            "min_samples_split": ParamSpec(kind="float", low=2, high=10, dtype="int"),
            "min_samples_leaf": ParamSpec(kind="float", low=1, high=5, dtype="int"),
            "max_features": ParamSpec(kind="categorical", categories=["sqrt", "log2", 0.5, 0.8]),
        },
        "simplicity_rank": 2,
    })

    specs.append({
        "model_name": "Gradient Boosting",
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "space": {
            "n_estimators": ParamSpec(kind="float", low=50, high=300, dtype="int"),
            "learning_rate": ParamSpec(kind="float", low=1e-2, high=3e-1, log=True, dtype="float"),
            "max_depth": ParamSpec(kind="float", low=1, high=4, dtype="int"),
            "subsample": ParamSpec(kind="float", low=0.5, high=1.0, dtype="float"),
            "min_samples_split": ParamSpec(kind="float", low=2, high=10, dtype="int"),
            "min_samples_leaf": ParamSpec(kind="float", low=1, high=5, dtype="int"),
        },
        "simplicity_rank": 3,
    })

    specs.append({
        "model_name": "XGBoost",
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False if "use_label_encoder" in XGBClassifier().get_params() else None,
        ),
        "space": {
            "n_estimators": ParamSpec(kind="float", low=50, high=300, dtype="int"),
            "learning_rate": ParamSpec(kind="float", low=1e-2, high=3e-1, log=True, dtype="float"),
            "max_depth": ParamSpec(kind="float", low=2, high=6, dtype="int"),
            "subsample": ParamSpec(kind="float", low=0.5, high=1.0, dtype="float"),
            "colsample_bytree": ParamSpec(kind="float", low=0.5, high=1.0, dtype="float"),
            "min_child_weight": ParamSpec(kind="float", low=1, high=8, dtype="int"),
        },
        "simplicity_rank": 4,
    })

    specs.append({
        "model_name": "KNN",
        "estimator": KNeighborsClassifier(),
        "space": {
            "n_neighbors": ParamSpec(kind="float", low=1, high=15, dtype="int"),
            "weights": ParamSpec(kind="categorical", categories=["uniform", "distance"]),
            "p": ParamSpec(kind="categorical", categories=[1, 2]),
        },
        "simplicity_rank": 5,
    })

    specs.append({
        "model_name": "CatBoost",
        "estimator": CatBoostClassifier(
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            bootstrap_type="Bayesian",
            use_best_model=False,
            verbose=False,
            allow_writing_files=False,
        ),
        "space": {
            "iterations": ParamSpec(kind="float", low=50, high=300, dtype="int"),
            "learning_rate": ParamSpec(kind="float", low=1e-2, high=3e-1, log=True, dtype="float"),
            "depth": ParamSpec(kind="float", low=2, high=6, dtype="int"),
            "l2_leaf_reg": ParamSpec(kind="float", low=1e-2, high=1e2, log=True, dtype="float"),
            "random_strength": ParamSpec(kind="float", low=1e-3, high=10, log=True, dtype="float"),
            "bagging_temperature": ParamSpec(kind="float", low=0, high=1, dtype="float"),
        },
        "simplicity_rank": 6,
    })

    return specs

def compute_metrics(y_true, y_pred, y_proba):
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return metrics
