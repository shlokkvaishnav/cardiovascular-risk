"""Optuna-based hyperparameter tuning for training candidates.

Each trial scores a candidate's hyperparameters via StratifiedKFold
cross-validation on X_train only. This is intentionally single-level CV for
tuning, not nested CV: the existing train/test split (feature_engineering.py)
already gives a held-out test set that tuning never touches, so it still
yields an honest generalization estimate once the final model is evaluated
against it in scripts/train.py. A full nested-CV-around-tuning loop would
cost roughly 5x the compute at this dataset size (70k rows) for a
generalization estimate the untouched test set already provides for free.
"""

import logging
from typing import Any, Callable, Dict, Tuple

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


def logistic_regression_param_space(trial: "optuna.Trial") -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
    }


def random_forest_param_space(trial: "optuna.Trial") -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1,
    }


def lightgbm_param_space(trial: "optuna.Trial") -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }


def xgboost_param_space(trial: "optuna.Trial") -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }


PARAM_SPACES: Dict[str, Callable[["optuna.Trial"], Dict[str, Any]]] = {
    "LogisticRegression": logistic_regression_param_space,
    "RandomForest": random_forest_param_space,
    "LightGBM": lightgbm_param_space,
    "XGBoost": xgboost_param_space,
}


class HyperparameterTuner:
    """Runs an Optuna search for one model family at a time, scored by CV
    ROC-AUC (or another configured metric) on the training split only."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def tune_model(
        self,
        model_name: str,
        model_factory: Callable[..., Any],
        preprocessor: Any,
        X_train: Any,
        y_train: Any,
        n_trials: int,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> Tuple[Dict[str, Any], float]:
        """Returns (best_params, best_cv_score). `best_params` are the kwargs
        `model_factory` was called with on the winning trial (so the caller
        can rebuild the winning model directly)."""
        param_space_fn = PARAM_SPACES.get(model_name)
        if param_space_fn is None:
            raise ValueError(f"No parameter space defined for model '{model_name}'")

        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_seed
        )

        def objective(trial: "optuna.Trial") -> float:
            params = param_space_fn(trial)
            model = model_factory(**params)
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            # n_jobs=1 here: the candidate models already parallelize
            # internally (n_jobs=-1), so parallelizing folds on top would
            # oversubscribe CPU cores for no benefit.
            scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
            )
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.random_seed)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info(
            f"{model_name}: best CV {scoring}={study.best_value:.4f} "
            f"after {n_trials} trials, params={study.best_params}"
        )
        return dict(study.best_params), float(study.best_value)
