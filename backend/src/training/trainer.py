import os
import mlflow
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from .tuning import HyperparameterTuner

logger = logging.getLogger(__name__)

CV_SCORING = ["accuracy", "f1", "roc_auc"]


class ModelTrainer:
    """Orchestrates model training with MLflow tracking, Optuna hyperparameter
    tuning, ROC-AUC-driven model selection, and probability calibration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    def train_model(
        self, model, X_train, y_train, model_name: str
    ) -> Tuple[Any, Dict[str, float]]:
        """Fit `model` (a Pipeline) with MLflow tracking. Returns the fitted
        model and a dict of CV accuracy/f1/roc_auc means+stds."""

        with mlflow.start_run(run_name=model_name):
            try:
                mlflow.log_params(model.get_params())
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Could not log params for {model_name}: {e}")

            cv = StratifiedKFold(
                n_splits=self.config["training"]["cv_folds"],
                shuffle=True,
                random_state=self.config["project"]["random_seed"],
            )
            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=CV_SCORING,
                n_jobs=self.config["training"]["n_jobs"],
            )

            metrics: Dict[str, float] = {}
            for metric in CV_SCORING:
                scores = cv_results[f"test_{metric}"]
                metrics[f"cv_{metric}_mean"] = float(scores.mean())
                metrics[f"cv_{metric}_std"] = float(scores.std())
                mlflow.log_metric(f"cv_{metric}_mean", metrics[f"cv_{metric}_mean"])
                mlflow.log_metric(f"cv_{metric}_std", metrics[f"cv_{metric}_std"])

            # Train final model on the full training split
            model.fit(X_train, y_train)

            # Log model. Explicit cloudpickle serialization avoids newer
            # mlflow's default skops format, which rejects common sklearn
            # pipeline internals (e.g. numpy.dtype) as "untrusted types".
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")

            logger.info(
                f"{model_name}: CV ROC-AUC={metrics['cv_roc_auc_mean']:.4f} "
                f"F1={metrics['cv_f1_mean']:.4f} Accuracy={metrics['cv_accuracy_mean']:.4f}"
            )

            return model, metrics

    def _build_candidate_factories(self):
        """Factory functions for each supported candidate. Each factory
        applies this project's fixed defaults first, then any tuned
        hyperparameters on top -- so an empty params dict (tuning disabled or
        failed) reproduces the previous hardcoded behavior exactly."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        n_jobs = self.config["training"].get("n_jobs", -1)

        def _merge(
            defaults: Dict[str, Any], overrides: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {**defaults, **overrides}

        return {
            "LogisticRegression": lambda **p: LogisticRegression(
                **_merge({"max_iter": 1000, "random_state": 42}, p)
            ),
            "RandomForest": lambda **p: RandomForestClassifier(
                **_merge(
                    {
                        "random_state": 42,
                        "n_estimators": 300,
                        "n_jobs": n_jobs,
                        "class_weight": "balanced_subsample",
                    },
                    p,
                )
            ),
            # SVM is deliberately excluded from candidates -- its only SHAP
            # option is the model-agnostic KernelExplainer, which took
            # ~30-60s per prediction in practice. LightGBM/XGBoost both get
            # shap.TreeExplainer (fast, exact), same tier as RandomForest.
            "LightGBM": lambda **p: LGBMClassifier(
                **_merge(
                    {
                        "random_state": 42,
                        "n_jobs": n_jobs,
                        "class_weight": "balanced",
                        "verbosity": -1,
                    },
                    p,
                )
            ),
            "XGBoost": lambda **p: XGBClassifier(
                **_merge(
                    {
                        "random_state": 42,
                        "n_jobs": n_jobs,
                        "eval_metric": "logloss",
                    },
                    p,
                )
            ),
        }

    def _screen_candidates(
        self, factories, candidate_names, X_train, y_train, preprocessor
    ) -> Dict[str, float]:
        """Cheap, untuned CV pass (default hyperparameters, few folds) to rank
        candidates before committing the full Optuna tuning budget to each.

        Across repeated retrains of this project, RandomForest has never won
        (consistently ~0.001-0.002 ROC-AUC behind LightGBM/XGBoost) while
        taking 5-10x longer to tune due to its larger search space. Rather
        than hardcoding "skip RandomForest", this screens every candidate
        cheaply first and only gives the full tuning budget to whichever ones
        are actually still competitive -- data-driven, so it adapts if a
        future dataset/feature change alters which model is strongest."""
        screening_cfg = self.config["training"].get("screening", {})
        cv = StratifiedKFold(
            n_splits=screening_cfg.get("cv_folds", 3),
            shuffle=True,
            random_state=self.config["project"]["random_seed"],
        )
        scores: Dict[str, float] = {}
        for name in candidate_names:
            if name == "Stacking":
                continue
            factory = factories.get(name)
            if factory is None:
                continue
            try:
                pipeline = Pipeline(
                    [("preprocessor", preprocessor), ("model", factory())]
                )
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=self.config["training"].get("n_jobs", -1),
                )
                scores[name] = float(cv_scores.mean())
                logger.info(f"{name}: screening CV ROC-AUC={scores[name]:.4f}")
            except Exception as e:
                logger.warning(f"Screening failed for {name}, skipping: {e}")
        return scores

    def train_all_models(
        self, X_train, y_train, preprocessor, artifacts_dir: str = "models/artifacts"
    ):
        """Train every configured candidate (optionally tuned via Optuna),
        select the best by the configured primary metric (default ROC-AUC,
        with an F1 tie-breaker), calibrate it, and persist it as
        best_model.pkl. Returns a dict of {model_name: cv_metrics_dict}."""
        factories = self._build_candidate_factories()

        candidate_names = self.config["training"].get(
            "candidate_models", list(factories.keys())
        )

        tuning_cfg = self.config["training"].get("tuning", {})
        tuning_enabled = tuning_cfg.get("enabled", False)
        # CI runs a tiny trial budget as a smoke test (proves the tuning code
        # path works); full local/manual retrains use the larger budget.
        is_ci = bool(os.environ.get("CI") or os.environ.get("CI_SMOKE"))
        n_trials = (
            tuning_cfg.get("n_trials_ci", 3)
            if is_ci
            else tuning_cfg.get("n_trials", 20)
        )
        tuner = HyperparameterTuner(random_seed=self.config["project"]["random_seed"])

        screening_cfg = self.config["training"].get("screening", {})
        screening_scores: Dict[str, float] = {}
        if tuning_enabled and screening_cfg.get("enabled", False):
            screening_scores = self._screen_candidates(
                factories, candidate_names, X_train, y_train, preprocessor
            )

        results: Dict[str, Dict[str, float]] = {}
        fitted_pipelines: Dict[str, Any] = {}

        for name in candidate_names:
            if name == "Stacking":
                continue  # built after the loop, from the other candidates' fitted pipelines

            factory = factories.get(name)
            if factory is None:
                logger.warning(f"Unknown candidate model '{name}', skipping")
                continue

            logger.info(f"Training {name}...")

            candidate_n_trials = n_trials
            if screening_scores and name in screening_scores:
                best_screen = max(screening_scores.values())
                margin = screening_cfg.get("margin", 0.01)
                gap = best_screen - screening_scores[name]
                if gap > margin:
                    candidate_n_trials = min(
                        screening_cfg.get("reduced_trials", n_trials), n_trials
                    )
                    logger.info(
                        f"{name}: screening ROC-AUC {screening_scores[name]:.4f} is "
                        f"{gap:.4f} behind the best screened candidate "
                        f"({best_screen:.4f}) -- reduced tuning budget: "
                        f"{candidate_n_trials} trials instead of {n_trials}"
                    )

            best_params: Dict[str, Any] = {}
            if tuning_enabled:
                try:
                    best_params, best_cv_score = tuner.tune_model(
                        name,
                        factory,
                        preprocessor,
                        X_train,
                        y_train,
                        n_trials=candidate_n_trials,
                        cv_folds=tuning_cfg.get(
                            "cv_folds", self.config["training"]["cv_folds"]
                        ),
                        scoring=tuning_cfg.get("scoring", "roc_auc"),
                    )
                    logger.info(
                        f"{name}: tuned params={best_params} (tuning CV "
                        f"{tuning_cfg.get('scoring', 'roc_auc')}={best_cv_score:.4f})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Hyperparameter tuning failed for {name}, "
                        f"falling back to defaults: {e}"
                    )

            model = factory(**best_params)
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            fitted_pipeline, cv_metrics = self.train_model(
                pipeline, X_train, y_train, name
            )
            results[name] = cv_metrics
            fitted_pipelines[name] = fitted_pipeline

        if "Stacking" in candidate_names:
            self._train_stacking(
                fitted_pipelines, X_train, y_train, results, candidate_names
            )

        best_name = self._select_best(results)
        best_model = fitted_pipelines.get(best_name)

        if best_model is not None:
            calib_cfg = self.config["training"].get("calibration", {})
            if calib_cfg.get("enabled", False):
                logger.info(
                    f"Calibrating best model ({best_name}) via "
                    f"{calib_cfg.get('method', 'isotonic')}"
                )
                best_model = CalibratedClassifierCV(
                    estimator=best_model,
                    method=calib_cfg.get("method", "isotonic"),
                    cv=calib_cfg.get("cv", 5),
                )
                best_model.fit(X_train, y_train)

            import joblib

            os.makedirs(artifacts_dir, exist_ok=True)
            joblib.dump(best_model, os.path.join(artifacts_dir, "best_model.pkl"))
            logger.info(f"Best model saved: {best_name}")

            self._save_shap_background(X_train, artifacts_dir)

        return results

    def _train_stacking(
        self,
        fitted_pipelines: Dict[str, Any],
        X_train,
        y_train,
        results: Dict[str, Dict[str, float]],
        candidate_names,
    ) -> None:
        """Add a stacking-ensemble candidate built from the other already-tuned
        candidates' pipelines -- reuses their tuned hyperparameters (no
        re-tuning), only the meta-learner is fit fresh.

        Evaluated on a single stratified train/validation split rather than a
        full outer k-fold: a StackingClassifier already runs its own internal
        k-fold CV to build leakage-free meta-features, so wrapping it in
        another outer k-fold would multiply cost by roughly folds^2 for a
        variance estimate this dataset's size doesn't need (see the other
        candidates' cv_*_std values, all under 0.004)."""
        base_names = [
            n for n in candidate_names if n != "Stacking" and n in fitted_pipelines
        ]
        if len(base_names) < 2:
            logger.warning(
                "Stacking needs at least 2 trained base candidates, skipping"
            )
            return

        try:
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

            estimators = [(name, fitted_pipelines[name]) for name in base_names]
            stacking_cv = self.config["training"].get("stacking_cv_folds", 3)
            stacking_model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                cv=stacking_cv,
                n_jobs=self.config["training"].get("n_jobs", -1),
                passthrough=False,
            )

            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=self.config["project"]["random_seed"],
                stratify=y_train,
            )

            with mlflow.start_run(run_name="Stacking"):
                stacking_model.fit(X_fit, y_fit)
                val_pred = stacking_model.predict(X_val)
                val_proba = stacking_model.predict_proba(X_val)[:, 1]

                stack_metrics = {
                    "cv_accuracy_mean": float(accuracy_score(y_val, val_pred)),
                    "cv_accuracy_std": 0.0,
                    "cv_f1_mean": float(f1_score(y_val, val_pred)),
                    "cv_f1_std": 0.0,
                    "cv_roc_auc_mean": float(roc_auc_score(y_val, val_proba)),
                    "cv_roc_auc_std": 0.0,
                }
                for key, value in stack_metrics.items():
                    mlflow.log_metric(key, value)

                # Refit on the full training split for the final artifact,
                # matching how every other candidate is trained.
                stacking_model.fit(X_train, y_train)
                mlflow.sklearn.log_model(
                    stacking_model, "model", serialization_format="cloudpickle"
                )

            results["Stacking"] = stack_metrics
            fitted_pipelines["Stacking"] = stacking_model
            logger.info(
                f"Stacking (base: {', '.join(base_names)}): "
                f"single-split val ROC-AUC={stack_metrics['cv_roc_auc_mean']:.4f} "
                f"F1={stack_metrics['cv_f1_mean']:.4f} "
                f"Accuracy={stack_metrics['cv_accuracy_mean']:.4f}"
            )
        except Exception as e:
            logger.warning(f"Stacking ensemble failed, skipping: {e}")

    def _select_best(self, results: Dict[str, Dict[str, float]]) -> Any:
        """Select the best candidate by `training.model_selection.primary_metric`
        (default roc_auc), breaking near-ties (within `tie_tolerance`) using
        `tie_breaker` (default f1). Replaces the old "highest raw CV
        accuracy wins" selection."""
        if not results:
            return None

        selection_cfg = self.config["training"].get("model_selection", {})
        primary = selection_cfg.get("primary_metric", "roc_auc")
        tie_breaker = selection_cfg.get("tie_breaker", "f1")
        tolerance = selection_cfg.get("tie_tolerance", 0.0)

        def metric(name: str, key: str) -> float:
            return results[name].get(f"cv_{key}_mean", 0.0)

        best_name = None
        for name in results:
            if best_name is None:
                best_name = name
                continue
            cur = metric(name, primary)
            best = metric(best_name, primary)
            if cur > best + tolerance:
                best_name = name
            elif abs(cur - best) <= tolerance and metric(name, tie_breaker) > metric(
                best_name, tie_breaker
            ):
                best_name = name

        return best_name

    def _save_shap_background(
        self, X_train, artifacts_dir: str, sample_size: int = 100
    ) -> None:
        """Persist a small stratified-by-index sample of raw training rows as the
        SHAP background/reference dataset, so serving doesn't need the full
        training set in memory. Column order is preserved exactly as trained."""
        import joblib
        import os

        n = min(sample_size, len(X_train))
        background = X_train.sample(
            n=n, random_state=self.config["project"]["random_seed"]
        )
        path = os.path.join(artifacts_dir, "shap_background.pkl")
        joblib.dump(background, path)
        logger.info(f"SHAP background sample saved: {path} ({n} rows)")
