import mlflow
import logging
from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates model training with MLflow tracking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    def train_model(self, model, X_train, y_train, model_name: str):
        """Train model with MLflow tracking"""

        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(model.get_params())

            # Cross-validation
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=self.config["training"]["cv_folds"],
                scoring="accuracy",
                n_jobs=self.config["training"]["n_jobs"],
            )

            # Log metrics
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())

            # Train final model
            model.fit(X_train, y_train)

            # Log model. Explicit cloudpickle serialization avoids newer
            # mlflow's default skops format, which rejects common sklearn
            # pipeline internals (e.g. numpy.dtype) as "untrusted types".
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")

            logger.info(f"{model_name}: CV Accuracy = {cv_scores.mean():.4f}")

            return model, cv_scores.mean()

    def train_all_models(
        self, X_train, y_train, preprocessor, artifacts_dir: str = "models/artifacts"
    ):
        """Train standard models with preprocessing pipeline.

        Only models with a fast, *exact* SHAP explainer are candidates for the
        served model: RandomForest (TreeExplainer) and LogisticRegression
        (LinearExplainer). SVM was deliberately dropped from selection -- its
        only SHAP option is the model-agnostic KernelExplainer, which took ~30-60s
        per prediction in practice (far past the frontend's request timeout) and
        yields only approximate values. Its marginal CV-accuracy edge (~0.01) did
        not justify making every explained prediction unusably slow.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        base_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(
                random_state=42,
                n_estimators=300,
                n_jobs=self.config["training"].get("n_jobs", -1),
                class_weight="balanced_subsample",
            ),
        }

        models = {
            name: Pipeline([("preprocessor", preprocessor), ("model", model)])
            for name, model in base_models.items()
        }

        results = {}
        best_score = 0
        best_model = None

        for name, model in models.items():
            logger.info(f"Training {name}...")
            trained_model, score = self.train_model(model, X_train, y_train, name)
            results[name] = score

            if score > best_score:
                best_score = score
                best_model = trained_model

        # Save best model to artifacts
        if best_model:
            import joblib
            import os

            os.makedirs(artifacts_dir, exist_ok=True)
            joblib.dump(best_model, os.path.join(artifacts_dir, "best_model.pkl"))
            logger.info(f"Best model saved: {best_model} with score {best_score}")

            self._save_shap_background(X_train, artifacts_dir)

        return results

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
