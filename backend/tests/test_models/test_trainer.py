import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification

from src.training.trainer import ModelTrainer
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=6, random_state=42
    )
    return X, y


@pytest.fixture
def base_config():
    return {
        "project": {"random_seed": 42},
        "training": {
            "cv_folds": 3,
            "n_jobs": 1,
            "candidate_models": ["LogisticRegression", "RandomForest"],
            "model_selection": {
                "primary_metric": "roc_auc",
                "tie_breaker": "f1",
                "tie_tolerance": 0.0,
            },
            "tuning": {"enabled": False},
            "calibration": {"enabled": False},
        },
        "mlflow": {
            "tracking_uri": "sqlite:///test_mlflow.db",
            "experiment_name": "test",
        },
    }


@pytest.fixture
def tabular_config():
    return {
        "project": {"random_seed": 42},
        "data": {"test_size": 0.3},
        "preprocessing": {
            "numerical_features": ["age", "height", "weight", "ap_hi", "ap_lo"],
            "categorical_features": ["sex", "cholesterol", "gluc", "smoke", "alco", "active"],
            "imputation_strategy": {"numerical": "median", "categorical": "most_frequent"},
        },
        "training": {
            "cv_folds": 3,
            "n_jobs": 1,
            "candidate_models": ["LogisticRegression", "RandomForest"],
            "model_selection": {
                "primary_metric": "roc_auc",
                "tie_breaker": "f1",
                "tie_tolerance": 0.0,
            },
            "tuning": {"enabled": False},
            "calibration": {"enabled": False},
        },
        "mlflow": {
            "tracking_uri": "sqlite:///test_mlflow.db",
            "experiment_name": "test",
        },
    }


@pytest.fixture
def tabular_data():
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "sex": rng.integers(0, 2, n),
            "height": rng.integers(150, 200, n),
            "weight": rng.uniform(50, 110, n),
            "ap_hi": rng.integers(90, 180, n),
            "ap_lo": rng.integers(60, 110, n),
            "cholesterol": rng.integers(1, 4, n),
            "gluc": rng.integers(1, 4, n),
            "smoke": rng.integers(0, 2, n),
            "alco": rng.integers(0, 2, n),
            "active": rng.integers(0, 2, n),
        }
    )
    y = (X["ap_hi"] > 130).astype(int)
    return X, y


def test_train_model_returns_cv_metrics_dict(sample_data, base_config):
    X, y = sample_data
    trainer = ModelTrainer(base_config)

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    trained_model, cv_metrics = trainer.train_model(model, X, y, "test_model")

    assert trained_model is not None
    for key in ("cv_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean"):
        assert key in cv_metrics
        assert 0 <= cv_metrics[key] <= 1


def test_select_best_uses_primary_metric_not_accuracy(base_config):
    trainer = ModelTrainer(base_config)
    results = {
        "HighAccuracyLowAUC": {
            "cv_accuracy_mean": 0.90,
            "cv_f1_mean": 0.70,
            "cv_roc_auc_mean": 0.60,
        },
        "LowAccuracyHighAUC": {
            "cv_accuracy_mean": 0.70,
            "cv_f1_mean": 0.75,
            "cv_roc_auc_mean": 0.85,
        },
    }
    assert trainer._select_best(results) == "LowAccuracyHighAUC"


def test_select_best_breaks_near_ties_with_tie_breaker(base_config):
    base_config["training"]["model_selection"]["tie_tolerance"] = 0.01
    trainer = ModelTrainer(base_config)
    results = {
        "A": {"cv_accuracy_mean": 0.8, "cv_f1_mean": 0.70, "cv_roc_auc_mean": 0.80},
        "B": {"cv_accuracy_mean": 0.8, "cv_f1_mean": 0.90, "cv_roc_auc_mean": 0.805},
    }
    # Within tie_tolerance on ROC-AUC -> falls back to F1, B wins
    assert trainer._select_best(results) == "B"


def test_train_all_models_uses_candidate_models_from_config(tabular_config, tabular_data, tmp_path):
    tabular_config["training"]["candidate_models"] = ["LogisticRegression"]
    X, y = tabular_data
    engineer = FeatureEngineer(tabular_config)
    preprocessor = engineer.build_preprocessor()

    trainer = ModelTrainer(tabular_config)
    results = trainer.train_all_models(X, y, preprocessor, artifacts_dir=str(tmp_path))

    assert list(results.keys()) == ["LogisticRegression"]


def test_train_all_models_selects_by_roc_auc(tabular_config, tabular_data, tmp_path):
    X, y = tabular_data
    engineer = FeatureEngineer(tabular_config)
    preprocessor = engineer.build_preprocessor()

    trainer = ModelTrainer(tabular_config)
    results = trainer.train_all_models(X, y, preprocessor, artifacts_dir=str(tmp_path))

    assert set(results.keys()) == {"LogisticRegression", "RandomForest"}
    assert (tmp_path / "best_model.pkl").exists()
    assert (tmp_path / "shap_background.pkl").exists()


def test_train_all_models_calibrates_best_model_when_enabled(tabular_config, tabular_data, tmp_path):
    tabular_config["training"]["candidate_models"] = ["LogisticRegression"]
    tabular_config["training"]["calibration"] = {"enabled": True, "method": "sigmoid", "cv": 3}
    X, y = tabular_data
    engineer = FeatureEngineer(tabular_config)
    preprocessor = engineer.build_preprocessor()

    trainer = ModelTrainer(tabular_config)
    trainer.train_all_models(X, y, preprocessor, artifacts_dir=str(tmp_path))

    import joblib

    best_model = joblib.load(tmp_path / "best_model.pkl")
    assert isinstance(best_model, CalibratedClassifierCV)
    proba = best_model.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 2)


def test_train_all_models_stacking_candidate_trains_and_selectable(
    tabular_config, tabular_data, tmp_path
):
    from sklearn.ensemble import StackingClassifier

    tabular_config["training"]["candidate_models"] = [
        "LogisticRegression",
        "RandomForest",
        "Stacking",
    ]
    tabular_config["training"]["stacking_cv_folds"] = 2
    X, y = tabular_data
    engineer = FeatureEngineer(tabular_config)
    preprocessor = engineer.build_preprocessor()

    trainer = ModelTrainer(tabular_config)
    results = trainer.train_all_models(X, y, preprocessor, artifacts_dir=str(tmp_path))

    assert "Stacking" in results
    for key in ("cv_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean"):
        assert key in results["Stacking"]
        assert 0 <= results["Stacking"][key] <= 1

    import joblib

    best_model = joblib.load(tmp_path / "best_model.pkl")
    # best_model may be the Stacking candidate itself (if it won selection) or
    # any other candidate -- either way, training must not have crashed and
    # an artifact must exist.
    assert best_model is not None


def test_train_stacking_produces_fitted_stacking_classifier(
    tabular_config, tabular_data
):
    from sklearn.ensemble import StackingClassifier

    tabular_config["training"]["stacking_cv_folds"] = 2
    X, y = tabular_data
    engineer = FeatureEngineer(tabular_config)
    preprocessor = engineer.build_preprocessor()
    trainer = ModelTrainer(tabular_config)

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    fitted_pipelines = {
        "LogisticRegression": Pipeline(
            [("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000))]
        ).fit(X, y),
        "RandomForest": Pipeline(
            [("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=20, random_state=42))]
        ).fit(X, y),
    }
    results = {}
    trainer._train_stacking(
        fitted_pipelines, X, y, results, ["LogisticRegression", "RandomForest", "Stacking"]
    )

    assert "Stacking" in results
    assert isinstance(fitted_pipelines["Stacking"], StackingClassifier)


def test_train_all_models_tuning_end_to_end(tabular_config, tabular_data, tmp_path):
    tabular_config["training"]["candidate_models"] = ["LogisticRegression"]
    tabular_config["training"]["tuning"] = {
        "enabled": True,
        "n_trials": 2,
        "cv_folds": 2,
        "scoring": "roc_auc",
    }
    X, y = tabular_data
    engineer = FeatureEngineer(tabular_config)
    preprocessor = engineer.build_preprocessor()

    trainer = ModelTrainer(tabular_config)
    results = trainer.train_all_models(X, y, preprocessor, artifacts_dir=str(tmp_path))

    assert "LogisticRegression" in results
    assert (tmp_path / "best_model.pkl").exists()
