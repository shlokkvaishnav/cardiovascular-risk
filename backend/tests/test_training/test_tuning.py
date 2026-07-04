import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.tuning import HyperparameterTuner


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=150, n_features=8, n_informative=5, random_state=42
    )
    return X, y


def test_tune_model_returns_valid_params_and_score(sample_data):
    X, y = sample_data
    tuner = HyperparameterTuner(random_seed=42)

    best_params, best_score = tuner.tune_model(
        "LogisticRegression",
        lambda **p: LogisticRegression(**p),
        StandardScaler(),
        X,
        y,
        n_trials=2,
        cv_folds=2,
        scoring="roc_auc",
    )

    assert "C" in best_params
    model = LogisticRegression(**best_params)
    pipeline = Pipeline([("preprocessor", StandardScaler()), ("model", model)])
    pipeline.fit(X, y)
    assert 0 <= best_score <= 1


def test_tune_model_raises_for_unknown_model(sample_data):
    X, y = sample_data
    tuner = HyperparameterTuner(random_seed=42)
    with pytest.raises(ValueError):
        tuner.tune_model(
            "NotAModel",
            lambda **p: None,
            StandardScaler(),
            X,
            y,
            n_trials=1,
        )
