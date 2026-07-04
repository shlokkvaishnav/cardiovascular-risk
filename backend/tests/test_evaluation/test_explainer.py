import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.explainer import SHAPExplainer, WeightedContributionExplainer
from src.features.feature_engineering import FeatureEngineer

FEATURE_NAMES = [
    "age",
    "sex",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]


@pytest.fixture
def config():
    return {
        "project": {"random_seed": 42},
        "data": {"test_size": 0.3},
        "preprocessing": {
            "numerical_features": ["age", "height", "weight", "ap_hi", "ap_lo"],
            "categorical_features": [
                "sex",
                "cholesterol",
                "gluc",
                "smoke",
                "alco",
                "active",
            ],
            "imputation_strategy": {
                "numerical": "median",
                "categorical": "most_frequent",
            },
        },
        "training": {"n_jobs": 1},
    }


@pytest.fixture
def training_data():
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
    y = (X["ap_hi"] > 130).astype(int)  # deterministic-ish signal for a meaningful fit
    return X, y


def _fit_pipeline(model, config, X, y):
    engineer = FeatureEngineer(config)
    preprocessor = engineer.build_preprocessor()
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, y)
    return pipeline


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: LogisticRegression(max_iter=1000),
        lambda: RandomForestClassifier(n_estimators=50, random_state=42),
    ],
)
def test_explainer_returns_signed_contributions_for_all_model_types(
    config, training_data, model_factory
):
    """Regression test for the fake-explanation bug: the old coefficient-proxy
    implementation silently returned None for RandomForest. Every supported
    model type must return real, non-null, signed SHAP contributions."""
    X, y = training_data
    pipeline = _fit_pipeline(model_factory(), config, X, y)

    background = X.sample(n=50, random_state=42)
    explainer = SHAPExplainer(pipeline, background, list(X.columns))

    row = X.iloc[[0]]
    contributions, baseline = explainer.explain(row)

    assert contributions is not None
    assert len(contributions) > 0
    assert baseline is not None
    assert 0 <= baseline <= 1

    # At least one contribution should be meaningfully non-zero
    values = [v for entry in contributions for v in entry.values()]
    assert any(abs(v) > 1e-9 for v in values)


def test_explainer_unwraps_calibrated_classifier_to_tree_explainer(config, training_data):
    """A CalibratedClassifierCV-wrapped RandomForest must still resolve to the
    fast, exact TreeExplainer -- not silently fall through to the
    30-60s/prediction KernelExplainer fallback."""
    X, y = training_data
    engineer = FeatureEngineer(config)
    preprocessor = engineer.build_preprocessor()
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=50, random_state=42)),
        ]
    )
    calibrated = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3)
    calibrated.fit(X, y)

    background = X.sample(n=30, random_state=42)
    explainer = SHAPExplainer(calibrated, background, list(X.columns))

    assert explainer._is_kernel is False
    contributions, baseline = explainer.explain(X.iloc[[0]])
    assert contributions is not None
    assert baseline is not None


def test_explainer_detects_lightgbm_as_tree_explainer(config, training_data):
    lgb = pytest.importorskip("lightgbm")
    X, y = training_data
    pipeline = _fit_pipeline(
        lgb.LGBMClassifier(n_estimators=50, random_state=42, verbosity=-1), config, X, y
    )
    background = X.sample(n=30, random_state=42)
    explainer = SHAPExplainer(pipeline, background, list(X.columns))

    assert explainer._is_kernel is False
    contributions, baseline = explainer.explain(X.iloc[[0]])
    assert contributions is not None


def test_explainer_detects_xgboost_as_tree_explainer(config, training_data):
    xgb = pytest.importorskip("xgboost")
    X, y = training_data
    pipeline = _fit_pipeline(
        xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric="logloss"),
        config,
        X,
        y,
    )
    background = X.sample(n=30, random_state=42)
    explainer = SHAPExplainer(pipeline, background, list(X.columns))

    assert explainer._is_kernel is False
    contributions, baseline = explainer.explain(X.iloc[[0]])
    assert contributions is not None


def test_weighted_contribution_explainer_resolves_for_stacking(config, training_data):
    """A fitted StackingClassifier must get fast, non-null explanations via
    WeightedContributionExplainer rather than the slow KernelExplainer path."""
    from sklearn.ensemble import StackingClassifier

    X, y = training_data
    engineer = FeatureEngineer(config)
    preprocessor = engineer.build_preprocessor()

    base_pipelines = [
        (
            "lr",
            Pipeline(
                [("preprocessor", engineer.build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]
            ),
        ),
        (
            "rf",
            Pipeline(
                [
                    ("preprocessor", engineer.build_preprocessor()),
                    ("model", RandomForestClassifier(n_estimators=30, random_state=42)),
                ]
            ),
        ),
    ]
    stacking = StackingClassifier(
        estimators=base_pipelines,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=2,
    )
    stacking.fit(X, y)

    background = X.sample(n=30, random_state=42)
    explainer = WeightedContributionExplainer(stacking, background, list(X.columns))

    contributions, baseline = explainer.explain(X.iloc[[0]])
    assert contributions is not None
    assert len(contributions) > 0
    assert baseline is not None
    assert 0 <= baseline <= 1


def test_explainer_handles_invalid_input_gracefully(config, training_data):
    X, y = training_data
    pipeline = _fit_pipeline(LogisticRegression(max_iter=1000), config, X, y)
    background = X.sample(n=20, random_state=42)
    explainer = SHAPExplainer(pipeline, background, list(X.columns))

    # Passing a DataFrame with a missing column should fail gracefully, not raise
    bad_row = X.iloc[[0]].drop(columns=["age"])
    contributions, baseline = explainer.explain(bad_row)
    assert contributions is None
    assert baseline is None
