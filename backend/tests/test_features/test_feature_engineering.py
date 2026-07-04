import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def config():
    return {
        "project": {"random_seed": 42},
        "data": {"test_size": 0.3},
        "preprocessing": {
            "numerical_features": ["age", "height", "weight", "bmi", "ap_hi", "ap_lo"],
            "categorical_features": ["sex", "cholesterol", "gluc", "smoke", "alco", "active"],
            "imputation_strategy": {"numerical": "median", "categorical": "most_frequent"},
        },
        "training": {"n_jobs": 1},
    }


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "sex": rng.integers(0, 2, n),
        "height": rng.integers(150, 200, n),
        "weight": rng.uniform(50, 110, n),
        "bmi": rng.uniform(18, 35, n),
        "ap_hi": rng.integers(90, 180, n),
        "ap_lo": rng.integers(60, 110, n),
        "cholesterol": rng.integers(1, 4, n),
        "gluc": rng.integers(1, 4, n),
        "smoke": rng.integers(0, 2, n),
        "alco": rng.integers(0, 2, n),
        "active": rng.integers(0, 2, n),
        "target": rng.integers(0, 2, n),
    })


def test_prepare_features_splits_and_drops_target(config, raw_df):
    engineer = FeatureEngineer(config)
    X_train, X_test, y_train, y_test = engineer.prepare_features(raw_df)

    assert "target" not in X_train.columns
    assert "target" not in X_test.columns
    assert len(X_train) + len(X_test) == len(raw_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_prepare_features_raises_without_target_column(config, raw_df):
    engineer = FeatureEngineer(config)
    with pytest.raises(ValueError, match="Target column"):
        engineer.prepare_features(raw_df.drop(columns=["target"]))


def test_prepare_features_respects_test_size(config, raw_df):
    config["data"]["test_size"] = 0.2
    engineer = FeatureEngineer(config)
    X_train, X_test, _, _ = engineer.prepare_features(raw_df)
    assert abs(len(X_test) / len(raw_df) - 0.2) < 0.05


def test_build_preprocessor_transforms_expected_shape(config, raw_df):
    engineer = FeatureEngineer(config)
    X_train, X_test, y_train, y_test = engineer.prepare_features(raw_df)
    preprocessor = engineer.build_preprocessor()

    transformed = preprocessor.fit_transform(X_train)
    # 6 numerical (passthrough/scaled) + one-hot expansion of 6 categorical columns
    assert transformed.shape[0] == len(X_train)
    assert transformed.shape[1] >= 6  # at least the numerical columns


def test_build_preprocessor_handles_missing_values(config, raw_df):
    engineer = FeatureEngineer(config)
    df_with_na = raw_df.copy()
    df_with_na.loc[0, "age"] = np.nan
    df_with_na.loc[1, "sex"] = np.nan

    X_train, X_test, y_train, y_test = engineer.prepare_features(df_with_na)
    preprocessor = engineer.build_preprocessor()

    # Should not raise despite missing values (imputers handle them)
    transformed = preprocessor.fit_transform(X_train)
    assert not np.isnan(transformed).any()
