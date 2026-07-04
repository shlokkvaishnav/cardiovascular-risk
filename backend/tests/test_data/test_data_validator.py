import pandas as pd
import pytest

from src.data.data_validator import DataValidator


@pytest.fixture
def config():
    return {
        "preprocessing": {
            "numerical_features": ["age", "height", "weight", "bmi", "ap_hi", "ap_lo"],
            "categorical_features": ["sex", "cholesterol", "gluc", "smoke", "alco", "active"],
        },
        "validation": {
            "ranges": {
                "age": {"min": 18, "max": 100},
                "height": {"min": 120, "max": 220},
                "weight": {"min": 30, "max": 250},
                "bmi": {"min": 12, "max": 80},
                "ap_hi": {"min": 70, "max": 240},
                "ap_lo": {"min": 40, "max": 160},
                "sex": {"min": 0, "max": 1},
                "cholesterol": {"min": 1, "max": 3},
                "gluc": {"min": 1, "max": 3},
                "smoke": {"min": 0, "max": 1},
                "alco": {"min": 0, "max": 1},
                "active": {"min": 0, "max": 1},
                "target": {"min": 0, "max": 1},
            }
        },
    }


@pytest.fixture
def valid_df():
    # Every column has >1 unique value: the validator's statistical checks
    # flag constant columns as a real data-quality issue (correctly).
    return pd.DataFrame({
        "age": [45, 58, 62, 33],
        "sex": [0, 1, 1, 0],
        "height": [165, 175, 180, 160],
        "weight": [60.0, 85.0, 78.0, 55.0],
        "bmi": [22.0, 27.8, 24.1, 21.5],
        "ap_hi": [110, 145, 130, 118],
        "ap_lo": [70, 90, 85, 75],
        "cholesterol": [1, 2, 1, 3],
        "gluc": [1, 1, 2, 1],
        "smoke": [0, 0, 1, 0],
        "alco": [0, 0, 0, 1],
        "active": [1, 1, 0, 1],
        "target": [0, 1, 0, 0],
    })


def test_validation_rules_built_from_config(config):
    validator = DataValidator(config)
    assert validator.validation_rules["age"]["type"] == "numeric"
    assert validator.validation_rules["sex"]["type"] == "categorical"
    assert validator.validation_rules["age"]["min"] == 18
    assert validator.validation_rules["age"]["max"] == 100
    assert "target" in validator.validation_rules


def test_validate_dataframe_passes_on_clean_data(config, valid_df):
    validator = DataValidator(config)
    is_valid, errors = validator.validate_dataframe(valid_df)
    assert is_valid
    assert errors == []


def test_validate_dataframe_flags_out_of_range_values(config, valid_df):
    validator = DataValidator(config)
    bad_df = valid_df.copy()
    bad_df.loc[0, "ap_hi"] = 500  # way out of range

    is_valid, errors = validator.validate_dataframe(bad_df)
    assert not is_valid
    assert any("ap_hi" in e for e in errors)


def test_validate_dataframe_flags_missing_columns(config, valid_df):
    validator = DataValidator(config)
    incomplete_df = valid_df.drop(columns=["ap_hi"])

    is_valid, errors = validator.validate_dataframe(incomplete_df)
    assert not is_valid
    assert any("Missing required columns" in e for e in errors)


def test_validate_single_instance_valid(config):
    validator = DataValidator(config)
    is_valid, errors = validator.validate_single_instance({"age": 45, "ap_hi": 120})
    assert is_valid
    assert errors == []


def test_validate_single_instance_out_of_range(config):
    validator = DataValidator(config)
    is_valid, errors = validator.validate_single_instance({"age": 200})
    assert not is_valid
    assert any("age" in e for e in errors)


def test_clean_data_removes_out_of_range_and_duplicate_rows(config, valid_df):
    validator = DataValidator(config)
    dirty_df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)  # duplicate row
    dirty_df.loc[len(dirty_df)] = dirty_df.iloc[0]
    dirty_df.loc[len(dirty_df) - 1, "ap_hi"] = 999  # out-of-range row

    cleaned = validator.clean_data(dirty_df)
    assert len(cleaned) < len(dirty_df)
    assert (cleaned["ap_hi"] <= 240).all()
    assert not cleaned.duplicated().any()


def test_get_data_quality_report_shape(config, valid_df):
    validator = DataValidator(config)
    report = validator.get_data_quality_report(valid_df)
    assert report["total_rows"] == len(valid_df)
    assert report["total_columns"] == len(valid_df.columns)
    assert "age" in report["summary_statistics"]
