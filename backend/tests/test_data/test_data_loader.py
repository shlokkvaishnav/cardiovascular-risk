import pandas as pd
import pytest

from src.data.data_loader import DataLoader


@pytest.fixture
def config():
    return {
        "data": {"raw_delimiter": ";"},
        "preprocessing": {"feature_engineering": {"enabled": True}},
    }


@pytest.fixture
def raw_cardio_df():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "age": [365.25 * 50, 365.25 * 60],  # 50 and 60 years, in days
            "gender": [1, 2],  # female, male
            "height": [165, 180],
            "weight": [70.0, 90.0],
            "ap_hi": [150, 110],  # stage2 hypertension / normal
            "ap_lo": [95, 70],
            "cholesterol": [2, 1],
            "gluc": [1, 1],
            "smoke": [0, 1],
            "alco": [0, 0],
            "active": [1, 1],
            "cardio": [1, 0],
        }
    )


def test_derived_features_present_when_enabled(config, raw_cardio_df):
    loader = DataLoader(config)
    df = loader._preprocess_cardio_lifestyle(raw_cardio_df)

    for col in [
        "bmi",
        "pulse_pressure",
        "map_pressure",
        "bp_category",
        "bmi_category",
        "age_bucket",
        "health_risk_score",
        "bmi_bp_interaction",
    ]:
        assert col in df.columns

    # Row 0: ap_hi=150, ap_lo=95 -> pulse_pressure=55, stage2 hypertension (ap_lo>=90)
    assert df.loc[0, "pulse_pressure"] == 55
    assert df.loc[0, "bp_category"] == 3
    # Row 0: cholesterol=2 (>1), gluc=1, smoke=0, alco=0, active=1, bp_category=3 (>=2)
    # -> risk flags: cholesterol + bp_category = 2
    assert df.loc[0, "health_risk_score"] == 2
    # bmi_bp_interaction = bmi * 1 (bp_category>=2) for row 0
    assert df.loc[0, "bmi_bp_interaction"] == pytest.approx(df.loc[0, "bmi"])

    # Row 1: ap_hi=110, ap_lo=70 -> normal, smoke=1 -> risk score includes smoking only
    assert df.loc[1, "bp_category"] == 0
    assert df.loc[1, "health_risk_score"] == 1
    assert df.loc[1, "bmi_bp_interaction"] == 0


def test_derived_features_absent_when_disabled(raw_cardio_df):
    config = {
        "data": {"raw_delimiter": ";"},
        "preprocessing": {"feature_engineering": {"enabled": False}},
    }
    loader = DataLoader(config)
    df = loader._preprocess_cardio_lifestyle(raw_cardio_df)

    assert "bmi" in df.columns  # bmi always computed
    for col in [
        "pulse_pressure",
        "map_pressure",
        "bp_category",
        "bmi_category",
        "age_bucket",
        "health_risk_score",
        "bmi_bp_interaction",
    ]:
        assert col not in df.columns


def test_ap_hi_ap_lo_clipped_before_derivation(config):
    df = pd.DataFrame(
        {
            "id": [1],
            "age": [365.25 * 40],
            "gender": [2],
            "height": [175],
            "weight": [80.0],
            "ap_hi": [16020],  # implausible data-entry error
            "ap_lo": [-150],
            "cholesterol": [1],
            "gluc": [1],
            "smoke": [0],
            "alco": [0],
            "active": [1],
            "cardio": [0],
        }
    )
    loader = DataLoader(config)
    result = loader._preprocess_cardio_lifestyle(df)

    assert result.loc[0, "ap_hi"] == 240  # clipped upper bound
    assert result.loc[0, "ap_lo"] == 40  # clipped lower bound
    assert result.loc[0, "pulse_pressure"] == 200
