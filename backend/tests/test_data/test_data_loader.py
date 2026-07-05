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
        "bp_category",
        "bmi_category",
        "age_bucket",
        "health_risk_score",
        "bmi_bp_interaction",
    ]:
        assert col in df.columns
    assert "map_pressure" not in df.columns  # dropped: near-redundant with ap_hi/ap_lo

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
        "bp_category",
        "bmi_category",
        "age_bucket",
        "health_risk_score",
        "bmi_bp_interaction",
    ]:
        assert col not in df.columns


def _row(**overrides):
    base = dict(
        id=1,
        age=365.25 * 40,
        gender=2,
        height=175,
        weight=80.0,
        ap_hi=140,
        ap_lo=90,
        cholesterol=1,
        gluc=1,
        smoke=0,
        alco=0,
        active=1,
        cardio=0,
    )
    base.update(overrides)
    return pd.DataFrame([base])


def test_ap_lo_digit_repair_recovers_plausible_value(config):
    """ap_lo=1000 is a systematic 'extra trailing digit' entry error (found
    in 96% of such out-of-range rows in the raw dataset) -- /10 recovers a
    plausible 100, not the arbitrary clip boundary of 160."""
    loader = DataLoader(config)
    result = loader._preprocess_cardio_lifestyle(_row(ap_hi=140, ap_lo=1000))
    assert result.loc[0, "ap_lo"] == 100


def test_ap_hi_digit_repair_recovers_plausible_value(config):
    loader = DataLoader(config)
    result = loader._preprocess_cardio_lifestyle(_row(ap_hi=1600, ap_lo=90))
    assert result.loc[0, "ap_hi"] == 160


def test_unrepairable_out_of_range_values_still_clipped(config):
    """16020 / 10 = 1602, still out of range -- falls back to the hard clip
    rather than being silently dropped or left implausible."""
    loader = DataLoader(config)
    result = loader._preprocess_cardio_lifestyle(_row(ap_hi=16020, ap_lo=90))
    assert result.loc[0, "ap_hi"] == 240


def test_negative_bp_values_recovered_via_abs(config):
    loader = DataLoader(config)
    result = loader._preprocess_cardio_lifestyle(_row(ap_hi=-150, ap_lo=80))
    assert result.loc[0, "ap_hi"] == 150  # abs() recovers a plausible value directly


def test_unrepairable_diastolic_exceeds_systolic_rows_dropped(config):
    """After repair+clip, rows where diastolic still exceeds systolic are
    physiologically impossible and dropped, not imputed or kept as noise."""
    loader = DataLoader(config)
    good = _row(id=1, ap_hi=140, ap_lo=90)
    bad = _row(id=2, ap_hi=90, ap_lo=140)  # unrepairable inconsistency
    combined = pd.concat([good, bad], ignore_index=True)
    result = loader._preprocess_cardio_lifestyle(combined)
    assert len(result) == 1
    assert result.iloc[0]["ap_hi"] == 140


def test_duplicate_rows_dropped(config):
    row = _row(id=1, ap_hi=140, ap_lo=90)
    duplicate = _row(id=2, ap_hi=140, ap_lo=90)  # differs only by id
    combined = pd.concat([row, duplicate], ignore_index=True)
    loader = DataLoader(config)
    result = loader._preprocess_cardio_lifestyle(combined)
    assert len(result) == 1


def test_height_weight_clipped_before_bmi_derivation(config):
    """height=55/weight=10 are physiologically impossible for an adult;
    clipping before BMI derivation (not just post-hoc flagging) keeps BMI
    and its downstream derived features from being computed on garbage."""
    loader = DataLoader(config)
    result = loader._preprocess_cardio_lifestyle(_row(height=55, weight=10))
    assert result.loc[0, "height"] == 120  # clipped to floor
    assert result.loc[0, "weight"] == 30  # clipped to floor
    expected_bmi = 30 / ((120 / 100) ** 2)
    assert result.loc[0, "bmi"] == pytest.approx(expected_bmi)


def test_ap_hi_ap_lo_clipped_before_derivation(config):
    df = pd.DataFrame(
        {
            "id": [1],
            "age": [365.25 * 40],
            "gender": [2],
            "height": [175],
            "weight": [80.0],
            "ap_hi": [16020],  # implausible data-entry error, unrepairable via /10
            "ap_lo": [90],
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

    assert result.loc[0, "ap_hi"] == 240  # clipped upper bound (fallback)
    assert result.loc[0, "pulse_pressure"] == 150
