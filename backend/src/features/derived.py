"""Shared clinical feature derivation.

Single source of truth for every column computed from the raw
age/height/weight/ap_hi/ap_lo fields, so training (DataLoader) and serving
(app._to_feature_vector) can never drift apart the way "bmi" alone used to
be duplicated in two places.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

# Column groups, for callers that need to register these with a
# ColumnTransformer/config alongside the raw features.
#
# Note: mean arterial pressure ("map_pressure") was tried in an earlier
# iteration and dropped -- once ap_hi/ap_lo are properly cleaned (see
# DataLoader._clean_blood_pressure), MAP correlates 0.92/0.94 with ap_hi/
# ap_lo (it's a linear combination of them by construction: ap_lo +
# (ap_hi-ap_lo)/3), so it added almost no information beyond the two raw
# columns already in the feature set while destabilizing the Logistic
# Regression base model's coefficients. pulse_pressure is kept: it's far
# less collinear (0.83 with ap_hi, only 0.23 with ap_lo) and captures real,
# distinct information about BP variability.
DERIVED_NUMERICAL_FEATURES = [
    "bmi",
    "pulse_pressure",
    "health_risk_score",
    "bmi_bp_interaction",
]
DERIVED_CATEGORICAL_FEATURES = ["bp_category", "bmi_category", "age_bucket"]


def _bp_category(ap_hi: pd.Series, ap_lo: pd.Series) -> pd.Series:
    """AHA blood-pressure category, ordinal 0-4:
    0=normal, 1=elevated, 2=stage1 hypertension, 3=stage2 hypertension,
    4=hypertensive crisis."""
    conditions = [
        (ap_hi >= 180) | (ap_lo >= 120),
        (ap_hi >= 140) | (ap_lo >= 90),
        (ap_hi >= 130) | (ap_lo >= 80),
        (ap_hi >= 120) & (ap_lo < 80),
    ]
    choices = [4, 3, 2, 1]
    return pd.Series(
        np.select(conditions, choices, default=0), index=ap_hi.index, dtype="int8"
    )


def _bmi_category(bmi: pd.Series) -> pd.Series:
    """WHO BMI category, ordinal 0-3: underweight/normal/overweight/obese."""
    conditions = [bmi < 18.5, bmi < 25, bmi < 30]
    choices = [0, 1, 2]
    return pd.Series(
        np.select(conditions, choices, default=3), index=bmi.index, dtype="int8"
    )


def _age_bucket(age: pd.Series) -> pd.Series:
    """Decade bin, e.g. 3=30s, 4=40s, ... Clipped to keep a small, bounded
    cardinality regardless of outlier ages."""
    return (age // 10).clip(lower=0, upper=9).astype("int8")


def _health_risk_score(
    cholesterol: pd.Series,
    gluc: pd.Series,
    smoke: pd.Series,
    alco: pd.Series,
    active: pd.Series,
    bp_category: pd.Series,
) -> pd.Series:
    """Interpretable 0-6 composite: sum of binary risk flags (above-normal
    cholesterol, above-normal glucose, smoking, alcohol, inactivity,
    stage-1-or-worse hypertension). Cheap for a linear model to use directly
    without needing to learn the combination itself."""
    score = (
        (cholesterol > 1).astype("int8")
        + (gluc > 1).astype("int8")
        + smoke.astype("int8")
        + alco.astype("int8")
        + (active == 0).astype("int8")
        + (bp_category >= 2).astype("int8")
    )
    return score.astype("int8")


def _bmi_bp_interaction(bmi: pd.Series, bp_category: pd.Series) -> pd.Series:
    """BMI scaled by presence of stage-1-or-worse hypertension, capturing the
    compounding risk of obesity and elevated blood pressure co-occurring
    rather than treating them as independent contributions."""
    return bmi * (bp_category >= 2).astype("float64")


def compute_derived_features(
    df: pd.DataFrame, feature_engineering_enabled: bool = True
) -> pd.DataFrame:
    """Add derived clinical columns to a raw feature frame.

    Expects `age`, `height`, `weight`, `ap_hi`, `ap_lo` to already be present
    (and numeric). `bmi` is always computed if missing, since every serving
    and training path has always needed it; the remaining derived features
    (pulse pressure, MAP, BP/BMI category, age bucket) are gated by
    `feature_engineering_enabled` so the pipeline can be ablated on/off via
    `preprocessing.feature_engineering.enabled` in config.yaml without a
    code change.
    """
    df = df.copy()

    if "bmi" not in df.columns:
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    if not feature_engineering_enabled:
        return df

    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["bp_category"] = _bp_category(df["ap_hi"], df["ap_lo"])
    df["bmi_category"] = _bmi_category(df["bmi"])
    df["age_bucket"] = _age_bucket(df["age"])

    df["health_risk_score"] = _health_risk_score(
        df["cholesterol"],
        df["gluc"],
        df["smoke"],
        df["alco"],
        df["active"],
        df["bp_category"],
    )
    df["bmi_bp_interaction"] = _bmi_bp_interaction(df["bmi"], df["bp_category"])

    return df


def compute_derived_features_from_dict(
    values: Dict[str, Any], feature_engineering_enabled: bool = True
) -> Dict[str, Any]:
    """Single-row convenience wrapper for request-time serving, where the
    caller already has a plain dict (e.g. from a Pydantic request) rather
    than a DataFrame."""
    row = pd.DataFrame([values])
    row = compute_derived_features(row, feature_engineering_enabled)
    return row.iloc[0].to_dict()
