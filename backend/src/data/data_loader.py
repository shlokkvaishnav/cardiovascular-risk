from pathlib import Path
from typing import Tuple
import pandas as pd
import logging
from pydantic import BaseModel, field_validator

from ..features.derived import compute_derived_features

logger = logging.getLogger(__name__)


class DataSchema(BaseModel):
    """Pydantic model for validating a single row of the Kaggle cardiovascular
    lifestyle dataset (age/sex/BMI/blood-pressure/lifestyle features)."""

    age: float
    sex: int
    height: float
    weight: float
    bmi: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int

    @field_validator("age")
    def validate_age(cls, v):
        if not 0 < v < 120:
            raise ValueError("Age must be between 0 and 120")
        return v


class DataLoader:
    """Robust data loader with validation. Supports two dataset schemas,
    auto-detected by column presence:
      - Kaggle cardiovascular lifestyle dataset (cardio_train.csv, primary/default)
      - Legacy UCI heart_disease_uci.csv (kept for reference/comparison)
    """

    def __init__(self, config: dict):
        self.config = config

    def load_data(self, path: Path) -> pd.DataFrame:
        """Load and validate data"""
        try:
            delimiter = self.config.get("data", {}).get("raw_delimiter", ",")
            df = pd.read_csv(path, sep=delimiter)
            logger.info(f"Loaded data with shape {df.shape}")
            if "cardio" in df.columns:
                return self._preprocess_cardio_lifestyle(df)
            return self._preprocess_uci(df)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    # ------------------------------------------------------------------
    # Kaggle cardiovascular lifestyle dataset (primary)
    # ------------------------------------------------------------------
    def _preprocess_cardio_lifestyle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the Kaggle cardiovascular disease dataset.

        Raw columns: id, age (days), gender (1=female, 2=male), height (cm),
        weight (kg), ap_hi/ap_lo (systolic/diastolic BP), cholesterol (1-3),
        gluc (1-3), smoke, alco, active, cardio (target).
        """
        df = df.copy()

        # age is stored in days in the source dataset
        df["age"] = pd.to_numeric(df["age"], errors="coerce") / 365.25

        # Normalize gender (1=female, 2=male) to this project's sex convention
        # (0=female, 1=male), matching the API schema elsewhere.
        df["sex"] = pd.to_numeric(df["gender"], errors="coerce").map({1: 0, 2: 1})

        numeric_cols = ["height", "weight", "ap_hi", "ap_lo"]
        categorical_cols = ["cholesterol", "gluc", "smoke", "alco", "active"]
        for col in numeric_cols + categorical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = self._clean_blood_pressure(df)
        df = self._clean_height_weight(df)

        # BMI plus, when enabled, pulse pressure/BP-category/BMI-category/
        # age-bucket/health-risk-score/BMI-BP-interaction -- shared with the
        # serving path (app._to_feature_vector) via compute_derived_features
        # so the two can never drift apart.
        fe_enabled = (
            self.config.get("preprocessing", {})
            .get("feature_engineering", {})
            .get("enabled", True)
        )
        df = compute_derived_features(df, feature_engineering_enabled=fe_enabled)

        df["target"] = (
            pd.to_numeric(df["cardio"], errors="coerce").fillna(0).astype("int8")
        )

        drop_cols = ["id", "gender", "cardio"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])

        return df

    @staticmethod
    def _clean_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
        """Repair, then clip, ap_hi/ap_lo; drop rows that remain physiologically
        impossible (diastolic > systolic) even after repair.

        EDA on the raw 70k-row Kaggle dataset found the out-of-range values
        (up to 16,020, down to -150) are not random noise but a systematic
        "extra trailing digit" data-entry error: dividing by 10 recovers a
        physiologically plausible value for 96% of out-of-range ap_lo rows
        (926/966) and 72% of out-of-range ap_hi rows (29/40). Repairing this
        (rather than just clipping to the boundary, which discards the true
        value entirely) raises ap_hi's/ap_lo's correlation with the target
        from 0.418/0.296 (clip-only) to 0.430/0.344, and drops ap_lo's
        skewness from 2.79 to 0.33. The hard clip below remains as a fallback
        safety net for whatever the digit-repair doesn't resolve.

        After repair+clip, ~297 rows (0.4% of the dataset) still have
        ap_lo > ap_hi (diastolic exceeding systolic -- physiologically
        impossible, down from 1,234 before repair); those are dropped rather
        than imputed, since there's no principled way to recover them.
        """
        df = df.copy()
        # Cast to float first: the raw column is int64, and assigning the
        # float division result (below) back into an int64 column raises a
        # pandas dtype-compatibility warning today and will be a hard error
        # in a future pandas version.
        df["ap_hi"] = df["ap_hi"].astype("float64").abs()
        df["ap_lo"] = df["ap_lo"].astype("float64").abs()

        hi_repairable = (df["ap_hi"] > 240) & ((df["ap_hi"] / 10).between(70, 240))
        df.loc[hi_repairable, "ap_hi"] = df.loc[hi_repairable, "ap_hi"] / 10
        lo_repairable = (df["ap_lo"] > 160) & ((df["ap_lo"] / 10).between(40, 160))
        df.loc[lo_repairable, "ap_lo"] = df.loc[lo_repairable, "ap_lo"] / 10

        df["ap_hi"] = df["ap_hi"].clip(lower=70, upper=240)
        df["ap_lo"] = df["ap_lo"].clip(lower=40, upper=160)

        return df[df["ap_lo"] <= df["ap_hi"]]

    @staticmethod
    def _clean_height_weight(df: pd.DataFrame) -> pd.DataFrame:
        """Clip height/weight to physiologically plausible adult ranges
        *before* BMI is derived from them -- previously these were only
        flagged post-hoc by DataValidator, never applied to the values BMI/
        bmi_category/bmi_bp_interaction were actually computed from."""
        df = df.copy()
        df["height"] = df["height"].clip(lower=120, upper=220)
        df["weight"] = df["weight"].clip(lower=30, upper=250)
        return df

    # ------------------------------------------------------------------
    # Legacy UCI heart disease dataset (kept for reference/comparison)
    # ------------------------------------------------------------------
    # The raw UCI multi-center CSV stores several fields as human-readable
    # strings (e.g. "Male", "typical angina", "TRUE") rather than the numeric
    # codes the rest of the pipeline expects. These must be mapped explicitly
    # -- a blanket pd.to_numeric(errors='coerce') silently turns every one of
    # these columns into 100% NaN.
    _UCI_CATEGORICAL_MAPS = {
        "sex": {"Male": 1, "Female": 0},
        "cp": {
            "typical angina": 0,
            "atypical angina": 1,
            "non-anginal": 2,
            "asymptomatic": 3,
        },
        "fbs": {"TRUE": 1, "FALSE": 0, True: 1, False: 0},
        "restecg": {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2},
        "exang": {"TRUE": 1, "FALSE": 0, True: 1, False: 0},
        "slope": {"upsloping": 0, "flat": 1, "downsloping": 2},
        "thal": {"normal": 0, "fixed defect": 1, "reversable defect": 2},
    }

    def _preprocess_uci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the legacy UCI heart_disease_uci.csv (13 clinical features)."""
        df = df.copy()
        if "thalch" in df.columns:
            df.rename(columns={"thalch": "thalach"}, inplace=True)

        for col, mapping in self._UCI_CATEGORICAL_MAPS.items():
            if col in df.columns:
                df[col] = (
                    df[col].map(mapping).where(df[col].isin(mapping.keys()), df[col])
                )

        numeric_cols = [
            "age",
            "trestbps",
            "chol",
            "thalach",
            "oldpeak",
            "ca",
            "slope",
            "thal",
            "cp",
            "restecg",
            "exang",
            "fbs",
            "sex",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "num" in df.columns:
            df["num"] = pd.to_numeric(df["num"], errors="coerce").fillna(0)
            df["target"] = (df["num"] > 0).astype("int8")

        drop_cols = ["id", "dataset", "num"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        return df
