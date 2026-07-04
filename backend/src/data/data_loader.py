from pathlib import Path
from typing import Tuple
import pandas as pd
import logging
from pydantic import BaseModel, field_validator

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

    @field_validator('age')
    def validate_age(cls, v):
        if not 0 < v < 120:
            raise ValueError('Age must be between 0 and 120')
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
            delimiter = self.config.get('data', {}).get('raw_delimiter', ',')
            df = pd.read_csv(path, sep=delimiter)
            logger.info(f"Loaded data with shape {df.shape}")
            if 'cardio' in df.columns:
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
        df['age'] = pd.to_numeric(df['age'], errors='coerce') / 365.25

        # Normalize gender (1=female, 2=male) to this project's sex convention
        # (0=female, 1=male), matching the API schema elsewhere.
        df['sex'] = pd.to_numeric(df['gender'], errors='coerce').map({1: 0, 2: 1})

        numeric_cols = ['height', 'weight', 'ap_hi', 'ap_lo']
        categorical_cols = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
        for col in numeric_cols + categorical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Derived feature: BMI. A handful of rows have implausible height/weight
        # (data-entry errors in the source, e.g. height in the 50s or 250s cm) --
        # these are left for DataValidator's outlier/range checks downstream
        # rather than silently dropped here.
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

        # ap_hi/ap_lo have known data-quality issues in the source dataset
        # (some rows have negative or wildly out-of-range values, e.g. -150 or
        # 16020, likely decimal-point entry errors). Clip to a physiologically
        # plausible range rather than dropping rows, consistent with this
        # pipeline's general "validate and impute, don't discard" approach.
        df['ap_hi'] = df['ap_hi'].clip(lower=70, upper=240)
        df['ap_lo'] = df['ap_lo'].clip(lower=40, upper=160)

        df['target'] = pd.to_numeric(df['cardio'], errors='coerce').fillna(0).astype('int8')

        drop_cols = ['id', 'gender', 'cardio']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

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
        'sex': {'Male': 1, 'Female': 0},
        'cp': {
            'typical angina': 0, 'atypical angina': 1,
            'non-anginal': 2, 'asymptomatic': 3,
        },
        'fbs': {'TRUE': 1, 'FALSE': 0, True: 1, False: 0},
        'restecg': {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2},
        'exang': {'TRUE': 1, 'FALSE': 0, True: 1, False: 0},
        'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
        'thal': {'normal': 0, 'fixed defect': 1, 'reversable defect': 2},
    }

    def _preprocess_uci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the legacy UCI heart_disease_uci.csv (13 clinical features)."""
        df = df.copy()
        if 'thalch' in df.columns:
            df.rename(columns={'thalch': 'thalach'}, inplace=True)

        for col, mapping in self._UCI_CATEGORICAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).where(
                    df[col].isin(mapping.keys()), df[col]
                )

        numeric_cols = [
            'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
            'ca', 'slope', 'thal', 'cp', 'restecg', 'exang', 'fbs', 'sex'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'num' in df.columns:
            df['num'] = pd.to_numeric(df['num'], errors='coerce').fillna(0)
            df['target'] = (df['num'] > 0).astype('int8')

        drop_cols = ['id', 'dataset', 'num']
        df.drop(columns=[c for c in drop_cols if c in df.columns],
                inplace=True)

        return df
