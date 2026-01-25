from pathlib import Path
from typing import Tuple
import pandas as pd
import logging
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

class DataSchema(BaseModel):
    """Pydantic model for data validation"""
    age: float
    sex: int
    cp: float
    trestbps: float
    chol: float
    fbs: int
    restecg: float
    thalach: float
    exang: int
    oldpeak: float
    slope: float
    ca: float
    thal: float
    
    @validator('age')
    def validate_age(cls, v):
        if not 0 < v < 120:
            raise ValueError('Age must be between 0 and 120')
        return v

class DataLoader:
    """Robust data loader with validation"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def load_data(self, path: Path) -> pd.DataFrame:
        """Load and validate data"""
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded data with shape {df.shape}")
            return self._preprocess_raw(df)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
            
    def _preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial preprocessing"""
        if 'thalch' in df.columns:
            df.rename(columns={'thalch': 'thalach'}, inplace=True)
            
        if 'num' in df.columns:
            df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
            
        drop_cols = ['id', 'dataset', 'num']
        df.drop(columns=[c for c in drop_cols if c in df.columns], 
                inplace=True)
        
        return df
