from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering and preprocessing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
        """
        Prepare features for training
        Returns: X_train, X_test, y_train, y_test
        """
        target_col = 'target' 
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split data
        test_size = self.config['data']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.config['project']['random_seed'],
            stratify=y if y.nunique() > 1 else None
        )
        
        return X_train, X_test, y_train, y_test

    def build_preprocessor(self):
        """Build column transformer for preprocessing"""
        numeric_features = self.config['preprocessing']['numerical_features']
        categorical_features = self.config['preprocessing']['categorical_features']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['preprocessing']['imputation_strategy']['numerical'])),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['preprocessing']['imputation_strategy']['categorical'])),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop',
            n_jobs=self.config['training'].get('n_jobs', None)
        )
            
        return preprocessor
