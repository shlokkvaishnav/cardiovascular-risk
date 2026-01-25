from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering and preprocessing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
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
            X, y, test_size=test_size, random_state=self.config['project']['random_seed']
        )
        
        # Build preprocessor
        self.preprocessor = self._build_preprocessor()
        
        # Fit and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test

    def _build_preprocessor(self):
        """Build column transformer for preprocessing"""
        numeric_features = self.config['preprocessing']['numerical_features']
        categorical_features = self.config['preprocessing']['categorical_features']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['preprocessing']['imputation_strategy']['numerical'])),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.config['preprocessing']['imputation_strategy']['categorical'])),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        return preprocessor
