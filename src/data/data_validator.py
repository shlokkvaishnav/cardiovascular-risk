"""
Data validation module for production ML pipeline
Ensures data quality and consistency
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for heart disease dataset"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = self._define_validation_rules()
        
    def _define_validation_rules(self) -> Dict[str, Dict]:
        """Define validation rules for each feature"""
        return {
            'age': {'min': 0, 'max': 120, 'type': 'numeric'},
            'sex': {'min': 0, 'max': 1, 'type': 'categorical'},
            'cp': {'min': 0, 'max': 3, 'type': 'categorical'},
            'trestbps': {'min': 80, 'max': 200, 'type': 'numeric'},
            'chol': {'min': 100, 'max': 600, 'type': 'numeric'},
            'fbs': {'min': 0, 'max': 1, 'type': 'categorical'},
            'restecg': {'min': 0, 'max': 2, 'type': 'categorical'},
            'thalach': {'min': 60, 'max': 220, 'type': 'numeric'},
            'exang': {'min': 0, 'max': 1, 'type': 'categorical'},
            'oldpeak': {'min': 0, 'max': 10, 'type': 'numeric'},
            'slope': {'min': 0, 'max': 2, 'type': 'categorical'},
            'ca': {'min': 0, 'max': 4, 'type': 'categorical'},
            'thal': {'min': 0, 'max': 3, 'type': 'categorical'},
            'target': {'min': 0, 'max': 1, 'type': 'categorical'}
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate entire dataframe
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for required columns
        required_cols = list(self.validation_rules.keys())
        missing_cols = [col for col in required_cols if col not in df.columns and col != 'target']
        
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors
        
        # Validate each column
        for col, rules in self.validation_rules.items():
            if col not in df.columns:
                continue
                
            # Check data type
            if rules['type'] == 'numeric':
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' should be numeric")
            
            # Check value ranges
            if 'min' in rules and 'max' in rules:
                out_of_range = df[(df[col] < rules['min']) | (df[col] > rules['max'])]
                if len(out_of_range) > 0:
                    errors.append(
                        f"Column '{col}' has {len(out_of_range)} values out of range "
                        f"[{rules['min']}, {rules['max']}]"
                    )
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            for col, count in cols_with_missing.items():
                pct = (count / len(df)) * 100
                if pct > 50:  # More than 50% missing
                    errors.append(f"Column '{col}' has {pct:.1f}% missing values")
                else:
                    logger.warning(f"Column '{col}' has {count} missing values ({pct:.1f}%)")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        # Statistical validation
        stats_errors = self._validate_statistics(df)
        errors.extend(stats_errors)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Data validation passed successfully")
        else:
            logger.error(f"Data validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")
        
        return is_valid, errors
    
    def _validate_statistics(self, df: pd.DataFrame) -> List[str]:
        """Validate statistical properties of the data"""
        errors = []
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                errors.append(f"Column '{col}' has only one unique value")
        
        # Check for extreme outliers (beyond 5 standard deviations)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                outliers = df[np.abs((df[col] - mean) / std) > 5]
                if len(outliers) > 0:
                    logger.warning(f"Column '{col}' has {len(outliers)} extreme outliers")
        
        return errors
    
    def validate_single_instance(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single prediction instance
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        for feature, value in data.items():
            if feature not in self.validation_rules:
                continue
            
            rules = self.validation_rules[feature]
            
            # Check type
            if rules['type'] == 'numeric' and not isinstance(value, (int, float)):
                errors.append(f"Feature '{feature}' should be numeric, got {type(value)}")
            
            # Check range
            if 'min' in rules and 'max' in rules:
                if not (rules['min'] <= value <= rules['max']):
                    errors.append(
                        f"Feature '{feature}' value {value} is out of range "
                        f"[{rules['min']}, {rules['max']}]"
                    )
        
        return len(errors) == 0, errors
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            'duplicates': int(df.duplicated().sum()),
            'column_types': df.dtypes.astype(str).to_dict(),
            'summary_statistics': {}
        }
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report['summary_statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        
        # Add value counts for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        report['categorical_distributions'] = {}
        for col in categorical_cols:
            # Convert both keys and values to native types
            counts = df[col].value_counts().to_dict()
            report['categorical_distributions'][col] = {
                str(k): int(v) for k, v in counts.items()
            }
        
        return report
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing invalid rows
        Returns cleaned dataframe
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Remove rows with values outside valid ranges
        for col, rules in self.validation_rules.items():
            if col not in df_clean.columns:
                continue
            
            if 'min' in rules and 'max' in rules:
                df_clean = df_clean[
                    (df_clean[col] >= rules['min']) & 
                    (df_clean[col] <= rules['max'])
                ]
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} invalid/duplicate rows during cleaning")
        
        return df_clean
