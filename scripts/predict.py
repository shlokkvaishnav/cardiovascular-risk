"""
Prediction script for making predictions on new data
"""
import argparse
import yaml
import pandas as pd
import joblib
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.data.data_validator import DataValidator

def main(args):
    """Main prediction function"""
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logger(config, log_file="logs/predict.log")
    logger.info("Starting prediction pipeline...")
    
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        sys.exit(1)
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} instances from {input_path}")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        sys.exit(1)
    
    # Validate data
    validator = DataValidator(config)
    is_valid, errors = validator.validate_dataframe(df)
    
    if not is_valid and not args.skip_validation:
        logger.error("Data validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Make predictions
    try:
        predictions = model.predict(df)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[:, 1]
        else:
            probabilities = predictions.astype(float)
        
        # Create results dataframe
        results = df.copy()
        results['prediction'] = predictions
        results['probability'] = probabilities
        results['risk_level'] = results['probability'].apply(
            lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
        )
        results['timestamp'] = datetime.now().isoformat()
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Print summary
        logger.info("\nPrediction Summary:")
        logger.info(f"  Total instances: {len(results)}")
        logger.info(f"  Predicted positive: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.1f}%)")
        logger.info(f"  Risk levels:")
        logger.info(f"    High:   {(results['risk_level'] == 'High').sum()}")
        logger.info(f"    Medium: {(results['risk_level'] == 'Medium').sum()}")
        logger.info(f"    Low:    {(results['risk_level'] == 'Low').sum()}")
        
        # Save summary as JSON
        if args.save_summary:
            summary = {
                'total_instances': len(results),
                'predicted_positive': int(predictions.sum()),
                'predicted_negative': int(len(predictions) - predictions.sum()),
                'risk_distribution': {
                    'high': int((results['risk_level'] == 'High').sum()),
                    'medium': int((results['risk_level'] == 'Medium').sum()),
                    'low': int((results['risk_level'] == 'Low').sum())
                },
                'timestamp': datetime.now().isoformat()
            }
            
            summary_path = output_path.parent / f"{output_path.stem}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary saved to {summary_path}")
        
        logger.info("Prediction pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on new data")
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        default="data/predictions/predictions.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--model-path",
        default="models/artifacts/best_model.pkl",
        help="Path to trained model"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation"
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save prediction summary as JSON"
    )
    
    args = parser.parse_args()
    main(args)
