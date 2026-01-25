"""
Training script for heart disease prediction models
"""
import argparse
import yaml
from pathlib import Path
import sys
import joblib
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.feature_engineering import FeatureEngineer
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizations import ModelVisualizer
from src.utils.logger import setup_logger

def main(args):
    """Main training function"""
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logger(config, log_file="logs/train.log")
    logger.info("="*60)
    logger.info("Starting training pipeline...")
    logger.info(f"Configuration: {config_path}")
    logger.info("="*60)
    
    try:
        # Load data
        logger.info("\n[1/6] Loading data...")
        data_loader = DataLoader(config)
        df = data_loader.load_data(Path(config['data']['raw_path']))
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Validate data
        if not args.skip_validation:
            logger.info("\n[2/6] Validating data...")
            validator = DataValidator(config)
            is_valid, errors = validator.validate_dataframe(df)
            
            if not is_valid:
                logger.error("Data validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                
                if args.strict:
                    logger.error("Exiting due to validation errors (strict mode)")
                    sys.exit(1)
                else:
                    logger.warning("Continuing despite validation errors...")
            else:
                logger.info("Data validation passed!")
            
            # Generate data quality report
            quality_report = validator.get_data_quality_report(df)
            report_path = Path("logs/data_quality_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(quality_report, f, indent=2)
            logger.info(f"Data quality report saved to {report_path}")
        else:
            logger.info("\n[2/6] Skipping data validation...")
        
        # Feature engineering
        logger.info("\n[3/6] Engineering features...")
        feature_engineer = FeatureEngineer(config)
        X_train, X_test, y_train, y_test = feature_engineer.prepare_features(df)
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Save preprocessor
        preprocessor_path = Path("models/artifacts/preprocessor.pkl")
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(feature_engineer.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Train models
        logger.info("\n[4/6] Training models...")
        trainer = ModelTrainer(config)
        results = trainer.train_all_models(X_train, y_train)
        
        logger.info("\nTraining Results:")
        for model_name, score in results.items():
            logger.info(f"  {model_name}: {score:.4f}")
        
        # Load best model for evaluation
        best_model_path = Path("models/artifacts/best_model.pkl")
        best_model = joblib.load(best_model_path)
        
        # Evaluate on test set
        logger.info("\n[5/6] Evaluating on test set...")
        y_pred = best_model.predict(X_test)
        y_pred_proba = None
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate_model(
            y_test, 
            y_pred, 
            y_pred_proba,
            model_name="best_model"
        )
        
        # Save metrics
        metrics_path = Path("logs/evaluation/best_model_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator.save_metrics(metrics, metrics_path)
        
        # Print summary
        summary = evaluator.get_performance_summary(metrics)
        print("\n" + summary)
        
        # Create visualizations
        if args.create_plots:
            logger.info("\n[6/6] Creating visualizations...")
            visualizer = ModelVisualizer(output_dir=Path("logs/visualizations"))
            
            feature_names = config['preprocessing']['numerical_features'] + \
                           config['preprocessing']['categorical_features']
            
            # Get feature importances if available
            feature_importances = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importances = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                feature_importances = abs(best_model.coef_[0])
            
            visualizer.create_evaluation_report(
                y_test,
                y_pred,
                y_pred_proba,
                model_name="best_model",
                feature_names=feature_names if feature_importances is not None else None,
                feature_importances=feature_importances
            )
            logger.info("Visualizations created successfully!")
        else:
            logger.info("\n[6/6] Skipping visualization creation...")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'config': config,
            'dataset_shape': df.shape,
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'models_trained': list(results.keys()),
            'best_model_metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }
        }
        
        metadata_path = Path("models/artifacts/training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Training metadata saved to {metadata_path}")
        
        logger.info("\n" + "="*60)
        logger.info("Training pipeline completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heart disease prediction models")
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation step"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit on validation errors"
    )
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create visualization plots"
    )
    
    args = parser.parse_args()
    main(args)
