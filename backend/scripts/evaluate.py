"""
Model evaluation script
"""
import argparse
import yaml
import pandas as pd
import joblib
import logging
from pathlib import Path
import sys
import json

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizations import ModelVisualizer
from src.features.feature_engineering import FeatureEngineer
from src.data.data_loader import DataLoader

def main(args):
    """Main evaluation function"""
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logger(config, log_file=str(PROJECT_ROOT / "logs" / "evaluate.log"))
    logger.info("Starting evaluation pipeline...")
    
    # Load model
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        sys.exit(1)
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load and prepare data
    data_loader = DataLoader(config)
    raw_path = Path(config['data']['raw_path'])
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    df = data_loader.load_data(raw_path)
    
    feature_engineer = FeatureEngineer(config)
    X_train, X_test, y_train, y_test = feature_engineer.prepare_features(df)
    
    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    evaluator = ModelEvaluator(config)
    metrics = evaluator.evaluate_model(
        y_test, 
        y_pred, 
        y_pred_proba,
        model_name=args.model_name
    )
    
    # Print performance summary
    summary = evaluator.get_performance_summary(metrics)
    print(summary)
    logger.info(summary)
    
    # Calculate business metrics
    business_metrics = evaluator.calculate_business_metrics(
        y_test, 
        y_pred,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn
    )
    
    logger.info("\nBusiness Metrics:")
    logger.info(f"  Total Cost: ${business_metrics['total_cost']:.2f}")
    logger.info(f"  Avg Cost per Prediction: ${business_metrics['avg_cost_per_prediction']:.2f}")
    
    # Save metrics
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / f"{args.model_name}_metrics.json"
    evaluator.save_metrics(metrics, metrics_path)
    
    # Create visualizations
    if args.create_plots:
        logger.info("Creating visualizations...")
        visualizer = ModelVisualizer(output_dir=output_dir)
        
        feature_names = config['preprocessing']['numerical_features'] + \
                       config['preprocessing']['categorical_features']
        
        # Get feature importances if available
        feature_importances = None
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importances = abs(model.coef_[0])
        
        visualizer.create_evaluation_report(
            y_test,
            y_pred,
            y_pred_proba,
            model_name=args.model_name,
            feature_names=feature_names if feature_importances is not None else None,
            feature_importances=feature_importances
        )
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "artifacts" / "best_model.pkl"),
        help="Path to trained model"
    )
    parser.add_argument(
        "--model-name",
        default="best_model",
        help="Name of the model"
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "logs" / "evaluation"),
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create visualization plots"
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="Cost of false positive"
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=10.0,
        help="Cost of false negative"
    )
    
    args = parser.parse_args()
    main(args)
