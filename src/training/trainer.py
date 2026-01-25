import mlflow
import logging
from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Orchestrates model training with MLflow tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
    def train_model(self, model, X_train, y_train, model_name: str):
        """Train model with MLflow tracking"""
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(model.get_params())
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config['training']['cv_folds'],
                scoring='accuracy',
                n_jobs=self.config['training']['n_jobs']
            )
            
            # Log metrics
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())
            
            # Train final model
            model.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"{model_name}: CV Accuracy = {cv_scores.mean():.4f}")
            
            return model, cv_scores.mean()

    def train_all_models(self, X_train, y_train):
        """Train standard models"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
        
        results = {}
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            trained_model, score = self.train_model(model, X_train, y_train, name)
            results[name] = score
            
            if score > best_score:
                best_score = score
                best_model = trained_model
                
        # Save best model to artifacts
        if best_model:
            import joblib
            import os
            os.makedirs("models/artifacts", exist_ok=True)
            joblib.dump(best_model, "models/artifacts/best_model.pkl")
            logger.info(f"Best model saved: {best_model} with score {best_score}")
            
        return results
