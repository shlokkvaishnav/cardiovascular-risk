import pytest
import numpy as np
from src.training.trainer import ModelTrainer
from sklearn.datasets import make_classification

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X, y

@pytest.fixture
def config():
    return {
        'training': {'cv_folds': 3, 'n_jobs': 1},
        'mlflow': {'tracking_uri': './test_mlruns', 'experiment_name': 'test'}
    }

def test_model_training(sample_data, config):
    X, y = sample_data
    trainer = ModelTrainer(config)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    
    trained_model, score = trainer.train_model(model, X, y, "test_model")
    
    assert trained_model is not None
    assert 0 <= score <= 1
