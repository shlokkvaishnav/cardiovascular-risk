from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass
