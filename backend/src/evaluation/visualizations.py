"""
Visualization utilities for model evaluation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc
)
from typing import Optional, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

class ModelVisualizer:
    """Create visualizations for model evaluation"""
    
    def __init__(self, output_dir: Path = Path("logs/visualizations")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None
    ):
        """Plot confusion matrix heatmap"""
        if labels is None:
            labels = ["No Disease", "Disease"]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[Path] = None
    ):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return plt.gcf()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: Optional[Path] = None
    ):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        return plt.gcf()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = "Feature Importance",
        top_n: int = 20,
        save_path: Optional[Path] = None
    ):
        """Plot feature importance"""
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_model_comparison(
        self,
        model_names: List[str],
        metrics_dict: dict,
        metric_name: str = "accuracy",
        title: Optional[str] = None,
        save_path: Optional[Path] = None
    ):
        """Plot comparison of multiple models"""
        if title is None:
            title = f"Model Comparison - {metric_name.capitalize()}"
        
        values = [metrics_dict[name].get(metric_name, 0) for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, values, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.ylabel(metric_name.capitalize(), fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1.0])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return plt.gcf()
    
    def create_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        model_name: str = "model",
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None
    ):
        """Create comprehensive evaluation report with all plots"""
        report_dir = self.output_dir / model_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            title=f"Confusion Matrix - {model_name}",
            save_path=report_dir / "confusion_matrix.png"
        )
        plt.close()
        
        # ROC Curve
        if y_pred_proba is not None:
            self.plot_roc_curve(
                y_true, y_pred_proba,
                title=f"ROC Curve - {model_name}",
                save_path=report_dir / "roc_curve.png"
            )
            plt.close()
            
            # Precision-Recall Curve
            self.plot_precision_recall_curve(
                y_true, y_pred_proba,
                title=f"Precision-Recall Curve - {model_name}",
                save_path=report_dir / "pr_curve.png"
            )
            plt.close()
        
        # Feature Importance
        if feature_names is not None and feature_importances is not None:
            self.plot_feature_importance(
                feature_names, feature_importances,
                title=f"Feature Importance - {model_name}",
                save_path=report_dir / "feature_importance.png"
            )
            plt.close()
        
        logger.info(f"Evaluation report created in {report_dir}")
