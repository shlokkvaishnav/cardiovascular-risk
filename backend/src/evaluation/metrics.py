"""
Evaluation metrics and model performance tracking
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any, Tuple, Optional
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and metrics tracking"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_history = []

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model being evaluated

        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(y_true),
        }

        # Basic classification metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        )
        metrics["f1_score"] = float(
            f1_score(y_true, y_pred, average="binary", zero_division=0)
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

        # Specificity and sensitivity
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
                metrics["average_precision"] = float(
                    average_precision_score(y_true, y_pred_proba)
                )

                # Calculate optimal threshold
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
                optimal_idx = np.argmax(tpr - fpr)
                metrics["optimal_threshold"] = float(thresholds[optimal_idx])

            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")

            try:
                calibration = self.assess_calibration(y_true, y_pred_proba)
                metrics["brier_score"] = calibration["brier_score"]
                metrics["calibration_curve"] = calibration["calibration_curve"]
            except Exception as e:
                logger.warning(f"Could not calculate calibration metrics: {e}")

        # Classification report
        try:
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            metrics["classification_report"] = report
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")

        # Log metrics
        logger.info(f"Model Evaluation - {model_name}")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if "roc_auc" in metrics:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Store in history
        self.metrics_history.append(metrics)

        return metrics

    def assess_calibration(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """Standalone calibration assessment: Brier score (lower is better,
        0 = perfect) plus a reliability-diagram curve (predicted probability
        vs. observed frequency per bin). Clinically, this matters more than
        accuracy/F1 for a risk-prediction tool -- a well-calibrated 30%
        should mean roughly 3 in 10 such patients actually have the
        condition, independent of the classification threshold."""
        brier = float(brier_score_loss(y_true, y_pred_proba))
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
        )
        return {
            "brier_score": brier,
            "calibration_curve": {
                "prob_true": [float(v) for v in prob_true],
                "prob_pred": [float(v) for v in prob_pred],
            },
        }

    def compare_models(self, metrics_list: list) -> Dict[str, Any]:
        """
        Compare multiple models based on their metrics

        Args:
            metrics_list: List of metrics dictionaries from evaluate_model

        Returns:
            Comparison summary
        """
        if not metrics_list:
            return {}

        comparison = {
            "models": [m["model_name"] for m in metrics_list],
            "best_by_metric": {},
        }

        # Find best model for each metric
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        for metric_key in metric_keys:
            values = [(m["model_name"], m.get(metric_key, 0)) for m in metrics_list]
            if values:
                best_model, best_value = max(values, key=lambda x: x[1])
                comparison["best_by_metric"][metric_key] = {
                    "model": best_model,
                    "value": best_value,
                }

        return comparison

    def save_metrics(self, metrics: Dict[str, Any], filepath: Path):
        """Save metrics to JSON file"""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def load_metrics(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load metrics from JSON file"""
        try:
            with open(filepath, "r") as f:
                metrics = json.load(f)
            logger.info(f"Metrics loaded from {filepath}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return None

    def evaluate_by_subgroup(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        subgroup_values,
        subgroup_name: str,
    ) -> Dict[str, Any]:
        """Break down accuracy/ROC-AUC/Brier score by each unique value of
        `subgroup_values` (e.g. sex, or age_bucket). Aggregate metrics can
        mask meaningful miscalibration or discrimination gaps in specific
        subgroups -- a documented pitfall in cardiovascular-risk-prediction
        literature. This is a reporting/audit function only: it does not
        change training or selection behavior, it surfaces a signal that a
        human should review."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        subgroup_values = np.asarray(subgroup_values)
        if y_pred_proba is not None:
            y_pred_proba = np.asarray(y_pred_proba)

        result: Dict[str, Any] = {"subgroup_name": subgroup_name, "groups": {}}

        for value in sorted(set(subgroup_values.tolist())):
            mask = subgroup_values == value
            n = int(mask.sum())
            group: Dict[str, Any] = {"n": n}
            if n == 0:
                result["groups"][str(value)] = group
                continue

            group["accuracy"] = float(accuracy_score(y_true[mask], y_pred[mask]))

            group_proba = y_pred_proba[mask] if y_pred_proba is not None else None
            if group_proba is not None and len(set(y_true[mask].tolist())) > 1:
                try:
                    group["roc_auc"] = float(roc_auc_score(y_true[mask], group_proba))
                    group["brier_score"] = float(
                        brier_score_loss(y_true[mask], group_proba)
                    )
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        f"Could not compute probability metrics for "
                        f"{subgroup_name}={value}: {e}"
                    )

            result["groups"][str(value)] = group

        return result

    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
    ) -> Dict[str, float]:
        """
        Calculate business-oriented metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_fp: Cost of false positive (unnecessary treatment)
            cost_fn: Cost of false negative (missed diagnosis)

        Returns:
            Business metrics including total cost
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        total_cost = (fp * cost_fp) + (fn * cost_fn)
        avg_cost_per_prediction = total_cost / len(y_true)

        return {
            "total_cost": float(total_cost),
            "avg_cost_per_prediction": float(avg_cost_per_prediction),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
        }

    def get_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable performance summary"""
        summary = f"""
Model Performance Summary: {metrics['model_name']}
{'='*60}
Samples Evaluated: {metrics['n_samples']}
Timestamp: {metrics['timestamp']}

Classification Metrics:
  - Accuracy:   {metrics['accuracy']:.4f}
  - Precision:  {metrics['precision']:.4f}
  - Recall:     {metrics['recall']:.4f}
  - F1-Score:   {metrics['f1_score']:.4f}
  - Sensitivity: {metrics['sensitivity']:.4f}
  - Specificity: {metrics['specificity']:.4f}
"""

        if "roc_auc" in metrics:
            summary += f"  - ROC-AUC:    {metrics['roc_auc']:.4f}\n"
        if "brier_score" in metrics:
            summary += (
                f"  - Brier Score: {metrics['brier_score']:.4f} (lower is better)\n"
            )

        summary += f"""
Confusion Matrix:
  - True Negatives:  {metrics['confusion_matrix']['tn']}
  - False Positives: {metrics['confusion_matrix']['fp']}
  - False Negatives: {metrics['confusion_matrix']['fn']}
  - True Positives:  {metrics['confusion_matrix']['tp']}
"""

        return summary
