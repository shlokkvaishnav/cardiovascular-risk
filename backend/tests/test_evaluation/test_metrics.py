import numpy as np
import pytest

from src.evaluation.metrics import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator()


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = y_true.copy()
    y_pred_proba = np.where(y_true == 1, 0.9, 0.1)
    return y_true, y_pred, y_pred_proba


def test_evaluate_model_perfect_predictions(evaluator, perfect_predictions):
    y_true, y_pred, y_pred_proba = perfect_predictions
    metrics = evaluator.evaluate_model(
        y_true, y_pred, y_pred_proba, model_name="perfect"
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_evaluate_model_with_errors(evaluator):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])  # one FP, one FN

    metrics = evaluator.evaluate_model(y_true, y_pred, model_name="imperfect")

    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert "roc_auc" not in metrics  # no y_pred_proba supplied


def test_evaluate_model_records_history(evaluator, perfect_predictions):
    y_true, y_pred, y_pred_proba = perfect_predictions
    evaluator.evaluate_model(y_true, y_pred, y_pred_proba, model_name="a")
    evaluator.evaluate_model(y_true, y_pred, y_pred_proba, model_name="b")
    assert len(evaluator.metrics_history) == 2


def test_compare_models_picks_best_by_metric(evaluator):
    metrics_list = [
        {
            "model_name": "low",
            "accuracy": 0.6,
            "precision": 0.5,
            "recall": 0.5,
            "f1_score": 0.5,
            "roc_auc": 0.5,
        },
        {
            "model_name": "high",
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.8,
            "f1_score": 0.8,
            "roc_auc": 0.85,
        },
    ]
    comparison = evaluator.compare_models(metrics_list)
    assert comparison["best_by_metric"]["accuracy"]["model"] == "high"
    assert comparison["best_by_metric"]["roc_auc"]["model"] == "high"


def test_compare_models_empty_list_returns_empty_dict(evaluator):
    assert evaluator.compare_models([]) == {}


def test_calculate_business_metrics_weighs_false_negatives_more(evaluator):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])  # 2 FN, 2 FP

    result = evaluator.calculate_business_metrics(
        y_true, y_pred, cost_fp=1.0, cost_fn=10.0
    )

    assert result["false_negatives"] == 2
    assert result["false_positives"] == 2
    assert result["total_cost"] == pytest.approx(2 * 1.0 + 2 * 10.0)
    assert result["avg_cost_per_prediction"] == pytest.approx(
        result["total_cost"] / len(y_true)
    )


def test_save_and_load_metrics_roundtrip(evaluator, tmp_path, perfect_predictions):
    y_true, y_pred, y_pred_proba = perfect_predictions
    metrics = evaluator.evaluate_model(
        y_true, y_pred, y_pred_proba, model_name="roundtrip"
    )

    path = tmp_path / "metrics.json"
    evaluator.save_metrics(metrics, path)
    assert path.exists()

    loaded = evaluator.load_metrics(path)
    assert loaded["accuracy"] == metrics["accuracy"]
    assert loaded["model_name"] == "roundtrip"


def test_evaluate_model_includes_brier_score_and_calibration_curve(
    evaluator, perfect_predictions
):
    y_true, y_pred, y_pred_proba = perfect_predictions
    metrics = evaluator.evaluate_model(
        y_true, y_pred, y_pred_proba, model_name="calibrated"
    )

    assert "brier_score" in metrics
    assert metrics["brier_score"] < 0.1  # near-perfect predictions -> low Brier score
    assert "calibration_curve" in metrics
    assert "prob_true" in metrics["calibration_curve"]
    assert "prob_pred" in metrics["calibration_curve"]


def test_evaluate_model_omits_calibration_when_no_proba(evaluator):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    metrics = evaluator.evaluate_model(y_true, y_pred, model_name="no_proba")
    assert "brier_score" not in metrics
    assert "calibration_curve" not in metrics


def test_assess_calibration_perfect_calibration_zero_brier(evaluator):
    y_true = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.0, 1.0, 0.0, 1.0])
    result = evaluator.assess_calibration(y_true, y_pred_proba, n_bins=2)
    assert result["brier_score"] == pytest.approx(0.0)


def test_evaluate_by_subgroup_reports_per_group_metrics(evaluator):
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7])
    sex = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    result = evaluator.evaluate_by_subgroup(y_true, y_pred, y_pred_proba, sex, "sex")

    assert result["subgroup_name"] == "sex"
    assert set(result["groups"].keys()) == {"0", "1"}
    for group in result["groups"].values():
        assert group["n"] == 4
        assert "accuracy" in group
        assert "roc_auc" in group
        assert "brier_score" in group


def test_evaluate_by_subgroup_handles_single_class_group(evaluator):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.4])
    group = np.array(["a", "a", "b", "b"])

    result = evaluator.evaluate_by_subgroup(y_true, y_pred, y_pred_proba, group, "test")
    # Should not raise even though each subgroup is small; roc_auc/brier may
    # be present or absent depending on class diversity within the group.
    assert result["groups"]["a"]["n"] == 2
    assert result["groups"]["b"]["n"] == 2


def test_get_performance_summary_contains_key_metrics(evaluator, perfect_predictions):
    y_true, y_pred, y_pred_proba = perfect_predictions
    metrics = evaluator.evaluate_model(
        y_true, y_pred, y_pred_proba, model_name="summary"
    )
    summary = evaluator.get_performance_summary(metrics)

    assert "summary" in metrics["model_name"] or "summary" in summary
    assert "Accuracy" in summary
    assert "ROC-AUC" in summary
