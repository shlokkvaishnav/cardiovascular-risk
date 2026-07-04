"""Diagnostic SHAP feature-importance report.

Loads the trained best_model.pkl + shap_background.pkl, computes SHAP values
over a sample of the test set, and ranks features by mean |SHAP value|.
Informational only -- this does NOT prune features automatically. Automatic
pruning would need its own held-out validation loop to avoid overfitting the
selection itself; that's a deliberate follow-up, not done here. Use this
report to manually spot obviously-dead features worth investigating.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import StackingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.feature_engineering import FeatureEngineer
from src.evaluation.explainer import SHAPExplainer, WeightedContributionExplainer
from src.utils.logger import setup_logger
from src.utils.model_introspection import unwrap_calibrated


def main(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config, log_file=str(PROJECT_ROOT / "logs" / "feature_importance_report.log"))

    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    data_loader = DataLoader(config)
    df = data_loader.load_data(raw_path)
    DataValidator(config)  # unused here, just mirrors train.py's pipeline shape

    feature_engineer = FeatureEngineer(config)
    _, X_test, _, y_test = feature_engineer.prepare_features(df)

    model_path = PROJECT_ROOT / "models" / "artifacts" / "best_model.pkl"
    background_path = PROJECT_ROOT / "models" / "artifacts" / "shap_background.pkl"
    model = joblib.load(model_path)
    background = joblib.load(background_path)

    feature_names = list(X_test.columns)
    unwrapped = unwrap_calibrated(model)
    is_stacking = isinstance(unwrapped, StackingClassifier)

    # WeightedContributionExplainer runs 4 separate per-base-model explainers
    # per row (several seconds each), so cap the sample size much lower than
    # the single-tree-model case to keep this report's runtime reasonable.
    sample_size = min(args.sample_size, 30) if is_stacking else args.sample_size
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=config["project"]["random_seed"])

    if is_stacking:
        explainer = WeightedContributionExplainer(unwrapped, background[feature_names], feature_names)
        totals: dict = {}
        counts: dict = {}
        for i in range(len(sample)):
            contributions, _ = explainer.explain(sample.iloc[[i]], top_k=len(feature_names))
            if contributions is None:
                continue
            for entry in contributions:
                for name, value in entry.items():
                    totals[name] = totals.get(name, 0.0) + abs(value)
                    counts[name] = counts.get(name, 0) + 1
        ranking = sorted(
            ((name, totals[name] / counts[name]) for name in totals),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        explainer = SHAPExplainer(model, background[feature_names], feature_names)
        transformed = explainer._transform(sample)

        if getattr(explainer, "_is_kernel", False):
            shap_values = explainer.explainer.shap_values(transformed, nsamples=128, silent=True)
        else:
            shap_values = explainer.explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[1 if len(shap_values) > 1 else 0])
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                arr = arr[:, :, -1]

        names = explainer.transformed_feature_names
        if len(names) != arr.shape[1]:
            names = [f"feature_{i}" for i in range(arr.shape[1])]

        mean_abs_shap = np.abs(arr).mean(axis=0)
        ranking = sorted(
            zip(names, mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True
        )

    report = {
        "sample_size": int(len(sample)),
        "ranking": [{"feature": name, "mean_abs_shap": value} for name, value in ranking],
    }

    out_path = PROJECT_ROOT / "logs" / "evaluation" / "shap_feature_ranking.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"SHAP feature ranking saved to {out_path}")
    print("\nTop features by mean |SHAP value|:")
    for name, value in ranking[:15]:
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a SHAP feature-importance diagnostic report")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()
    main(args)
