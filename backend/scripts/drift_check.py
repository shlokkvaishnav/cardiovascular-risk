"""
Lightweight data-drift check: compares a "recent" data sample against the
original training data's summary statistics, reusing DataValidator's
statistical profiling (get_data_quality_report) rather than adding a new
dependency (e.g. evidently.ai) for what is, at this scale, a straightforward
mean/std shift comparison.

Usage:
    python scripts/drift_check.py --recent-data path/to/recent_sample.csv

If --recent-data is omitted, this runs against the training data's own test
split as a smoke test (should report no drift, since it's the same
distribution the model was trained on).
"""
import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.utils.logger import setup_logger

# A mean shift beyond this many training-set standard deviations is flagged.
# Deliberately conservative (drift checks are for humans to review, not to
# auto-block anything) -- tune based on observed false-positive rate.
DRIFT_THRESHOLD_STD = 0.5


def compare_distributions(baseline_stats: dict, recent_stats: dict, threshold: float = DRIFT_THRESHOLD_STD):
    """Compare per-column summary statistics, flagging columns whose mean has
    shifted by more than `threshold` baseline standard deviations."""
    drifted = []
    for column, baseline in baseline_stats.items():
        if column not in recent_stats:
            continue
        recent = recent_stats[column]
        std = baseline["std"] or 1.0  # avoid division by zero for constant columns
        shift = abs(recent["mean"] - baseline["mean"]) / std
        if shift > threshold:
            drifted.append({
                "column": column,
                "baseline_mean": baseline["mean"],
                "recent_mean": recent["mean"],
                "shift_in_std": round(shift, 2),
            })
    return drifted


def main(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config, log_file=str(PROJECT_ROOT / "logs" / "drift_check.log"))

    data_loader = DataLoader(config)
    validator = DataValidator(config)

    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    baseline_df = data_loader.load_data(raw_path)
    baseline_report = validator.get_data_quality_report(baseline_df)

    if args.recent_data:
        recent_path = Path(args.recent_data)
        recent_df = data_loader.load_data(recent_path)
    else:
        logger.info("No --recent-data supplied; using the training data's own held-out "
                    "split as a smoke test (expect no drift reported).")
        from sklearn.model_selection import train_test_split
        _, recent_df = train_test_split(
            baseline_df, test_size=0.3, random_state=config["project"]["random_seed"]
        )
    recent_report = validator.get_data_quality_report(recent_df)

    drifted = compare_distributions(
        baseline_report["summary_statistics"], recent_report["summary_statistics"], args.threshold
    )

    if not drifted:
        logger.info("No significant drift detected.")
        print("No significant drift detected.")
        return

    logger.warning(f"Drift detected in {len(drifted)} column(s):")
    print(f"\nDrift detected in {len(drifted)} column(s):")
    for d in drifted:
        line = (
            f"  {d['column']}: baseline mean={d['baseline_mean']:.2f}, "
            f"recent mean={d['recent_mean']:.2f}, shift={d['shift_in_std']} std"
        )
        logger.warning(line)
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for data drift against the training distribution")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--recent-data", default=None, help="Path to a CSV of recent production inputs")
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD_STD,
                         help="Mean shift threshold, in baseline standard deviations")
    args = parser.parse_args()
    main(args)
