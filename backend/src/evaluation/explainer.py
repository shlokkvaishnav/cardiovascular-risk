"""
Real SHAP-based model explainability.

Replaces the old coefficient-multiplication proxy (which only worked for
linear models and silently returned nothing for RandomForest/SVM) with a
proper SHAP explainer selected by model type:
  - TreeExplainer   for tree ensembles (RandomForestClassifier, LightGBM,
                     XGBoost)
  - LinearExplainer for linear models (LogisticRegression)
  - KernelExplainer as a model-agnostic fallback (e.g. SVC), using a small
    k-means-summarized background for tractable latency.

Contributions are signed (positive = increases predicted risk, negative =
protective), unlike the old implementation which only reported magnitude.

A model selected via ModelTrainer's ROC-AUC-driven selection may be wrapped
in a sklearn.calibration.CalibratedClassifierCV (see training/trainer.py).
That wrapper exposes neither `.named_steps`, `.estimators_`, nor `.coef_` at
its own top level, so it's unwrapped to the underlying fitted
Pipeline/estimator first -- otherwise every calibrated model would silently
fall through to the slow KernelExplainer fallback below.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.model_introspection import unwrap_calibrated

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Wraps a fitted sklearn Pipeline (preprocessor + model) with a SHAP explainer."""

    def __init__(self, pipeline: Any, background: Any, raw_feature_names: List[str]):
        """background: a DataFrame (or ndarray) of raw, untransformed rows in
        raw_feature_names order. Must be a DataFrame if the pipeline's
        ColumnTransformer selects columns by name (the common case here)."""
        import shap

        self.raw_feature_names = list(raw_feature_names)
        self.preprocessor = None
        self.core_model = pipeline
        self._is_kernel = False

        pipeline = unwrap_calibrated(pipeline)

        if hasattr(pipeline, "named_steps"):
            self.preprocessor = pipeline.named_steps.get("preprocessor")
            self.core_model = pipeline.named_steps.get("model", pipeline)
        else:
            self.core_model = pipeline

        self.transformed_feature_names = self._resolve_transformed_feature_names()

        background_transformed = self._transform(background)
        self.explainer = self._build_explainer(shap, background_transformed)

    def _resolve_transformed_feature_names(self) -> List[str]:
        if self.preprocessor is not None and hasattr(
            self.preprocessor, "get_feature_names_out"
        ):
            try:
                return list(self.preprocessor.get_feature_names_out())
            except Exception:  # pragma: no cover - defensive
                pass
        return list(self.raw_feature_names)

    def _transform(self, features: Any) -> np.ndarray:
        if self.preprocessor is not None:
            return self.preprocessor.transform(features)
        return features

    def _build_explainer(
        self, shap_module: Any, background_transformed: np.ndarray
    ) -> Any:
        from sklearn.ensemble import StackingClassifier

        # StackingClassifier also has `.estimators_` (its list of fitted base
        # estimators), which would otherwise wrongly match the tree-ensemble
        # branch below. Stacked models are handled by WeightedContributionExplainer
        # instead (see app.py's explainer dispatch) -- this check is just a
        # defensive guard in case a StackingClassifier ever reaches this class
        # directly.
        if isinstance(self.core_model, StackingClassifier):
            raise TypeError(
                "SHAPExplainer does not support StackingClassifier directly; "
                "use WeightedContributionExplainer instead."
            )

        is_tree_ensemble = hasattr(self.core_model, "estimators_") and hasattr(
            self.core_model, "predict_proba"
        )
        # LightGBM/XGBoost don't use sklearn's `.estimators_` naming, so they
        # need their own detection: LGBMClassifier exposes `.booster_`,
        # XGBClassifier exposes `.get_booster()`. Both are fully supported by
        # shap.TreeExplainer (fast, exact), same tier as RandomForest.
        is_lightgbm_or_xgboost = hasattr(self.core_model, "booster_") or hasattr(
            self.core_model, "get_booster"
        )
        if (is_tree_ensemble or is_lightgbm_or_xgboost) and hasattr(
            self.core_model, "predict_proba"
        ):
            return shap_module.TreeExplainer(self.core_model)

        if hasattr(self.core_model, "coef_"):
            # Linear model, e.g. LogisticRegression
            return shap_module.LinearExplainer(self.core_model, background_transformed)

        # Model-agnostic fallback, e.g. SVC. KernelExplainer is expensive per
        # call, so we summarize the background with k-means to keep it tractable.
        # (The trainer excludes SVM from served models for exactly this reason;
        # this path only runs if some future non-tree/non-linear model is served.)
        self._is_kernel = True
        n_background = background_transformed.shape[0]
        k = min(10, n_background)
        summary = (
            shap_module.kmeans(background_transformed, k)
            if k > 1
            else background_transformed
        )
        predict_fn = getattr(self.core_model, "predict_proba", self.core_model.predict)
        return shap_module.KernelExplainer(predict_fn, summary)

    def _baseline_probability(self) -> Optional[float]:
        expected_value = getattr(self.explainer, "expected_value", None)
        if expected_value is None:
            return None
        try:
            arr = np.asarray(expected_value, dtype=float).reshape(-1)
            return float(arr[-1])
        except Exception:  # pragma: no cover - defensive
            return None

    def explain(
        self, raw_features: Any, top_k: int = 5
    ) -> Tuple[Optional[List[Dict[str, float]]], Optional[float]]:
        """Return (signed top-k SHAP contributions, baseline probability) for one row of raw features.
        raw_features must be a DataFrame if the pipeline's preprocessor selects columns by name.

        Never raises: any SHAP failure logs a warning and returns (None, None)
        so a bad explanation can never take down a prediction request.
        """
        try:
            transformed = self._transform(raw_features)
            # Cap KernelExplainer sampling so the model-agnostic fallback stays
            # within interactive latency (its default 'auto' can take tens of
            # seconds). Tree/Linear explainers ignore this kwarg.
            if getattr(self, "_is_kernel", False):
                shap_values = self.explainer.shap_values(
                    transformed, nsamples=128, silent=True
                )
            else:
                shap_values = self.explainer.shap_values(transformed)
            row = self._positive_class_row(shap_values)

            names = self.transformed_feature_names
            if len(names) != row.shape[0]:
                names = [f"feature_{i}" for i in range(row.shape[0])]

            order = np.argsort(np.abs(row))[::-1][:top_k]
            contributions = [{names[i]: float(row[i])} for i in order]
            return contributions, self._baseline_probability()
        except Exception as exc:
            logger.warning(
                "SHAP explanation failed, omitting top_contributors: %s", exc
            )
            return None, None

    @staticmethod
    def _positive_class_row(shap_values: Any) -> np.ndarray:
        """Normalize SHAP output across explainer types to a single 1D array of
        signed contributions toward the positive (disease) class, for one sample."""
        if isinstance(shap_values, list):
            # One array per class (common for Tree/KernelExplainer on binary clf)
            idx = 1 if len(shap_values) > 1 else 0
            arr = np.asarray(shap_values[idx])
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                # (n_samples, n_features, n_classes)
                arr = arr[:, :, -1]
        return arr[0]


class WeightedContributionExplainer:
    """Approximate SHAP-style explainer for a fitted sklearn StackingClassifier.

    An exact SHAP explanation for the full stack would require
    shap.KernelExplainer on the meta-learner -- the same 30-60s/prediction cost
    that already ruled out SVM as a served model. Instead, this computes each
    base estimator's own fast/exact TreeExplainer or LinearExplainer
    contributions independently (via a per-base-estimator SHAPExplainer), then
    combines them weighted by the meta-learner's fitted coefficient for that
    base estimator's positive-class column.

    This is NOT a Shapley-exact value for the full stack -- it ignores any
    nonlinearity the meta-learner could in principle capture (though with a
    LogisticRegression meta-learner, as used here, there is none to ignore).
    It stays fast and is directionally correct. This approximation is
    disclosed explicitly in MODEL_CARD.md rather than presented as exact.
    """

    def __init__(self, stacking_model: Any, background: Any, raw_feature_names: List[str]):
        self.raw_feature_names = list(raw_feature_names)
        self._is_kernel = False

        base_names = [name for name, _ in stacking_model.estimators]
        self.base_explainers: Dict[str, SHAPExplainer] = {}
        for name, fitted_pipeline in zip(base_names, stacking_model.estimators_):
            try:
                self.base_explainers[name] = SHAPExplainer(
                    fitted_pipeline, background, raw_feature_names
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Could not build sub-explainer for '{name}': {exc}")

        self.weights = self._compute_weights(stacking_model, base_names)

    @staticmethod
    def _compute_weights(stacking_model: Any, base_names: List[str]) -> Dict[str, float]:
        """Extract each base estimator's weight from the meta-learner's fitted
        coefficients. With stack_method="predict_proba" (the default when
        every base estimator supports it, as here) and passthrough=False, the
        meta-learner's input is n_estimators * n_classes columns, ordered per
        estimator; we take the positive-class (last) column of each block."""
        final_estimator = stacking_model.final_estimator_
        n_estimators = len(base_names)
        try:
            coef = np.asarray(final_estimator.coef_, dtype=float).reshape(-1)
            n_classes = len(coef) // n_estimators if n_estimators else 1
            weights = {}
            for i, name in enumerate(base_names):
                idx = i * n_classes + (n_classes - 1)
                weights[name] = float(coef[idx]) if idx < len(coef) else 1.0
            return weights
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"Could not extract meta-learner weights, falling back to "
                f"equal weighting: {exc}"
            )
            return {name: 1.0 for name in base_names}

    def explain(
        self, raw_features: Any, top_k: int = 5
    ) -> Tuple[Optional[List[Dict[str, float]]], Optional[float]]:
        """Never raises: any failure logs a warning and returns (None, None),
        matching SHAPExplainer's contract."""
        try:
            combined: Dict[str, float] = {}
            weighted_baseline = 0.0
            baseline_weight_total = 0.0

            for name, sub_explainer in self.base_explainers.items():
                weight = self.weights.get(name, 0.0)
                contributions, baseline = sub_explainer.explain(
                    raw_features, top_k=len(sub_explainer.transformed_feature_names)
                )
                if contributions is None:
                    continue
                for entry in contributions:
                    for feature_name, value in entry.items():
                        combined[feature_name] = combined.get(feature_name, 0.0) + weight * value
                if baseline is not None:
                    weighted_baseline += weight * baseline
                    baseline_weight_total += weight

            if not combined:
                return None, None

            ordered = sorted(combined.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
            result = [{name: value} for name, value in ordered]
            baseline_probability = (
                weighted_baseline / baseline_weight_total
                if baseline_weight_total
                else None
            )
            return result, baseline_probability
        except Exception as exc:
            logger.warning(
                "Weighted-contribution explanation failed, omitting top_contributors: %s",
                exc,
            )
            return None, None
