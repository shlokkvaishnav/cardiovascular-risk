"""
Real SHAP-based model explainability.

Replaces the old coefficient-multiplication proxy (which only worked for
linear models and silently returned nothing for RandomForest/SVM) with a
proper SHAP explainer selected by model type:
  - TreeExplainer   for tree ensembles (RandomForestClassifier)
  - LinearExplainer for linear models (LogisticRegression)
  - KernelExplainer as a model-agnostic fallback (e.g. SVC), using a small
    k-means-summarized background for tractable latency.

Contributions are signed (positive = increases predicted risk, negative =
protective), unlike the old implementation which only reported magnitude.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

        if hasattr(pipeline, "named_steps"):
            self.preprocessor = pipeline.named_steps.get("preprocessor")
            self.core_model = pipeline.named_steps.get("model", pipeline)

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
        if hasattr(self.core_model, "estimators_") and hasattr(
            self.core_model, "predict_proba"
        ):
            # Tree ensemble, e.g. RandomForestClassifier
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
