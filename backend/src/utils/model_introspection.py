"""Helpers for introspecting a fitted model that may be wrapped in a
sklearn.calibration.CalibratedClassifierCV.

Calibration (see ModelTrainer's use of CalibratedClassifierCV) changes the
saved best_model.pkl from a bare sklearn Pipeline to a CalibratedClassifierCV
wrapping one. A CalibratedClassifierCV has neither `.named_steps` nor
`.feature_importances_`/`.coef_` at its own top level -- those live on
`calibrated_classifiers_[i].estimator`, the fitted clone of the base
estimator/pipeline it wraps. Anything that inspects a fitted model's internal
structure (feature-importance extraction in scripts/train.py, SHAP explainer
construction in evaluation/explainer.py) must unwrap through this layer
first, or it silently falls through to whatever fallback it has (e.g.
explainer.py's slow KernelExplainer path) instead of failing loudly.
"""

from typing import Any


def unwrap_calibrated(model: Any) -> Any:
    """Return the underlying fitted estimator/pipeline if `model` is a
    CalibratedClassifierCV, otherwise return `model` unchanged."""
    if hasattr(model, "calibrated_classifiers_"):
        calibrated = model.calibrated_classifiers_[0]
        inner = getattr(calibrated, "estimator", None)
        if inner is not None:
            return inner
    return model
