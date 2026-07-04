"""
Optional report history: save, list, and export cardiovascular risk
assessments for a logged-in user. Guest/anonymous usage never touches this
router -- saving a report is an explicit, authenticated action.

The prediction is always recomputed server-side from the submitted inputs
(reusing the same model/explainer the live /predict endpoint uses) rather
than trusting a client-submitted probability/SHAP payload, so a report's
stored result can't be spoofed by a malicious or buggy client.
"""
import io
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import Report, User
from .auth import get_current_user
from .schemas import PredictionRequest

router = APIRouter(prefix="/reports", tags=["reports"])


class SaveReportRequest(BaseModel):
    inputs: PredictionRequest
    note: Optional[str] = None


class ReportSummary(BaseModel):
    id: str
    risk_level: str
    probability: float
    created_at: datetime
    note: Optional[str] = None


class ReportDetail(ReportSummary):
    inputs: dict
    prediction: int
    top_contributors: Optional[List[dict]] = None
    baseline_probability: Optional[float] = None


def _run_prediction(inputs: PredictionRequest):
    """Deferred import avoids a circular import at module load time: app.py
    imports this router, so `model`/`explainer` can only be resolved once
    app.py has finished initializing, i.e. at request time, not import time."""
    from .app import FEATURE_NAMES, _to_feature_vector, _model_predict, _risk_level, model, explainer  # noqa: F401

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = _to_feature_vector(inputs)
    predictions, probabilities, _ = _model_predict(model, features)
    probability = float(probabilities[0])
    prediction = int(predictions[0])

    top_contributors, baseline_probability = (None, None)
    if explainer is not None:
        top_contributors, baseline_probability = explainer.explain(features)

    return prediction, probability, _risk_level(probability), top_contributors, baseline_probability


@router.post("", response_model=ReportDetail, status_code=status.HTTP_201_CREATED)
def save_report(
    req: SaveReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prediction, probability, risk_level, top_contributors, baseline_probability = _run_prediction(req.inputs)

    report = Report(
        user_id=current_user.id,
        inputs=req.inputs.model_dump(),
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        top_contributors=top_contributors,
        baseline_probability=baseline_probability,
        note=req.note,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return ReportDetail(
        id=report.id,
        risk_level=report.risk_level,
        probability=report.probability,
        created_at=report.created_at,
        note=report.note,
        inputs=report.inputs,
        prediction=report.prediction,
        top_contributors=report.top_contributors,
        baseline_probability=report.baseline_probability,
    )


@router.get("", response_model=List[ReportSummary])
def list_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
):
    reports = (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .order_by(Report.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    return [
        ReportSummary(
            id=r.id, risk_level=r.risk_level, probability=r.probability, created_at=r.created_at, note=r.note
        )
        for r in reports
    ]


def _get_owned_report(report_id: str, current_user: User, db: Session) -> Report:
    report = db.get(Report, report_id)
    if report is None or report.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@router.get("/{report_id}", response_model=ReportDetail)
def get_report(report_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    r = _get_owned_report(report_id, current_user, db)
    return ReportDetail(
        id=r.id, risk_level=r.risk_level, probability=r.probability, created_at=r.created_at, note=r.note,
        inputs=r.inputs, prediction=r.prediction, top_contributors=r.top_contributors,
        baseline_probability=r.baseline_probability,
    )


@router.get("/{report_id}/pdf")
def get_report_pdf(report_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from .pdf_report import build_report_pdf  # deferred: reportlab is only needed here

    r = _get_owned_report(report_id, current_user, db)
    pdf_bytes = build_report_pdf(r)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=cardio-report-{r.id}.pdf"},
    )
