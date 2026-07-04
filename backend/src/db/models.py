"""
SQLAlchemy models for the optional accounts/report-history feature.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    reports: Mapped[list["Report"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Report(Base):
    """A saved cardiovascular risk assessment: the submitted inputs plus the
    model's prediction and explanation at the time it was generated. Reports
    are only ever created via an explicit, logged-in "Save this result"
    action -- guest/anonymous assessments never reach this table."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)

    # Submitted clinical inputs (backend PredictionRequest fields)
    inputs: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Model output at the time of saving
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    top_contributors: Mapped[list] = mapped_column(JSON, nullable=True)
    baseline_probability: Mapped[float] = mapped_column(Float, nullable=True)

    note: Mapped[str] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    user: Mapped["User"] = relationship(back_populates="reports")
