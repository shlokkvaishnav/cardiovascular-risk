"""
PDF report generation for saved cardiovascular risk assessments.
Layout style loosely follows the project's earlier prototype (clinical
summary + factor table + explanation section), rebuilt for the current
11-feature lifestyle schema.
"""
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "height": "Height (cm)",
    "weight": "Weight (kg)",
    "ap_hi": "Systolic BP (mmHg)",
    "ap_lo": "Diastolic BP (mmHg)",
    "cholesterol": "Cholesterol category",
    "gluc": "Glucose category",
    "smoke": "Smoker",
    "alco": "Alcohol intake",
    "active": "Physically active",
}


def _humanize_contributor(raw_key: str) -> str:
    stripped = raw_key.replace("num__", "").replace("cat__", "")
    if stripped in FEATURE_LABELS:
        return FEATURE_LABELS[stripped]
    base = stripped.rsplit("_", 1)[0]
    return FEATURE_LABELS.get(base, stripped)


def build_report_pdf(report) -> bytes:
    """report: a db.models.Report instance."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#922B21"))
    story = []

    story.append(Paragraph("CardioRisk Assessment Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {report.created_at.strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#d1d5db")))
    story.append(Spacer(1, 12))

    risk_color_name = {"Low": "green", "Medium": "orange", "High": "red"}.get(report.risk_level, "black")
    story.append(Paragraph(
        f"<font color='{risk_color_name}'><b>{report.risk_level.upper()} RISK</b></font> "
        f"-- predicted probability {report.probability * 100:.1f}%",
        styles["Heading2"],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Submitted Inputs", styles["Heading3"]))
    input_rows = [["Field", "Value"]] + [
        [FEATURE_LABELS.get(k, k), str(v)] for k, v in report.inputs.items()
    ]
    input_table = Table(input_rows, colWidths=[8 * cm, 6 * cm])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 12))

    if report.top_contributors:
        story.append(Paragraph("Top Contributing Factors (SHAP)", styles["Heading3"]))
        contributor_rows = [["Factor", "Impact"]] + [
            [_humanize_contributor(k), f"{v:+.3f}"]
            for entry in report.top_contributors
            for k, v in entry.items()
        ]
        contributor_table = Table(contributor_rows, colWidths=[8 * cm, 6 * cm])
        contributor_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(contributor_table)
        story.append(Spacer(1, 12))

    if report.note:
        story.append(Paragraph("Note", styles["Heading3"]))
        story.append(Paragraph(report.note, styles["Normal"]))
        story.append(Spacer(1, 12))

    story.append(HRFlowable(width="100%", color=colors.HexColor("#d1d5db")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This report is educational decision-support and does not replace professional "
        "medical diagnosis or treatment.",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    return buf.getvalue()
