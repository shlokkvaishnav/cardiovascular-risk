"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, GaugeCircle, ShieldCheck, TriangleAlert, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { RiskTimeline } from "./results/RiskTimeline";
import { computeHeartAge } from "../lib/heartAge";
import { useAssessmentHistory } from "../hooks/useAssessmentHistory";
import { getRiskCategory, type RiskResult, scoreRisk, type AssessmentData } from "../lib/risk";
import { createShareableResultsUrl, decodeSharedResult } from "../lib/shareUrl";
import { RiskSimulator } from "./simulator/RiskSimulator";
import { HeartAge } from "../results/HeartAge";
import { runRiskCalc } from "../lib/riskEngine";

const STORAGE_KEY = "cardio_assessment_v1";

type StoredPayload = {
  data: AssessmentData;
  result: RiskResult;
};

function statusClass(score: number) {
  const category = getRiskCategory(score);
  if (category === "low") return "chip chip-success";
  if (category === "moderate") return "chip chip-warning";
  return "chip chip-danger";
}

function esc(text: string) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function buildDoctorQuestions(result: RiskResult): string[] {
  const labels = result.drivers.map((driver) => driver.label.toLowerCase());
  const questions = [
    "What is the most important risk factor for me to target first?",
    "Should I get updated lipid and glucose labs in the next 4-12 weeks?",
  ];

  if (labels.some((label) => label.includes("blood pressure"))) {
    questions.push("Do I need home blood pressure monitoring or medication adjustment?");
  }
  if (labels.some((label) => label.includes("cholesterol"))) {
    questions.push("Would statin therapy or other lipid-lowering treatment be appropriate?");
  }
  if (labels.some((label) => label.includes("smoking"))) {
    questions.push("What smoking cessation plan is most realistic for me?");
  }
  if (labels.some((label) => label.includes("diabetes"))) {
    questions.push("How should diabetes control be optimized to lower cardiovascular risk?");
  }

  return questions.slice(0, 5);
}

function contributorInsight(label: string, data: AssessmentData | null): { meaning: string; value: string; action: string } {
  const key = label.toLowerCase();
  if (key.includes("blood pressure")) {
    return {
      meaning: "Higher systolic blood pressure increases arterial strain and event probability.",
      value: data?.systolicBp ? `${data.systolicBp} mmHg (target: <120)` : "Not provided",
      action: "Track home BP readings and discuss sodium reduction + treatment optimization.",
    };
  }
  if (key.includes("cholesterol")) {
    return {
      meaning: "A higher total/HDL cholesterol ratio is associated with higher atherosclerotic risk.",
      value:
        data?.totalCholesterol && data?.hdlCholesterol
          ? `TC ${data.totalCholesterol} / HDL ${data.hdlCholesterol} mg/dL`
          : "Not provided",
      action: "Review lipid panel with your clinician and target sustained diet/activity changes.",
    };
  }
  if (key.includes("smoking")) {
    return {
      meaning: "Smoking accelerates vascular inflammation and thrombosis risk.",
      value: data?.smoking ? "Current smoker" : "Not currently smoking",
      action: "Set a quit date and use structured cessation support.",
    };
  }
  if (key.includes("diabetes")) {
    return {
      meaning: "Diabetes elevates long-term cardiovascular event probability.",
      value: data?.diabetes ? "Diabetes history: Yes" : "Diabetes history: No/unknown",
      action: "Prioritize glucose management and regular follow-up testing.",
    };
  }
  return {
    meaning: "This factor contributes to your composite risk score.",
    value: "Context based on submitted profile",
    action: "Address this with your clinician as part of your prevention plan.",
  };
}

type EscalationContext = {
  show: boolean;
  urgent: boolean;
  reasons: string[];
};

function getEscalationContext(result: RiskResult, data: AssessmentData | null, dismissed: boolean): EscalationContext {
  if (!result || !data || dismissed) {
    return { show: false, urgent: false, reasons: [] };
  }

  const reasons: string[] = [];
  const highRiskScore = result.score > 30;
  const highSystolic = data.systolicBp !== null && data.systolicBp >= 180;
  const extremeSystolic = data.systolicBp !== null && data.systolicBp > 200;

  if (highRiskScore) reasons.push("10-year risk estimate is above 30%");
  if (highSystolic) reasons.push("Systolic blood pressure is 180 mmHg or higher");

  let highRedFactors = 0;
  if (data.smoking) highRedFactors += 1;
  if (data.diabetes) highRedFactors += 1;
  if (data.familyHistory) highRedFactors += 1;
  if ((data.activityLevel ?? "moderate") === "low") highRedFactors += 1;
  if (data.systolicBp !== null && data.systolicBp >= 160) highRedFactors += 1;

  if (highRedFactors >= 3) {
    reasons.push("Multiple high-risk factors are present simultaneously");
  }

  const show = reasons.length > 0;
  return { show, urgent: extremeSystolic, reasons };
}

export function ResultsDashboard() {
  const [result, setResult] = useState<RiskResult | null>(null);
  const [assessmentData, setAssessmentData] = useState<AssessmentData | null>(null);
  const [simulatedResult, setSimulatedResult] = useState<RiskResult | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "simulate">("overview");
  const [projectionMode, setProjectionMode] = useState<"current" | "best" | "worst">("current");
  const [expandedDriver, setExpandedDriver] = useState<string | null>(null);
  const [animatedScore, setAnimatedScore] = useState(0);
  const [hasData, setHasData] = useState(true);
  const [dismissedEscalation, setDismissedEscalation] = useState(false);
  const [shareMessage, setShareMessage] = useState("");
  const { history, historyCount, clearHistory } = useAssessmentHistory();

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        const params = new URLSearchParams(window.location.search);
        const shared = params.get("data");
        let parsed: StoredPayload | null = null;

        if (shared) {
          parsed = decodeSharedResult(shared);
        }

        if (!parsed) {
          const raw = window.sessionStorage.getItem(STORAGE_KEY);
          if (!raw) {
            setHasData(false);
            return;
          }
          parsed = JSON.parse(raw) as StoredPayload;
        }

        if (parsed?.result) {
          setResult(parsed.result);
          setAssessmentData(parsed.data);
          window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
        } else if (parsed?.data) {
          const computed = scoreRisk(parsed.data);
          setResult(computed);
          setAssessmentData(parsed.data);
          window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ data: parsed.data, result: computed }));
        } else {
          setHasData(false);
        }
      } catch {
        setHasData(false);
      }
    }, 0);

    return () => window.clearTimeout(timer);
  }, []);

  const displayedResult = activeTab === "simulate" && simulatedResult ? simulatedResult : result;

  const projectedResult = useMemo(() => {
    if (!assessmentData || !result) return null;
    if (projectionMode === "current") return result;
    if (projectionMode === "best") {
      return runRiskCalc({
        ...assessmentData,
        systolicBp: 118,
        totalCholesterol: 170,
        hdlCholesterol: 60,
        smoking: false,
        smokingStatus: "never",
        activityLevel: "high",
      });
    }
    return runRiskCalc({
      ...assessmentData,
      systolicBp: 180,
      totalCholesterol: 300,
      hdlCholesterol: 30,
      smoking: true,
      smokingStatus: "current",
      activityLevel: "low",
    });
  }, [assessmentData, projectionMode, result]);

  const activeResult = activeTab === "overview" ? projectedResult : displayedResult;
  const activeScore = activeResult?.score ?? null;

  useEffect(() => {
    if (activeScore === null) return;
    let frame = 0;
    const totalFrames = 90;
    const timer = window.setInterval(() => {
      frame += 1;
      setAnimatedScore((activeScore * frame) / totalFrames);
      if (frame >= totalFrames) {
        window.clearInterval(timer);
      }
    }, 16);
    return () => window.clearInterval(timer);
  }, [activeScore]);

  const escalation = useMemo(
    () =>
      activeResult
        ? getEscalationContext(activeResult, assessmentData, dismissedEscalation)
        : { show: false, urgent: false, reasons: [] },
    [assessmentData, dismissedEscalation, activeResult],
  );

  const chartStyle = useMemo(() => {
    if (!activeResult) return {};
    const degree = Math.max(6, Math.round((animatedScore / 45) * 360));
    return {
      background: `conic-gradient(from -90deg, ${
        activeResult.category === "low"
          ? "#27ae60"
          : activeResult.category === "moderate"
            ? "#f39c12"
            : "#e74c3c"
      } 0deg, ${
        activeResult.category === "low"
          ? "#27ae60"
          : activeResult.category === "moderate"
            ? "#f39c12"
            : "#e74c3c"
      } ${degree}deg, #eef1f4 ${degree}deg 360deg)`,
    };
  }, [activeResult, animatedScore]);

  const heartAge = useMemo(() => {
    if (!activeResult || !assessmentData?.age) return null;
    return computeHeartAge(assessmentData.age, activeResult.score, assessmentData);
  }, [assessmentData, activeResult]);

  const exportClinicalSummaryPdf = () => {
    if (!assessmentData || !activeResult) return;
    const generatedAt = new Date();
    const assessmentDate = generatedAt.toLocaleString();
    const questions = buildDoctorQuestions(activeResult);
    const shareUrl = createShareableResultsUrl({ data: assessmentData, result: activeResult });

    const rows: Array<[string, string]> = [
      ["Age", assessmentData.age ? `${assessmentData.age} years` : "Not provided"],
      ["Sex at birth", assessmentData.sex ?? "Not provided"],
      [
        "Systolic BP",
        assessmentData.unknownVitals.systolicBp
          ? "Unknown"
          : assessmentData.systolicBp
            ? `${assessmentData.systolicBp} mmHg`
            : "Not provided",
      ],
      [
        "Total Cholesterol",
        assessmentData.unknownVitals.totalCholesterol
          ? "Unknown"
          : assessmentData.totalCholesterol
            ? `${assessmentData.totalCholesterol} mg/dL`
            : "Not provided",
      ],
      [
        "HDL Cholesterol",
        assessmentData.unknownVitals.hdlCholesterol
          ? "Unknown"
          : assessmentData.hdlCholesterol
            ? `${assessmentData.hdlCholesterol} mg/dL`
            : "Not provided",
      ],
      ["Smoking", assessmentData.smoking === null ? "Not provided" : assessmentData.smoking ? "Yes" : "No"],
      ["Activity", assessmentData.activityLevel ?? "Not provided"],
      ["Diabetes", assessmentData.diabetes === null ? "Not provided" : assessmentData.diabetes ? "Yes" : "No"],
      [
        "Family History",
        assessmentData.familyHistory === null ? "Not provided" : assessmentData.familyHistory ? "Yes" : "No",
      ],
      [
        "BP Medication",
        assessmentData.onBpMedication === null ? "Not provided" : assessmentData.onBpMedication ? "Yes" : "No",
      ],
    ];

    const popup = window.open("", "_blank", "noopener,noreferrer,width=980,height=900");
    if (!popup) {
      setShareMessage("Popup blocked. Please allow popups to download the clinical summary.");
      window.setTimeout(() => setShareMessage(""), 3500);
      return;
    }

    popup.document.write(`<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CardioRisk Clinical Summary</title>
  <style>
    body { font-family: "Segoe UI", Arial, sans-serif; margin: 28px; color: #1f2937; line-height: 1.45; }
    h1 { margin: 0 0 4px 0; font-size: 22px; color: #922b21; }
    h2 { margin: 18px 0 8px 0; font-size: 16px; color: #111827; }
    .meta { color: #4b5563; font-size: 12px; margin-bottom: 12px; }
    .box { border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; margin-top: 10px; }
    .risk { font-size: 22px; font-weight: 700; color: #922b21; margin-right: 12px; }
    .label { font-weight: 600; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    td { border-bottom: 1px solid #e5e7eb; padding: 7px 4px; font-size: 13px; vertical-align: top; }
    td:first-child { width: 34%; color: #374151; font-weight: 600; }
    ul { margin: 8px 0 0 18px; padding: 0; }
    li { margin: 4px 0; }
    .foot { margin-top: 16px; font-size: 11px; color: #6b7280; }
    .warn { background: #fff7ed; border: 1px solid #fdba74; padding: 10px; border-radius: 8px; font-size: 12px; }
    @media print { body { margin: 18px; } .no-print { display: none; } }
  </style>
</head>
<body>
  <h1>Patient-generated cardiovascular risk assessment</h1>
  <div class="meta">For clinical discussion only | Generated: ${esc(assessmentDate)}</div>
  <div class="box">
    <span class="risk">${esc(activeResult.score.toFixed(1))}%</span>
    <span class="label">${esc(activeResult.category.toUpperCase())} 10-year risk estimate</span>
    ${heartAge !== null && assessmentData.age !== null ? `<div style="margin-top:6px;">Heart Age: <strong>${esc(String(heartAge))}</strong> (Chronological age: ${esc(String(assessmentData.age))})</div>` : ""}
    <div style="margin-top:6px;">Summary: ${esc(activeResult.summary)}</div>
  </div>

  <h2>Submitted Inputs</h2>
  <table>
    ${rows.map((row) => `<tr><td>${esc(row[0])}</td><td>${esc(row[1])}</td></tr>`).join("")}
  </table>

  <h2>Top Contributors</h2>
  <ul>
    ${activeResult.drivers.map((driver) => `<li>${esc(driver.label)} (+${esc((driver.impact * 100).toFixed(1))} pts)</li>`).join("") || "<li>No major upward contributors detected.</li>"}
  </ul>

  <h2>Questions to Ask Your Doctor</h2>
  <ul>
    ${questions.map((question) => `<li>${esc(question)}</li>`).join("")}
  </ul>

  <h2>Action Recommendations</h2>
  <ul>
    ${activeResult.recommendations.map((item) => `<li>${esc(item)}</li>`).join("")}
  </ul>

  <div class="warn">
    This report is educational decision-support and does not replace clinical diagnosis or physician judgment.
  </div>

  <div class="foot">
    Methodology: Calculated using risk-factor logic aligned with Framingham / ACC/AHA Pooled Cohort-style inputs.
    Reference: Goff DC et al., 2014 ACC/AHA Guideline on the Assessment of Cardiovascular Risk.
    ${shareUrl ? `<br/>Shareable result URL (contains health inputs): ${esc(shareUrl)}` : ""}
  </div>
  <button class="no-print" onclick="window.print()" style="margin-top:16px;padding:8px 12px;">Print / Save as PDF</button>
</body>
</html>`);
    popup.document.close();
    popup.focus();
    popup.print();
  };

  if (!hasData) {
    return (
      <section className="section">
        <div className="container">
          <div className="card-panel empty-state">
            <TriangleAlert size={20} />
            <h1 className="type-h2">No assessment session found</h1>
            <p className="lead">Complete the guided assessment first to generate your risk dashboard.</p>
            <Link href="/assess" className="btn btn-primary">
              Start Assessment
            </Link>
          </div>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="section">
        <div className="container">
          <div className="card-panel empty-state">
            <GaugeCircle className="spin" size={22} />
            <p className="lead">Loading dashboard...</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="section">
      <div className="container dashboard-grid">
        {escalation.show && (
          <div className="escalation-overlay" role="dialog" aria-modal="true" aria-label="Urgent risk alert">
            <div className="escalation-modal">
              <button type="button" className="escalation-close" onClick={() => setDismissedEscalation(true)}>
                <X size={16} />
              </button>
              <h2 className="type-h3">Please seek prompt medical attention</h2>
              <p className="lead">
                Your inputs suggest a potentially high immediate cardiovascular risk pattern. A same-day clinical review
                is recommended.
              </p>
              <ul className="escalation-reasons">
                {escalation.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
              <div className="escalation-actions">
                <a
                  className="btn btn-primary"
                  href="https://www.google.com/maps/search/cardiologist+near+me/"
                  target="_blank"
                  rel="noreferrer"
                >
                  Find a Cardiologist Near You
                </a>
                {escalation.urgent && (
                  <a className="btn btn-subtle" href="tel:911">
                    Call Emergency Services
                  </a>
                )}
                <button type="button" className="btn btn-subtle" onClick={() => setDismissedEscalation(true)}>
                  Continue to Dashboard
                </button>
              </div>
            </div>
          </div>
        )}

        <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <div className="dashboard-head">
            <h1 className="type-h2">Your Cardiovascular Risk Dashboard</h1>
            <span className={statusClass(activeResult!.score)}>{activeResult!.category.toUpperCase()} RISK</span>
          </div>
          <p className="lead">{activeResult!.summary}</p>
          <div className="results-tabs">
            <button
              type="button"
              className={`btn btn-subtle ${activeTab === "overview" ? "tab-active" : ""}`}
              onClick={() => setActiveTab("overview")}
            >
              Overview
            </button>
            <button
              type="button"
              className={`btn btn-subtle ${activeTab === "simulate" ? "tab-active" : ""}`}
              onClick={() => setActiveTab("simulate")}
            >
              Simulate
            </button>
          </div>

          {activeTab === "overview" && (
            <div className="results-tabs">
              <button
                type="button"
                className={`btn btn-subtle ${projectionMode === "current" ? "tab-active" : ""}`}
                onClick={() => setProjectionMode("current")}
              >
                Current
              </button>
              <button
                type="button"
                className={`btn btn-subtle ${projectionMode === "best" ? "tab-active" : ""}`}
                onClick={() => setProjectionMode("best")}
              >
                Best Case
              </button>
              <button
                type="button"
                className={`btn btn-subtle ${projectionMode === "worst" ? "tab-active" : ""}`}
                onClick={() => setProjectionMode("worst")}
              >
                Worst Case
              </button>
            </div>
          )}

          {activeTab === "overview" && (
            <div className="gauge-row">
              <div className="gauge-shell" style={{ ...chartStyle, boxShadow: `0 20px 40px rgba(26, 26, 46, 0.12)` }}>
                <div className="gauge-core">
                  <strong className="mono score">{animatedScore.toFixed(1)}%</strong>
                  <span>10-year estimate</span>
                </div>
              </div>
              <div className="zone-legend">
                <div className="zone">
                  <span className="dot low" />
                  <span>Low: 0 - 7.5%</span>
                </div>
                <div className="zone">
                  <span className="dot moderate" />
                  <span>Moderate: 7.5 - 20%</span>
                </div>
                <div className="zone">
                  <span className="dot high" />
                  <span>High: {">"}20%</span>
                </div>
              </div>
            </div>
          )}
          {activeTab === "overview" && (
            <div className="risk-context">
              <strong className="mono">1 in {Math.max(1, Math.round(100 / Math.max(0.1, activeResult!.score)))}</strong>
              <span>chance of a cardiovascular event in the next 10 years.</span>
            </div>
          )}

          {activeTab === "simulate" && assessmentData && (
            <RiskSimulator baselineData={assessmentData} baselineResult={result} onResultChange={setSimulatedResult} />
          )}
          {heartAge !== null && assessmentData?.age !== null ? (
            <div style={{ marginTop: 20 }}>
              <HeartAge heartAge={heartAge} age={assessmentData.age} />
            </div>
          ) : null}
        </motion.div>

        <motion.div
          className="card-panel"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <h2 className="type-h3">Top Risk Drivers</h2>
          <div className="driver-list">
            {activeResult!.drivers.length === 0 ? (
              <p className="lead">No major upward contributors detected from submitted data.</p>
            ) : (
              activeResult!.drivers.map((driver) => {
                const info = contributorInsight(driver.label, assessmentData);
                return (
                  <div key={driver.label}>
                    <button
                      type="button"
                      className="driver-row driver-click"
                      onClick={() => setExpandedDriver((prev) => (prev === driver.label ? null : driver.label))}
                    >
                      <span>{driver.label}</span>
                      <strong className="mono">+{(driver.impact * 100).toFixed(1)} pts</strong>
                      <span className="driver-bar">
                        <i style={{ width: `${Math.min(100, driver.impact * 900)}%` }} />
                      </span>
                    </button>
                    {expandedDriver === driver.label && (
                      <div className="driver-detail">
                        <p><strong>What it means:</strong> {info.meaning}</p>
                        <p><strong>Your value:</strong> {info.value}</p>
                        <p><strong>Next action:</strong> {info.action}</p>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>

          <h2 className="type-h3">Action Plan</h2>
          <div className="action-list">
            {activeResult!.recommendations.map((item) => (
              <div key={item} className="action-item">
                <CheckCircle2 size={16} />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          className="card-panel trust-banner"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div>
            <ShieldCheck size={16} />
            <span>No data stored: assessment stays in your browser session.</span>
          </div>
          <div>
            <AlertTriangle size={16} />
            <span>Decision-support only. Clinical judgment remains essential.</span>
          </div>
          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center" }}>
            <Link href="/assess" className="btn btn-subtle">
              Re-run Assessment
            </Link>
            <button
              type="button"
              className="btn btn-subtle"
              onClick={() => {
                if (!assessmentData) return;
                const shareUrl = createShareableResultsUrl({ data: assessmentData, result: activeResult! });
                if (!shareUrl) return;
                navigator.clipboard.writeText(shareUrl);
                setShareMessage("Shareable link copied. It contains your health inputs.");
                window.setTimeout(() => setShareMessage(""), 3000);
              }}
            >
              Copy Shareable Link
            </button>
            <button type="button" className="btn btn-subtle" onClick={exportClinicalSummaryPdf}>
              Download Clinical Summary
            </button>
            <a className="btn btn-subtle" href="https://www.google.com/maps/search/cardiologist+near+me/" target="_blank" rel="noreferrer">
              Find a Cardiologist
            </a>
            <button
              type="button"
              className="btn btn-subtle"
              onClick={() => {
                const shareText = `I checked my heart health on CardioRisk and got a ${activeResult!.category.toUpperCase()} risk category. Check yours.`;
                const url = `${window.location.origin}`;
                if (navigator.share) {
                  navigator.share({ title: "CardioRisk", text: shareText, url }).catch(() => {});
                } else {
                  navigator.clipboard.writeText(`${shareText} ${url}`);
                  setShareMessage("Category-only share text copied.");
                  window.setTimeout(() => setShareMessage(""), 2500);
                }
              }}
            >
              Share Category Card
            </button>
          </div>
          {shareMessage ? <p className="lead" style={{ margin: 0 }}>{shareMessage}</p> : null}
        </motion.div>

        <motion.div
          className="card-panel"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <RiskTimeline
            history={history}
            onExportCsv={() => {
              if (historyCount === 0) return;
              const headers = [
                "timestamp",
                "score",
                "category",
                "age",
                "systolic_bp",
                "total_cholesterol",
                "hdl_cholesterol",
                "smoking",
                "activity_level",
              ];
              const rows = history.map((entry) =>
                [
                  entry.timestamp,
                  entry.score.toFixed(2),
                  entry.category,
                  entry.age ?? "",
                  entry.systolicBp ?? "",
                  entry.totalCholesterol ?? "",
                  entry.hdlCholesterol ?? "",
                  entry.smoking === null ? "" : entry.smoking ? "yes" : "no",
                  entry.activityLevel ?? "",
                ].join(","),
              );
              const csv = [headers.join(","), ...rows].join("\n");
              const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `cardiorisk-history-${new Date().toISOString().slice(0, 10)}.csv`;
              a.click();
              URL.revokeObjectURL(url);
            }}
            onClearHistory={() => {
              if (window.confirm("Delete all saved assessment history from this browser?")) {
                clearHistory();
              }
            }}
          />
        </motion.div>

        <motion.div
          className="card-panel action-sidebar"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
        >
          <h2 className="type-h3">Take Action</h2>
          <p className="lead">Top 3 personalized priorities:</p>
          <div className="action-list">
            {activeResult!.recommendations.slice(0, 3).map((item) => (
              <div key={item} className="action-item">
                <CheckCircle2 size={16} />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
