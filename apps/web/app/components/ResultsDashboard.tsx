"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, GaugeCircle, ShieldCheck, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { getRiskCategory, type RiskResult, scoreRisk, type AssessmentData } from "../lib/risk";

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

export function ResultsDashboard() {
  const [result, setResult] = useState<RiskResult | null>(null);
  const [hasData, setHasData] = useState(true);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        const raw = window.sessionStorage.getItem(STORAGE_KEY);
        if (!raw) {
          setHasData(false);
          return;
        }
        const parsed = JSON.parse(raw) as StoredPayload;
        if (parsed?.result) {
          setResult(parsed.result);
        } else if (parsed?.data) {
          setResult(scoreRisk(parsed.data));
        } else {
          setHasData(false);
        }
      } catch {
        setHasData(false);
      }
    }, 0);

    return () => window.clearTimeout(timer);
  }, []);

  const chartStyle = useMemo(() => {
    if (!result) return {};
    const degree = Math.max(6, Math.round((result.score / 45) * 360));
    return {
      background: `conic-gradient(from -90deg, ${
        result.category === "low" ? "#27ae60" : result.category === "moderate" ? "#f39c12" : "#e74c3c"
      } 0deg, ${
        result.category === "low" ? "#27ae60" : result.category === "moderate" ? "#f39c12" : "#e74c3c"
      } ${degree}deg, #eef1f4 ${degree}deg 360deg)`,
    };
  }, [result]);

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
        <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <div className="dashboard-head">
            <h1 className="type-h2">Your Cardiovascular Risk Dashboard</h1>
            <span className={statusClass(result.score)}>{result.category.toUpperCase()} RISK</span>
          </div>
          <p className="lead">{result.summary}</p>

          <div className="gauge-row">
            <div className="gauge-shell" style={{ ...chartStyle, boxShadow: `0 20px 40px rgba(26, 26, 46, 0.12)` }}>
              <div className="gauge-core">
                <strong className="mono score">{result.score.toFixed(1)}%</strong>
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
        </motion.div>

        <motion.div
          className="card-panel"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <h2 className="type-h3">Top Risk Drivers</h2>
          <div className="driver-list">
            {result.drivers.length === 0 ? (
              <p className="lead">No major upward contributors detected from submitted data.</p>
            ) : (
              result.drivers.map((driver) => (
                <div className="driver-row" key={driver.label}>
                  <span>{driver.label}</span>
                  <strong className="mono">+{(driver.impact * 100).toFixed(1)} pts</strong>
                </div>
              ))
            )}
          </div>

          <h2 className="type-h3">Action Plan</h2>
          <div className="action-list">
            {result.recommendations.map((item) => (
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
          <Link href="/assess" className="btn btn-subtle">
            Re-run Assessment
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
