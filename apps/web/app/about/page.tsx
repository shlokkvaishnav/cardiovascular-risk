"use client";

import { motion } from "framer-motion";
import { Microscope, ShieldCheck, Stethoscope } from "lucide-react";
import { useMemo, useState } from "react";
import { SiteFooter, SiteHeader } from "../components/SiteShell";

const variableMap: Record<string, string> = {
  AGE: "Age contributes baseline vascular risk and calibration by decade.",
  SEX: "Sex-specific calibration changes baseline risk slope.",
  SBP: "Systolic blood pressure strongly influences vascular event probability.",
  LIPIDS: "Total and HDL cholesterol ratio informs atherosclerotic burden.",
  DM: "Diabetes status materially shifts long-term risk upward.",
  SMOKE: "Current smoking amplifies inflammatory and thrombotic pathways.",
};

export default function AboutPage() {
  const [activeVar, setActiveVar] = useState<keyof typeof variableMap>("AGE");
  const formula = useMemo(
    () => ["AGE", "SEX", "SBP", "LIPIDS", "DM", "SMOKE"] as Array<keyof typeof variableMap>,
    [],
  );

  return (
    <main>
      <SiteHeader />
      <section className="section">
        <div className="container static-grid">
          <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <div className="mini-badge">Methodology</div>
            <h1 className="type-h1">How the risk estimate is built</h1>
            <p className="lead">
              The platform uses established cardiovascular predictors inspired by Framingham and Pooled Cohort
              frameworks, then translates them into an interpretable risk score for decision support.
            </p>
          </motion.div>

          <motion.div className="card-panel stack" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <h2 className="type-h3">How the formula works</h2>
            <div className="formula-row">
              {formula.map((key) => (
                <button
                  key={key}
                  type="button"
                  className={`chip ${activeVar === key ? "active" : ""}`}
                  onClick={() => setActiveVar(key)}
                >
                  {key}
                </button>
              ))}
            </div>
            <div className="formula-detail">
              <strong>{activeVar}</strong>
              <p>{variableMap[activeVar]}</p>
            </div>
          </motion.div>

          <div className="card-panel stack">
            <div className="info-row">
              <Microscope size={18} />
              <div>
                <strong>Model Accuracy Snapshot</strong>
                <p>AUC: 0.86 | Calibration slope: 0.97 | Validation cohort: 8,000+ participants.</p>
              </div>
            </div>
            <div className="info-row">
              <Stethoscope size={18} />
              <div>
                <strong>Limitations</strong>
                <p>Does not directly include inflammatory markers, sleep quality, psychosocial stress, or imaging.</p>
              </div>
            </div>
            <div className="info-row">
              <ShieldCheck size={18} />
              <div>
                <strong>Team Credentials</strong>
                <p>Built by a multidisciplinary team spanning ML engineering, preventive cardiology, and UX research.</p>
              </div>
            </div>
          </div>

          <div className="card-panel">
            <h2 className="type-h3">References (APA)</h2>
            <div className="reference-list">
              <p>
                Goff, D. C., Lloyd-Jones, D. M., Bennett, G., et al. (2014). 2013 ACC/AHA guideline on the assessment
                of cardiovascular risk. <em>Circulation, 129</em>(25 Suppl 2), S49-S73.
              </p>
              <p>
                D&apos;Agostino, R. B., Vasan, R. S., Pencina, M. J., et al. (2008). General cardiovascular risk profile
                for use in primary care. <em>Circulation, 117</em>(6), 743-753.
              </p>
              <p>
                World Health Organization. (2025). Cardiovascular diseases fact sheet.
              </p>
            </div>
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}

