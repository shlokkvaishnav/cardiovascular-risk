"use client";

import { motion } from "framer-motion";
import { Microscope, ShieldCheck, Stethoscope } from "lucide-react";
import { useMemo, useState } from "react";
import { SiteFooter, SiteHeader } from "../components/SiteShell";

const variableMap: Record<string, string> = {
  AGE: "Age contributes baseline vascular risk and calibration by decade.",
  SEX: "Sex-specific calibration changes baseline risk slope.",
  BMI: "Body mass index (from height and weight) reflects long-term metabolic load.",
  BP: "Systolic and diastolic blood pressure strongly influence vascular event probability.",
  CHOL: "Cholesterol and glucose category inform atherosclerotic and metabolic burden.",
  LIFESTYLE: "Smoking, alcohol intake, and physical activity are modifiable behavioral factors.",
};

export default function AboutPage() {
  const [activeVar, setActiveVar] = useState<keyof typeof variableMap>("AGE");
  const formula = useMemo(
    () => ["AGE", "SEX", "BMI", "BP", "CHOL", "LIFESTYLE"] as Array<keyof typeof variableMap>,
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
                <p>
                  ROC-AUC: 0.79 | Accuracy: 72% | Trained on 70,000 records from the public Kaggle cardiovascular
                  disease dataset (49,000 train / 21,000 held-out test). See the model card for the full breakdown.
                </p>
              </div>
            </div>
            <div className="info-row">
              <Stethoscope size={18} />
              <div>
                <strong>Limitations</strong>
                <p>
                  This is a lifestyle risk-screening model, not a diagnostic tool -- it uses self-reportable factors
                  (age, sex, BMI, blood pressure, cholesterol/glucose category, smoking, alcohol, activity) and does
                  not include ECG, imaging, lab-confirmed lipid panels, or family history.
                </p>
              </div>
            </div>
            <div className="info-row">
              <ShieldCheck size={18} />
              <div>
                <strong>Explainability</strong>
                <p>
                  Every prediction is accompanied by real SHAP (SHapley Additive exPlanations) values showing which
                  factors pushed your risk estimate up or down, not just a static list of &quot;important&quot; features.
                </p>
              </div>
            </div>
          </div>

          <div className="card-panel">
            <h2 className="type-h3">References (APA)</h2>
            <div className="reference-list">
              <p>
                Ulianova, S. (2019). Cardiovascular Disease dataset [Data set]. Kaggle.
                https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
              </p>
              <p>
                Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions.
                <em> Advances in Neural Information Processing Systems, 30</em>.
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

