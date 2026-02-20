"use client";

import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  GaugeCircle,
  HeartPulse,
  Info,
  Loader2,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import clsx from "clsx";

interface PredictionRequest {
  age: number;
  sex: number;
  cp: number;
  trestbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalach: number;
  exang: number;
  oldpeak: number;
  slope: number;
  ca: number;
  thal: number;
}

interface PredictionResponse {
  prediction: number;
  probability: number;
  risk_level: "Low" | "Medium" | "High";
  confidence: number;
  timestamp: string;
  top_contributors?: Record<string, number>[] | null;
}

const initialFormData: PredictionRequest = {
  age: 50,
  sex: 1,
  cp: 0,
  trestbps: 120,
  chol: 200,
  fbs: 0,
  restecg: 0,
  thalach: 150,
  exang: 0,
  oldpeak: 0,
  slope: 1,
  ca: 0,
  thal: 2,
};

const fieldMeta: Array<{ key: keyof PredictionRequest; label: string }> = [
  { key: "age", label: "Age" },
  { key: "trestbps", label: "Resting BP" },
  { key: "chol", label: "Cholesterol" },
  { key: "thalach", label: "Max HR" },
  { key: "oldpeak", label: "ST Depression" },
];

const riskAdvice = {
  High: [
    "Escalate to urgent cardiology consult.",
    "Prioritize ECG and troponin-based workup.",
    "Consider inpatient monitoring based on symptoms.",
  ],
  Medium: [
    "Schedule specialist follow-up within 1-4 weeks.",
    "Reinforce lifestyle + blood pressure control.",
    "Repeat risk reassessment after interventions.",
  ],
  Low: [
    "Continue preventive screening cadence.",
    "Maintain exercise and lipid management plan.",
    "Reassess if symptoms or vitals change.",
  ],
} as const;

function validateField(name: keyof PredictionRequest, value: number, all: PredictionRequest): string | null {
  if (!Number.isFinite(value)) return "Required";

  switch (name) {
    case "age":
      return value < 1 || value > 120 ? "Age must be 1-120" : null;
    case "trestbps":
      return value < 50 || value > 300 ? "BP must be 50-300" : null;
    case "chol":
      return value < 50 || value > 800 ? "Cholesterol must be 50-800" : null;
    case "thalach": {
      if (value < 30 || value > 250) return "Heart rate must be 30-250";
      const limit = (220 - all.age) * 1.1;
      return value > limit ? `Likely too high for age (${Math.round(limit)} bpm)` : null;
    }
    case "oldpeak":
      return value < 0 || value > 10 ? "ST depression must be 0-10" : null;
    default:
      return null;
  }
}

export default function Home() {
  const [formData, setFormData] = useState<PredictionRequest>(initialFormData);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const completeness = useMemo(() => {
    const done = Object.values(formData).filter((v) => Number.isFinite(v)).length;
    return Math.round((done / Object.keys(formData).length) * 100);
  }, [formData]);

  const validationStatus = Object.keys(errors).length === 0 ? "All checks passed" : `${Object.keys(errors).length} validation issue(s)`;

  const handleInputChange = (field: keyof PredictionRequest, value: number) => {
    const next = { ...formData, [field]: value };
    setFormData(next);

    const message = validateField(field, value, next);
    setErrors((prev) => {
      const out = { ...prev };
      if (message) out[field] = message;
      else delete out[field];
      return out;
    });
  };

  const applyPreset = (preset: "baseline" | "highRisk") => {
    const next =
      preset === "baseline"
        ? initialFormData
        : {
            ...initialFormData,
            age: 68,
            cp: 3,
            trestbps: 172,
            chol: 295,
            fbs: 1,
            thalach: 118,
            exang: 1,
            oldpeak: 3.2,
            ca: 3,
          };

    setFormData(next);
    const nextErrors: Record<string, string> = {};
    fieldMeta.forEach(({ key }) => {
      const msg = validateField(key, next[key], next);
      if (msg) nextErrors[key] = msg;
    });
    setErrors(nextErrors);
    setResult(null);
    setApiError(null);
  };

  const hasErrors = Object.keys(errors).length > 0;
  const gauge = Math.max(0, Math.min(100, (result?.probability ?? 0) * 100));

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (hasErrors) return;

    setLoading(true);
    setApiError(null);
    setResult(null);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": process.env.NEXT_PUBLIC_API_KEY ?? "cardiovascular-risk-secret-key-123",
        },
        body: JSON.stringify(formData),
      });

      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(payload?.detail ?? "Unable to complete risk analysis");
      }

      setResult(payload as PredictionResponse);
    } catch (error: unknown) {
      setApiError(error instanceof Error ? error.message : "Unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen pb-16 bg-slate-50 relative overflow-hidden">
      <div className="absolute inset-x-0 top-0 h-80 bg-gradient-to-b from-blue-100/60 via-sky-50 to-transparent pointer-events-none" />

      <header className="relative z-10 max-w-7xl mx-auto px-6 pt-12 pb-8">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div>
            <div className="brand-pill mb-4">
              <HeartPulse size={16} />
              CardioRisk AI · Clinical Workbench
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-slate-900">Enhanced Cardiovascular Risk Triage</h1>
            <p className="text-slate-600 mt-3 max-w-3xl">
              Faster clinician workflow with clearer validation, richer scorecards, and explainability-aware outputs.
            </p>
          </div>
          <div className="hidden md:flex items-center gap-5 text-sm">
            <div className="info-chip"><ShieldCheck size={15} /> Guardrails active</div>
            <div className="info-chip"><Sparkles size={15} /> Explainability ready</div>
          </div>
        </div>
      </header>

      <section className="relative z-10 max-w-7xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        <form onSubmit={submit} className="lg:col-span-8 space-y-6">
          <div className="medical-card p-6 md:p-8">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
              <h2 className="section-title"><Activity size={18} /> Patient Intake</h2>
              <div className="flex gap-2">
                <button type="button" className="preset-btn" onClick={() => applyPreset("baseline")}>Baseline preset</button>
                <button type="button" className="preset-btn" onClick={() => applyPreset("highRisk")}>High-risk preset</button>
              </div>
            </div>

            <div className="status-grid mb-6">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide">Completeness</p>
                <p className="text-lg font-semibold text-slate-900">{completeness}%</p>
              </div>
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide">Validation</p>
                <p className={clsx("text-sm font-semibold", hasErrors ? "text-amber-600" : "text-emerald-600")}>{validationStatus}</p>
              </div>
              <div className="status-bar"><span style={{ width: `${completeness}%` }} /></div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <InputGroup label="Age" error={errors.age} suffix="years"><input type="number" className="medical-input" value={formData.age} onChange={(e) => handleInputChange("age", Number(e.target.value))} /></InputGroup>
              <InputGroup label="Sex">
                <div className="btn-select-group">
                  <button type="button" className={clsx("btn-select-option", formData.sex === 1 && "selected")} onClick={() => handleInputChange("sex", 1)}>Male</button>
                  <button type="button" className={clsx("btn-select-option", formData.sex === 0 && "selected")} onClick={() => handleInputChange("sex", 0)}>Female</button>
                </div>
              </InputGroup>

              <InputGroup label="Resting BP" error={errors.trestbps} suffix="mmHg"><input type="number" className="medical-input" value={formData.trestbps} onChange={(e) => handleInputChange("trestbps", Number(e.target.value))} /></InputGroup>
              <InputGroup label="Cholesterol" error={errors.chol} suffix="mg/dL"><input type="number" className="medical-input" value={formData.chol} onChange={(e) => handleInputChange("chol", Number(e.target.value))} /></InputGroup>

              <InputGroup label="Max Heart Rate" error={errors.thalach} suffix="bpm"><input type="number" className="medical-input" value={formData.thalach} onChange={(e) => handleInputChange("thalach", Number(e.target.value))} /></InputGroup>
              <InputGroup label="Fasting Blood Sugar > 120">
                <div className="btn-select-group">
                  <button type="button" className={clsx("btn-select-option", formData.fbs === 1 && "selected")} onClick={() => handleInputChange("fbs", 1)}>Yes</button>
                  <button type="button" className={clsx("btn-select-option", formData.fbs === 0 && "selected")} onClick={() => handleInputChange("fbs", 0)}>No</button>
                </div>
              </InputGroup>

              <InputGroup label="Chest Pain Type">
                <select className="medical-input" value={formData.cp} onChange={(e) => handleInputChange("cp", Number(e.target.value))}>
                  <option value={0}>Typical Angina</option>
                  <option value={1}>Atypical Angina</option>
                  <option value={2}>Non-anginal Pain</option>
                  <option value={3}>Asymptomatic</option>
                </select>
              </InputGroup>
              <InputGroup label="Resting ECG">
                <select className="medical-input" value={formData.restecg} onChange={(e) => handleInputChange("restecg", Number(e.target.value))}>
                  <option value={0}>Normal</option>
                  <option value={1}>ST-T Wave Abnormality</option>
                  <option value={2}>Left Ventricular Hypertrophy</option>
                </select>
              </InputGroup>

              <InputGroup label="Exercise-induced Angina">
                <div className="btn-select-group">
                  <button type="button" className={clsx("btn-select-option", formData.exang === 1 && "selected")} onClick={() => handleInputChange("exang", 1)}>Yes</button>
                  <button type="button" className={clsx("btn-select-option", formData.exang === 0 && "selected")} onClick={() => handleInputChange("exang", 0)}>No</button>
                </div>
              </InputGroup>
              <InputGroup label="ST Depression" error={errors.oldpeak}><input type="number" step="0.1" className="medical-input" value={formData.oldpeak} onChange={(e) => handleInputChange("oldpeak", Number(e.target.value))} /></InputGroup>

              <InputGroup label="ST Slope">
                <select className="medical-input" value={formData.slope} onChange={(e) => handleInputChange("slope", Number(e.target.value))}>
                  <option value={0}>Upsloping</option>
                  <option value={1}>Flat</option>
                  <option value={2}>Downsloping</option>
                </select>
              </InputGroup>
              <InputGroup label="Major Vessels (ca)"><input type="number" className="medical-input" value={formData.ca} onChange={(e) => handleInputChange("ca", Number(e.target.value))} /></InputGroup>
            </div>

            <div className="mt-7 flex justify-end">
              <button type="submit" disabled={loading || hasErrors} className="btn-medical-primary px-8 py-3.5 flex items-center gap-2">
                {loading ? <Loader2 size={18} className="animate-spin" /> : <GaugeCircle size={18} />}
                {loading ? "Analyzing..." : "Run Risk Analysis"}
                <ArrowRight size={17} />
              </button>
            </div>
          </div>
        </form>

        <aside className="lg:col-span-4 space-y-5 lg:sticky lg:top-6 lg:h-fit">
          <AnimatePresence mode="wait">
            {!result && !loading && (
              <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="result-shell">
                <Activity size={26} />
                <h3 className="font-semibold text-slate-800">Awaiting Analysis</h3>
                <p className="text-sm text-slate-500 text-center">Enter patient data and submit to generate a stratified clinical risk report.</p>
              </motion.div>
            )}

            {loading && (
              <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="result-shell">
                <Loader2 size={28} className="animate-spin" />
                <h3 className="font-semibold text-slate-800">Inference in progress</h3>
                <p className="text-sm text-slate-500">Computing risk probability and confidence profile...</p>
              </motion.div>
            )}

            {result && (
              <motion.div key="result" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={clsx("result-card", `tone-${result.risk_level.toLowerCase()}`)}>
                <p className="text-xs uppercase tracking-wide text-slate-500">Predicted risk</p>
                <h3 className="text-3xl font-bold mt-1">{result.risk_level}</h3>

                <div className="gauge-wrap mt-4">
                  <div className="risk-gauge" style={{ background: `conic-gradient(var(--gauge-color) ${gauge}%, #e2e8f0 ${gauge}% 100%)` }}>
                    <div className="risk-gauge-core">
                      <span className="text-2xl font-bold">{gauge.toFixed(0)}%</span>
                      <span className="text-[11px] text-slate-500">Probability</span>
                    </div>
                  </div>
                </div>

                <div className="metric-row mt-4">
                  <span>Model confidence</span>
                  <strong>{(result.confidence * 100).toFixed(0)}%</strong>
                </div>
                <div className="metric-track"><span style={{ width: `${result.confidence * 100}%` }} /></div>

                {result.top_contributors && result.top_contributors.length > 0 && (
                  <div className="mt-5">
                    <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Top contributing factors</p>
                    <ul className="space-y-2">
                      {result.top_contributors.map((entry, idx) => {
                        const [k, v] = Object.entries(entry)[0];
                        return (
                          <li key={`${k}-${idx}`} className="contrib-row">
                            <span>{k}</span>
                            <span>{v.toFixed(2)}</span>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}

                <div className="advice-box mt-5">
                  <Info size={16} className="shrink-0 mt-0.5" />
                  <ul className="space-y-1">
                    {riskAdvice[result.risk_level].map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {apiError && (
            <div className="error-box">
              <AlertCircle size={18} className="shrink-0 mt-0.5" />
              <p className="text-sm">{apiError}</p>
            </div>
          )}

          <div className="trust-box">
            <CheckCircle2 size={16} className="text-emerald-600" />
            <p className="text-xs text-slate-600">Decision-support only. Always combine with physician judgment and complete diagnostic context.</p>
          </div>
        </aside>
      </section>
    </main>
  );
}

function InputGroup({
  label,
  children,
  suffix,
  error,
}: {
  label: string;
  children: React.ReactNode;
  suffix?: string;
  error?: string;
}) {
  return (
    <div>
      <label className={clsx("medical-label flex items-center justify-between", error && "text-red-500")}>
        {label}
        {error ? <span className="text-[10px] normal-case font-normal">{error}</span> : null}
      </label>
      <div className="relative">
        {children}
        {suffix ? <span className="medical-suffix">{suffix}</span> : null}
      </div>
    </div>
  );
}
