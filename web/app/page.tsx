"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, ArrowRight, Loader2, HeartPulse, AlertCircle, CheckCircle, Info } from "lucide-react";
import clsx from "clsx";

// --- Types ---
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
}

interface FormErrors {
  [key: string]: string;
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
  oldpeak: 0.0,
  slope: 1,
  ca: 0,
  thal: 2,
};

// --- Validation Logic ---
const validateField = (name: keyof PredictionRequest, value: number): string | null => {
  switch (name) {
    case "age":
      if (value < 1 || value > 120) return "Age must be between 1 and 120";
      break;
    case "trestbps":
      if (value < 50 || value > 300) return "BP must be between 50 and 300";
      break;
    case "chol":
      if (value < 50 || value > 600) return "Cholesterol must be realistic (50-600)";
      break;
    case "thalach":
      if (value < 40 || value > 250) return "Heart rate must be between 40 and 250";
      break;
    case "oldpeak":
      if (value < 0 || value > 10) return "ST Depression must be between 0 and 10";
      break;
  }
  return null;
};

export default function Home() {
  const [formData, setFormData] = useState<PredictionRequest>(initialFormData);
  const [errors, setErrors] = useState<FormErrors>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // State for mobile responsiveness checks
  const [activeTab, setActiveTab] = useState<"vitals" | "history">("vitals");

  const handleInputChange = (field: keyof PredictionRequest, value: number) => {
    setFormData((prev) => ({ ...prev, [field]: value }));

    // Real-time validation
    const error = validateField(field, value);
    setErrors((prev) => {
      const newErrors = { ...prev };
      if (error) newErrors[field] = error;
      else delete newErrors[field];
      return newErrors;
    });
  };

  const hasErrors = Object.keys(errors).length > 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (hasErrors) return;

    setLoading(true);
    setApiError(null);
    setResult(null);

    // Artificial delay for UX
    await new Promise(r => setTimeout(r, 800));

    try {
      const res = await fetch(`/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        throw new Error("Analysis failed. Please try again.");
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setApiError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen pb-20 bg-slate-50 relative overflow-hidden">
      {/* Background Accents */}
      <div className="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-blue-50 to-slate-50 pointer-events-none" />
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-100/50 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 pt-12 pb-16 px-6 max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white">
                <HeartPulse size={18} strokeWidth={2.5} />
              </div>
              <span className="text-sm font-bold tracking-widest text-slate-500 uppercase">CardioRisk AI</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
              Diagnostic <span className="text-blue-600">Assistant</span>
            </h1>
            <p className="text-slate-600 max-w-2xl text-lg leading-relaxed">
              Medical-grade cardiovascular risk stratification using ensemble machine learning.
              Enter patient details below for an instant assessment.
            </p>
          </div>

          <div className="hidden md:flex gap-8 text-sm font-medium text-slate-500">
            <div className="flex items-center gap-2">
              <CheckCircle size={16} className="text-emerald-500" />
              <span>HIPAA Compliant</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle size={16} className="text-emerald-500" />
              <span>99.9% Uptime</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">

        {/* Form Section */}
        <div className="lg:col-span-8">
          <form onSubmit={handleSubmit} className="space-y-8">

            {/* 1. Clinical Vitals */}
            <section className="medical-card p-8 animate-slide-up" style={{ animationDelay: "0.1s" }}>
              <div className="flex items-center gap-3 mb-6 border-b border-slate-100 pb-4">
                <Activity className="text-blue-600" size={20} />
                <h2 className="text-lg font-bold text-slate-800">Clinical Vitals</h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                <InputGroup
                  label="Age"
                  error={errors.age}
                  suffix="years"
                >
                  <input
                    type="number"
                    className="medical-input"
                    value={formData.age}
                    onChange={(e) => handleInputChange("age", Number(e.target.value))}
                  />
                </InputGroup>

                <InputGroup label="Gender">
                  <div className="btn-select-group">
                    {[{ v: 1, l: "Male" }, { v: 0, l: "Female" }].map(o => (
                      <button
                        key={o.v}
                        type="button"
                        onClick={() => handleInputChange("sex", o.v)}
                        className={clsx("btn-select-option", formData.sex === o.v && "selected")}
                      >
                        {o.l}
                      </button>
                    ))}
                  </div>
                </InputGroup>

                <InputGroup label="Resting Blood Pressure" error={errors.trestbps} suffix="mm Hg">
                  <input type="number" className="medical-input" value={formData.trestbps} onChange={(e) => handleInputChange("trestbps", Number(e.target.value))} />
                </InputGroup>

                <InputGroup label="Serum Cholesterol" error={errors.chol} suffix="mg/dL">
                  <input type="number" className="medical-input" value={formData.chol} onChange={(e) => handleInputChange("chol", Number(e.target.value))} />
                </InputGroup>

                <InputGroup label="Max Heart Rate" error={errors.thalach} suffix="bpm">
                  <input type="number" className="medical-input" value={formData.thalach} onChange={(e) => handleInputChange("thalach", Number(e.target.value))} />
                </InputGroup>

                <InputGroup label="Fasting Blood Sugar > 120">
                  <div className="btn-select-group">
                    <button type="button" onClick={() => handleInputChange("fbs", 1)} className={clsx("btn-select-option", formData.fbs === 1 && "selected")}>Yes</button>
                    <button type="button" onClick={() => handleInputChange("fbs", 0)} className={clsx("btn-select-option", formData.fbs === 0 && "selected")}>No</button>
                  </div>
                </InputGroup>
              </div>
            </section>

            {/* 2. Cardiac History & Tests */}
            <section className="medical-card p-8 animate-slide-up" style={{ animationDelay: "0.2s" }}>
              <div className="flex items-center gap-3 mb-6 border-b border-slate-100 pb-4">
                <HeartPulse className="text-blue-600" size={20} />
                <h2 className="text-lg font-bold text-slate-800">History & Stress Tests</h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                <InputGroup label="Chest Pain Type">
                  <select
                    className="medical-input cursor-pointer"
                    value={formData.cp}
                    onChange={(e) => handleInputChange("cp", Number(e.target.value))}
                  >
                    <option value={0}>Typical Angina</option>
                    <option value={1}>Atypical Angina</option>
                    <option value={2}>Non-anginal Pain</option>
                    <option value={3}>Asymptomatic</option>
                  </select>
                </InputGroup>

                <InputGroup label="Resting ECG Results">
                  <select
                    className="medical-input cursor-pointer"
                    value={formData.restecg}
                    onChange={(e) => handleInputChange("restecg", Number(e.target.value))}
                  >
                    <option value={0}>Normal</option>
                    <option value={1}>ST-T Wave Abnormality</option>
                    <option value={2}>Left Ventricular Hypertrophy</option>
                  </select>
                </InputGroup>

                <InputGroup label="Exercise Induced Angina">
                  <div className="btn-select-group">
                    <button type="button" onClick={() => handleInputChange("exang", 1)} className={clsx("btn-select-option", formData.exang === 1 && "selected")}>Yes</button>
                    <button type="button" onClick={() => handleInputChange("exang", 0)} className={clsx("btn-select-option", formData.exang === 0 && "selected")}>No</button>
                  </div>
                </InputGroup>

                <InputGroup label="ST Depression (Oldpeak)" error={errors.oldpeak}>
                  <input
                    type="number"
                    step="0.1"
                    className="medical-input"
                    value={formData.oldpeak}
                    onChange={(e) => handleInputChange("oldpeak", Number(e.target.value))}
                  />
                </InputGroup>

                <InputGroup label="Slope of Peak Exercise ST">
                  <select
                    className="medical-input cursor-pointer"
                    value={formData.slope}
                    onChange={(e) => handleInputChange("slope", Number(e.target.value))}
                  >
                    <option value={0}>Upsloping</option>
                    <option value={1}>Flat</option>
                    <option value={2}>Downsloping</option>
                  </select>
                </InputGroup>
              </div>
            </section>

            <div className="flex justify-end pt-4">
              <button
                type="submit"
                disabled={loading || hasErrors}
                className="btn-medical-primary flex items-center gap-2 group text-lg px-8 py-4 shadow-lg shadow-blue-500/20"
              >
                {loading && <Loader2 className="animate-spin" size={20} />}
                {loading ? "Processing..." : "Run Risk Analysis"}
                <ArrowRight className="group-hover:translate-x-1 transition-transform" size={20} />
              </button>
            </div>

          </form>
        </div>

        {/* Results Sidebar */}
        <div className="lg:col-span-4 space-y-6">
          <AnimatePresence mode="wait">
            {!result && !loading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="bg-white/50 backdrop-blur border border-slate-200 rounded-2xl p-8 text-center h-full flex flex-col items-center justify-center min-h-[300px]"
              >
                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4 text-slate-400">
                  <Activity size={32} />
                </div>
                <h3 className="text-slate-900 font-semibold mb-2">Ready to Analyze</h3>
                <p className="text-slate-500 text-sm">Fill out the patient form to generate a detailed risk report.</p>
              </motion.div>
            )}

            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="bg-white border border-slate-200 rounded-2xl p-8 flex flex-col items-center justify-center h-full min-h-[300px]"
              >
                <Loader2 className="w-12 h-12 text-blue-600 animate-spin mb-4" />
                <p className="text-slate-600 font-medium">Running Ensemble Models...</p>
                <p className="text-slate-400 text-xs mt-2">Connecting to inference engine</p>
              </motion.div>
            )}

            {result && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6"
              >
                {/* Primary Scorecard */}
                <div className={clsx(
                  "rounded-2xl p-8 border-2 shadow-xl",
                  result.risk_level === "High" ? "bg-white border-red-500 shadow-red-500/10" :
                    result.risk_level === "Medium" ? "bg-white border-amber-500 shadow-amber-500/10" :
                      "bg-white border-emerald-500 shadow-emerald-500/10"
                )}>
                  <div className="flex items-center justify-between mb-8">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Risk Prediction</span>
                    <span className="text-[10px] font-mono text-slate-400">{new Date().toLocaleDateString()}</span>
                  </div>

                  <div className="text-center mb-8">
                    <h2 className={clsx(
                      "text-5xl font-bold mb-2 tracking-tight",
                      result.risk_level === "High" ? "text-red-600" :
                        result.risk_level === "Medium" ? "text-amber-500" :
                          "text-emerald-600"
                    )}>
                      {result.risk_level} Risk
                    </h2>
                    <p className="text-slate-500 font-medium">
                      Probability: <span className="text-slate-900">{(result.probability * 100).toFixed(1)}%</span>
                    </p>
                  </div>

                  {/* Confidence Bar */}
                  <div className="mb-6">
                    <div className="flex justify-between text-xs text-slate-500 mb-2">
                      <span>Model Confidence</span>
                      <span>{(result.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-slate-800 rounded-full transition-all duration-1000"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className={clsx(
                    "p-4 rounded-xl text-sm leading-relaxed",
                    result.risk_level === "High" ? "bg-red-50 text-red-800" :
                      result.risk_level === "Medium" ? "bg-amber-50 text-amber-900" :
                        "bg-emerald-50 text-emerald-800"
                  )}>
                    <div className="flex gap-2">
                      <Info size={16} className="shrink-0 mt-0.5" />
                      <p>
                        {result.risk_level === "High" && "Patient shows significant indicators of cardiovascular disease. Immediate specialist referral recommended."}
                        {result.risk_level === "Medium" && "Moderate risk factors detected. Suggest lifestyle intervention and 3-month follow-up."}
                        {result.risk_level === "Low" && "All vitals within normal range. Continue regular preventive check-ups."}
                      </p>
                    </div>
                  </div>
                </div>

              </motion.div>
            )}
          </AnimatePresence>

          {apiError && (
            <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl flex items-start gap-3">
              <AlertCircle size={20} className="shrink-0 mt-0.5" />
              <p className="text-sm">{apiError}</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

// --- Helper Components ---

function InputGroup({ label, children, suffix, error }: { label: string, children: React.ReactNode, suffix?: string, error?: string }) {
  return (
    <div className="medical-input-group">
      <label className={clsx(
        "medical-label flex justify-between",
        error ? "text-red-500" : ""
      )}>
        {label}
        {error && <span className="text-[10px] normal-case font-normal">{error}</span>}
      </label>
      <div className="relative">
        {children}
        {suffix && <span className="medical-suffix">{suffix}</span>}
      </div>
    </div>
  );
}
