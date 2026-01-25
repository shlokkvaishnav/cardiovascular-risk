"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, Heart, AlertCircle, CheckCircle2, Info, ArrowRight, Loader2 } from "lucide-react";
import clsx from "clsx";

// Types corresponding to the API
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

const initialFormData: PredictionRequest = {
  age: 50,
  sex: 1,
  cp: 0,
  trestbps: 120,
  chol: 200,
  fbs: 0,
  restecg: 1,
  thalach: 150,
  exang: 0,
  oldpeak: 1.0,
  slope: 1,
  ca: 0,
  thal: 2,
};

export default function Home() {
  const [formData, setFormData] = useState<PredictionRequest>(initialFormData);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: keyof PredictionRequest, value: number) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Use relative path for same-domain deployment
      const API_URL = "/api";
      const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "cardiovascular-risk-secret-key-123";

      // Note: We mount the API at /api/v1 in FastAPI, or just proxy. 
      // Let's assume FastAPI mounts the API router at /api or the root.
      // If we assume the existing FastAPI structure, the endpoint is /predict.
      // So if we serve frontend at /, and API is at /, then we fetch /predict.

      const res = await fetch(`/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY,
        },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to fetch prediction");
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen mesh-gradient text-slate-100 selection:bg-blue-500/30">
      {/* Enhanced Mesh Gradient Background */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-15%] left-[-10%] w-[50%] h-[50%] bg-blue-600/20 rounded-full blur-[120px] animate-float" />
        <div className="absolute top-[20%] right-[-5%] w-[40%] h-[40%] bg-purple-600/15 rounded-full blur-[100px] animate-float" style={{ animationDelay: "2s" }} />
        <div className="absolute bottom-[-10%] left-[30%] w-[45%] h-[45%] bg-pink-600/10 rounded-full blur-[110px] animate-float" style={{ animationDelay: "4s" }} />
        <div className="absolute bottom-[30%] right-[20%] w-[35%] h-[35%] bg-cyan-500/10 rounded-full blur-[90px] animate-float" style={{ animationDelay: "1s" }} />

        {/* Noise texture overlay */}
        <div className="absolute inset-0 opacity-[0.015]" style={{
          backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 400 400\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noiseFilter\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.9\' numOctaves=\'4\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noiseFilter)\'/%3E%3C/svg%3E")',
        }} />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Enhanced Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-20"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="inline-flex items-center justify-center gap-2 px-4 py-2.5 mb-6 rounded-full bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 border border-blue-500/20 backdrop-blur-xl shadow-[0_0_30px_rgba(59,130,246,0.15)]"
          >
            <div className="relative">
              <Activity className="w-4 h-4 text-blue-400 animate-pulse-glow" />
              <div className="absolute inset-0 w-4 h-4 bg-blue-400/30 blur-sm animate-pulse-glow" />
            </div>
            <span className="text-blue-200 text-sm font-semibold tracking-wide">AI-Powered Health Analytics</span>
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse shadow-[0_0_8px_rgba(74,222,128,0.8)]" />
          </motion.div>

          {/* Main Title */}
          <h1 className="text-6xl md:text-7xl lg:text-8xl font-black mb-6 tracking-tight leading-none">
            <motion.span
              className="block mb-2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              Cardiovascular Risk
            </motion.span>
            <motion.span
              className="gradient-text block"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              Prediction
            </motion.span>
          </h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="text-slate-400 text-lg md:text-xl max-w-3xl mx-auto leading-relaxed font-light text-balance"
          >
            Advanced machine learning ensemble model combining{" "}
            <span className="text-blue-400 font-medium">Logistic Regression</span>,{" "}
            <span className="text-purple-400 font-medium">Random Forest</span>, and{" "}
            <span className="text-pink-400 font-medium">SVM</span> to assess cardiovascular risk factors with exceptional precision.
          </motion.p>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="flex items-center justify-center gap-8 mt-8 flex-wrap"
          >
            {[
              { label: "Accuracy", value: "85.2%" },
              { label: "Precision", value: "0.89" },
              { label: "AUC Score", value: "0.91" }
            ].map((stat, idx) => (
              <div key={idx} className="stats-card group">
                <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                <div className="text-xs text-slate-500 uppercase tracking-wider">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Form Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-7"
          >
            <div className="glass-card rounded-2xl p-8 hover-lift transition-all duration-300">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-blue-500/30">
                    <Heart className="w-5 h-5 text-blue-400" />
                  </div>
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">Patient Parameters</h2>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-800/50 px-3 py-1.5 rounded-full border border-slate-700/50">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="font-mono font-medium">v1.0.0</span>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <NumberInput
                  label="Age"
                  value={formData.age}
                  onChange={(v) => handleInputChange("age", v)}
                  min={1} max={120}
                  suffix="years"
                />

                <SelectInput
                  label="Sex"
                  value={formData.sex}
                  onChange={(v) => handleInputChange("sex", v)}
                  options={[
                    { value: 1, label: "Male" },
                    { value: 0, label: "Female" }
                  ]}
                />

                <SelectInput
                  label="Chest Pain Type"
                  value={formData.cp}
                  onChange={(v) => handleInputChange("cp", v)}
                  options={[
                    { value: 0, label: "Typical Angina" },
                    { value: 1, label: "Atypical Angina" },
                    { value: 2, label: "Non-anginal Pain" },
                    { value: 3, label: "Asymptomatic" }
                  ]}
                />

                <NumberInput
                  label="Resting BP"
                  value={formData.trestbps}
                  onChange={(v) => handleInputChange("trestbps", v)}
                  min={80} max={200}
                  suffix="mm Hg"
                />

                <NumberInput
                  label="Cholesterol"
                  value={formData.chol}
                  onChange={(v) => handleInputChange("chol", v)}
                  min={100} max={600}
                  suffix="mg/dl"
                />

                <SelectInput
                  label="Fasting BS > 120"
                  value={formData.fbs}
                  onChange={(v) => handleInputChange("fbs", v)}
                  options={[
                    { value: 1, label: "True" },
                    { value: 0, label: "False" }
                  ]}
                />

                <SelectInput
                  label="Resting ECG"
                  value={formData.restecg}
                  onChange={(v) => handleInputChange("restecg", v)}
                  options={[
                    { value: 0, label: "Normal" },
                    { value: 1, label: "ST-T Wave Abnormality" },
                    { value: 2, label: "Left Ventricular Hypertrophy" }
                  ]}
                />

                <NumberInput
                  label="Max Heart Rate"
                  value={formData.thalach}
                  onChange={(v) => handleInputChange("thalach", v)}
                  min={60} max={220}
                  suffix="bpm"
                />

                <SelectInput
                  label="Exercise Angina"
                  value={formData.exang}
                  onChange={(v) => handleInputChange("exang", v)}
                  options={[
                    { value: 1, label: "Yes" },
                    { value: 0, label: "No" }
                  ]}
                />

                <NumberInput
                  label="Oldpeak (ST Depression)"
                  value={formData.oldpeak}
                  onChange={(v) => handleInputChange("oldpeak", v)}
                  min={0} max={10} step={0.1}
                />

                <SelectInput
                  label="Slope"
                  value={formData.slope}
                  onChange={(v) => handleInputChange("slope", v)}
                  options={[
                    { value: 0, label: "Upsloping" },
                    { value: 1, label: "Flat" },
                    { value: 2, label: "Downsloping" }
                  ]}
                />

                <SelectInput
                  label="Major Vessels (CA)"
                  value={formData.ca}
                  onChange={(v) => handleInputChange("ca", v)}
                  options={[
                    { value: 0, label: "0" },
                    { value: 1, label: "1" },
                    { value: 2, label: "2" },
                    { value: 3, label: "3" },
                    { value: 4, label: "4" }
                  ]}
                />

                <div className="md:col-span-2">
                  <SelectInput
                    label="Thalassemia"
                    value={formData.thal}
                    onChange={(v) => handleInputChange("thal", v)}
                    options={[
                      { value: 1, label: "Fixed Defect" },
                      { value: 2, label: "Normal" },
                      { value: 3, label: "Reversible Defect" }
                    ]}
                  />
                </div>

                <div className="md:col-span-2 mt-6">
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full relative group overflow-hidden bg-gradient-to-r from-blue-600 via-blue-500 to-purple-600 hover:from-blue-500 hover:via-purple-500 hover:to-pink-500 text-white font-bold py-4 px-6 rounded-xl transition-all duration-500 shadow-[0_0_30px_rgba(59,130,246,0.4)] hover:shadow-[0_0_50px_rgba(147,51,234,0.6)] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none transform hover:scale-[1.02] active:scale-[0.98]"
                  >
                    {/* Shimmer effect */}
                    <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />

                    {/* Button content */}
                    <span className="relative flex items-center justify-center gap-3 text-base">
                      {loading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          <span className="font-semibold">Analyzing Risk Factors...</span>
                        </>
                      ) : (
                        <>
                          <span className="font-semibold">Generate Risk Assessment</span>
                          <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        </>
                      )}
                    </span>
                  </button>
                </div>
              </form>
            </div>
          </motion.div>

          {/* Results Section */}
          <motion.div
            className="lg:col-span-5"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <div className="sticky top-8 space-y-6">
              {/* Instructions Card (Only show if no result) */}
              {!result && !loading && (
                <div className="glass-card rounded-2xl p-8 border-l-4 border-blue-500">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-blue-500/10 rounded-lg">
                      <Info className="w-6 h-6 text-blue-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold mb-2">How it works</h3>
                      <p className="text-slate-400 leading-relaxed">
                        Enter the patient's clinical parameters in the form. Our AI model utilizing an ensemble of Logistic Regression, Random Forest, and SVM will analyze the data to predict cardiovascular risk probability in real-time.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Error State */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="glass-card rounded-2xl p-6 border-l-4 border-red-500 bg-red-500/5"
                  >
                    <div className="flex items-center gap-3 text-red-400">
                      <AlertCircle className="w-6 h-6" />
                      <span className="font-medium">{error}</span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Result State */}
              <AnimatePresence>
                {result && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className={clsx(
                      "glass-card rounded-2xl overflow-hidden border-t-4",
                      result.risk_level === "High" ? "border-red-500 shadow-[0_0_50px_rgba(239,68,68,0.2)]" :
                        result.risk_level === "Medium" ? "border-yellow-500 shadow-[0_0_50px_rgba(245,158,11,0.2)]" :
                          "border-green-500 shadow-[0_0_50px_rgba(16,185,129,0.2)]"
                    )}
                  >
                    <div className="p-8">
                      <div className="text-center mb-8">
                        <p className="text-sm uppercase tracking-wider text-slate-400 font-semibold mb-2">Risk Assessment</p>
                        <h3 className={clsx(
                          "text-4xl font-bold mb-2",
                          result.risk_level === "High" ? "text-red-400" :
                            result.risk_level === "Medium" ? "text-yellow-400" :
                              "text-green-400"
                        )}>
                          {result.risk_level} Risk
                        </h3>
                        <div className="text-slate-500 text-sm">
                          Confidence: {(result.confidence * 100).toFixed(1)}%
                        </div>
                      </div>

                      {/* Probability Meter */}
                      <div className="mb-8">
                        <div className="flex justify-between text-xs text-slate-400 mb-2 font-medium">
                          <span>Low</span>
                          <span>High</span>
                        </div>
                        <div className="h-4 bg-slate-800 rounded-full overflow-hidden relative">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${result.probability * 100}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className={clsx(
                              "h-full rounded-full relative",
                              result.risk_level === "High" ? "bg-gradient-to-r from-red-600 to-red-400" :
                                result.risk_level === "Medium" ? "bg-gradient-to-r from-yellow-600 to-yellow-400" :
                                  "bg-gradient-to-r from-green-600 to-green-400"
                            )}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white/50 shadow-[0_0_10px_white]" />
                          </motion.div>
                        </div>
                        <div className="mt-2 text-center text-2xl font-bold font-mono">
                          {(result.probability * 100).toFixed(1)}%
                        </div>
                        <p className="text-center text-slate-500 text-xs mt-1">Probability of Cardiovascular Disease</p>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                          <p className="text-xs text-slate-400 mb-1">Prediction</p>
                          <p className="text-lg font-semibold">
                            {result.prediction === 1 ? "Positive" : "Negative"}
                          </p>
                        </div>
                        <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                          <p className="text-xs text-slate-400 mb-1">Timestamp</p>
                          <p className="text-sm font-mono text-slate-300 truncate">
                            {new Date(result.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className={clsx(
                      "p-4 text-center text-sm font-medium",
                      result.risk_level === "High" ? "bg-red-500/10 text-red-300" :
                        result.risk_level === "Medium" ? "bg-yellow-500/10 text-yellow-300" :
                          "bg-green-500/10 text-green-300"
                    )}>
                      {result.risk_level === "High" ? "Immediate medical consultation recommended." :
                        result.risk_level === "Medium" ? "Lifestyle changes and monitoring advised." :
                          "Regular health checkups recommended."}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </div>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-24 pt-12 border-t border-slate-800/50"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            {/* Branding */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20">
                  <Heart className="w-5 h-5 text-blue-400" />
                </div>
                <span className="font-bold text-lg gradient-text">CardioPredict</span>
              </div>
              <p className="text-slate-500 text-sm leading-relaxed">
                Advanced AI-powered cardiovascular risk assessment using state-of-the-art machine learning ensemble methods.
              </p>
            </div>

            {/* Links */}
            <div>
              <h3 className="font-semibold text-slate-300 mb-3 text-sm uppercase tracking-wider">Resources</h3>
              <ul className="space-y-2">
                {['Documentation', 'API Reference', 'Research Paper', 'About the Model'].map((item) => (
                  <li key={item}>
                    <a href="#" className="text-slate-500 hover:text-blue-400 text-sm transition-colors">
                      {item}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Info */}
            <div>
              <h3 className="font-semibold text-slate-300 mb-3 text-sm uppercase tracking-wider">Project Info</h3>
              <div className="space-y-2 text-sm text-slate-500">
                <p>Built with PyTorch & FastAPI</p>
                <p>Model Accuracy: <span className="text-blue-400 font-semibold">85.2%</span></p>
                <p>Last Updated: January 2026</p>
              </div>
            </div>
          </div>

          {/* Bottom */}
          <div className="pt-6 border-t border-slate-800/50 flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-slate-600 text-sm">
              © 2026 Cardiovascular Risk Prediction. Built for educational purposes.
            </p>
            <div className="flex items-center gap-4 text-sm text-slate-600">
              <a href="#" className="hover:text-blue-400 transition-colors">Privacy</a>
              <span>•</span>
              <a href="#" className="hover:text-blue-400 transition-colors">Terms</a>
              <span>•</span>
              <a href="https://github.com/shlokkvaishnav/cardiovascular-risk" className="hover:text-blue-400 transition-colors">GitHub</a>
            </div>
          </div>
        </motion.footer>
      </div>
    </main>
  );
}

// Components
interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
}

function NumberInput({ label, value, onChange, min, max, step = 1, suffix }: NumberInputProps) {
  return (
    <div className="space-y-2.5 group">
      <label className="text-sm font-semibold text-slate-300 flex justify-between items-baseline">
        <span className="flex items-center gap-2">
          {label}
          <span className="w-1 h-1 rounded-full bg-blue-400/50" />
        </span>
        {suffix && <span className="text-slate-500 text-xs font-mono">{suffix}</span>}
      </label>
      <div className="relative">
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          min={min}
          max={max}
          step={step}
          className="w-full bg-slate-900/50 border border-slate-700/50 rounded-xl px-4 py-3.5 text-slate-100 font-medium focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 hover:border-slate-600 transition-all outline-none group-hover:bg-slate-900/70"
        />
        <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
      </div>
    </div>
  );
}

interface SelectInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  options: { value: number; label: string }[];
}

function SelectInput({ label, value, onChange, options }: SelectInputProps) {
  return (
    <div className="space-y-2.5">
      <label className="text-sm font-semibold text-slate-300 flex items-center gap-2">
        {label}
        <span className="w-1 h-1 rounded-full bg-blue-400/50" />
      </label>
      <div className="grid grid-cols-2 gap-2.5">
        {options.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={clsx(
              "relative px-4 py-3 text-sm font-medium rounded-xl border transition-all duration-300 overflow-hidden group",
              value === opt.value
                ? "bg-gradient-to-r from-blue-600 to-purple-600 border-blue-500/50 text-white shadow-[0_0_20px_rgba(59,130,246,0.4)] scale-[1.02]"
                : "bg-slate-900/30 border-slate-700/50 text-slate-400 hover:bg-slate-800/50 hover:border-slate-600 hover:text-slate-300"
            )}
          >
            {value === opt.value && (
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
            )}
            <span className="relative">{opt.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
