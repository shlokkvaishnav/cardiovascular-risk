"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, ArrowRight, Loader2, Check, AlertCircle } from "lucide-react";
import clsx from "clsx";

// Types
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

    // Artificial delay for slick UX feel
    await new Promise(r => setTimeout(r, 800));

    try {
      const API_URL = "/api"; 
      // In production, this would be an env var. 
      // Assuming proxy or direct call for now.
      
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
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-6 md:p-12 lg:p-24 max-w-[1600px] mx-auto">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row md:items-end justify-between mb-24 gap-8"
      >
        <div>
          <div className="flex items-center gap-2 mb-4">
             <span className="w-2 h-2 bg-zinc-500 rounded-full animate-pulse-subtle"></span>
             <span className="text-xs uppercase tracking-[0.2em] text-zinc-500 font-medium">Diagnostic Tool v2.0</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-light tracking-tight text-white mb-2">
            Cardio<span className="text-zinc-600">Risk</span>
          </h1>
          <p className="text-zinc-500 max-w-md font-light leading-relaxed">
            Advanced risk assessment using ensemble machine learning classifiers.
          </p>
        </div>

        <div className="flex gap-12 text-sm text-zinc-500 font-mono">
          <div className="flex flex-col">
            <span className="text-zinc-700 uppercase tracking-wider text-[10px] mb-1">Accuracy</span>
            <span className="text-white">85.2%</span>
          </div>
          <div className="flex flex-col">
            <span className="text-zinc-700 uppercase tracking-wider text-[10px] mb-1">Model</span>
            <span className="text-white">Ensemble</span>
          </div>
          <div className="flex flex-col">
            <span className="text-zinc-700 uppercase tracking-wider text-[10px] mb-1">Status</span>
            <span className="flex items-center gap-2 text-emerald-500">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.4)]"></span>
              Online
            </span>
          </div>
        </div>
      </motion.header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-24">
        
        {/* Input Section */}
        <div className="lg:col-span-7">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="mb-8"
          >
            <h2 className="text-xl font-light text-white mb-8 border-b border-zinc-900 pb-4">
              Patient Parameters
            </h2>
          </motion.div>

          <form onSubmit={handleSubmit} className="space-y-12">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-10">
              <InputGroup label="Age" suffix="yrs">
                <input
                  type="number"
                  value={formData.age}
                  onChange={(e) => handleInputChange("age", Number(e.target.value))}
                  className="glass-input w-full px-0 py-3 bg-transparent border-0 border-b border-zinc-800 focus:border-white transition-colors text-lg font-light placeholder-zinc-700"
                  placeholder="0"
                />
              </InputGroup>

              <SelectGroup label="Sex">
                 <div className="flex gap-4 pt-2">
                    {[
                      { value: 1, label: "Male" },
                      { value: 0, label: "Female" }
                    ].map(opt => (
                       <button
                        key={opt.value}
                        type="button"
                        onClick={() => handleInputChange("sex", opt.value)}
                        className={clsx(
                          "text-sm px-4 py-2 rounded-full border transition-all",
                          formData.sex === opt.value
                            ? "border-white text-white bg-white/5"
                            : "border-zinc-800 text-zinc-500 hover:border-zinc-700"
                        )}
                       >
                         {opt.label}
                       </button>
                    ))}
                 </div>
              </SelectGroup>

              <InputGroup label="Resting Blood Pressure" suffix="mm Hg">
                <input
                  type="number"
                  value={formData.trestbps}
                  onChange={(e) => handleInputChange("trestbps", Number(e.target.value))}
                  className="glass-input w-full px-0 py-3 bg-transparent border-0 border-b border-zinc-800 focus:border-white transition-colors text-lg font-light"
                />
              </InputGroup>

              <InputGroup label="Serum Cholesterol" suffix="mg/dl">
                 <input
                  type="number"
                  value={formData.chol}
                  onChange={(e) => handleInputChange("chol", Number(e.target.value))}
                  className="glass-input w-full px-0 py-3 bg-transparent border-0 border-b border-zinc-800 focus:border-white transition-colors text-lg font-light"
                />
              </InputGroup>

              <SelectGroup label="Chest Pain Type">
                <select 
                  value={formData.cp}
                  onChange={(e) => handleInputChange("cp", Number(e.target.value))}
                  className="w-full bg-transparent border-b border-zinc-800 text-white py-3 focus:outline-none focus:border-white transition-colors appearance-none cursor-pointer"
                >
                    <option value={0} className="bg-zinc-950">Typical Angina</option>
                    <option value={1} className="bg-zinc-950">Atypical Angina</option>
                    <option value={2} className="bg-zinc-950">Non-anginal Pain</option>
                    <option value={3} className="bg-zinc-950">Asymptomatic</option>
                </select>
              </SelectGroup>

              <SelectGroup label="Resting ECG">
                <select 
                  value={formData.restecg}
                  onChange={(e) => handleInputChange("restecg", Number(e.target.value))}
                  className="w-full bg-transparent border-b border-zinc-800 text-white py-3 focus:outline-none focus:border-white transition-colors appearance-none cursor-pointer"
                >
                    <option value={0} className="bg-zinc-950">Normal</option>
                    <option value={1} className="bg-zinc-950">ST-T Wave Abnormality</option>
                    <option value={2} className="bg-zinc-950">Left Ventricular Hypertrophy</option>
                </select>
              </SelectGroup>

              <InputGroup label="Max Heart Rate" suffix="bpm">
                 <input
                  type="number"
                  value={formData.thalach}
                  onChange={(e) => handleInputChange("thalach", Number(e.target.value))}
                  className="glass-input w-full px-0 py-3 bg-transparent border-0 border-b border-zinc-800 focus:border-white transition-colors text-lg font-light"
                />
              </InputGroup>

              <InputGroup label="ST Depression (Oldpeak)">
                 <input
                  type="number"
                  step="0.1"
                  value={formData.oldpeak}
                  onChange={(e) => handleInputChange("oldpeak", Number(e.target.value))}
                  className="glass-input w-full px-0 py-3 bg-transparent border-0 border-b border-zinc-800 focus:border-white transition-colors text-lg font-light"
                />
              </InputGroup>
              
              <div className="md:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-8">
                 <Toggle label="Fasting BS > 120" value={formData.fbs === 1} onChange={(v) => handleInputChange("fbs", v ? 1 : 0)} />
                 <Toggle label="Exercise Angina" value={formData.exang === 1} onChange={(v) => handleInputChange("exang", v ? 1 : 0)} />
              </div>

            </div>

             <div className="pt-8">
              <button
                type="submit"
                disabled={loading}
                className="group relative inline-flex items-center gap-3 px-8 py-4 bg-white text-black rounded-full font-medium text-sm transition-all hover:bg-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                 {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                 <span>{loading ? "Processing Analysis" : "Run Analysis"}</span>
                 <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
              </button>
             </div>
          </form>
        </div>

        {/* Results Section */}
        <div className="lg:col-span-5 relative">
           <div className="sticky top-12">
              <AnimatePresence mode="wait">
                {!result && !error && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="glass-panel p-8 rounded-2xl h-[400px] flex flex-col items-center justify-center text-center border-dashed border-zinc-800"
                  >
                     <Activity className="w-12 h-12 text-zinc-800 mb-6" strokeWidth={1} />
                     <h3 className="text-zinc-500 font-light text-lg mb-2">Ready for Analysis</h3>
                     <p className="text-zinc-700 text-sm max-w-xs">
                       Complete the patient parameters form to generate a real-time risk assessment.
                     </p>
                  </motion.div>
                )}
                
                {result && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="glass-panel rounded-3xl overflow-hidden border-sculpted"
                  >
                    <div className="p-10">
                      <div className="flex justify-between items-start mb-12">
                        <div>
                          <p className="text-zinc-500 uppercase tracking-widest text-xs mb-2">Risk Assessment</p>
                          <h2 className={clsx(
                            "text-4xl font-light",
                            result.risk_level === "High" ? "text-red-400" :
                            result.risk_level === "Medium" ? "text-amber-200" :
                            "text-emerald-200"
                          )}>
                             {result.risk_level} Risk
                          </h2>
                        </div>
                        <div className="text-right">
                           <p className="text-zinc-500 text-xs font-mono mb-1">PROBABILITY</p>
                           <p className="text-3xl font-mono text-white">{(result.probability * 100).toFixed(1)}%</p>
                        </div>
                      </div>

                      <div className="space-y-8">
                         <div>
                            <div className="flex justify-between text-xs text-zinc-600 mb-3 uppercase tracking-wider">
                               <span>Clinical Confidence</span>
                               <span>{(result.confidence * 100).toFixed(0)}%</span>
                            </div>
                            <div className="h-[2px] w-full bg-zinc-800">
                               <motion.div 
                                 initial={{ width: 0 }}
                                 animate={{ width: `${result.confidence * 100}%` }}
                                 className="h-full bg-white"
                               />
                            </div>
                         </div>

                         <div className="p-6 bg-white/5 rounded-2xl border border-white/5">
                            <h4 className="text-white text-sm font-medium mb-3">Analysis Summary</h4>
                            <p className="text-zinc-400 text-sm leading-relaxed">
                              Based on the provided parameters, the ensemble model indicates a <span className="text-white">{result.risk_level.toLowerCase()} probability</span> of cardiovascular disease. 
                              {result.risk_level === "High" 
                                ? " Immediate clinical consultation is strongly recommended." 
                                : " Routine monitoring is advised."}
                            </p>
                         </div>
                      </div>
                    </div>
                    
                    <div className="bg-zinc-900/50 p-6 border-t border-white/5 flex justify-between items-center text-xs text-zinc-500 font-mono">
                       <span>ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</span>
                       <span>{new Date().toLocaleDateString()}</span>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
           </div>
        </div>
      </div>
    </main>
  );
}

// Subcomponents

function InputGroup({ label, children, suffix }: { label: string, children: React.ReactNode, suffix?: string }) {
  return (
    <div className="group">
      <label className="block text-xs uppercase tracking-wider text-zinc-500 mb-2 group-focus-within:text-white transition-colors">
        {label}
      </label>
      <div className="relative">
        {children}
        {suffix && <span className="absolute right-0 top-3 text-zinc-600 text-sm pointer-events-none">{suffix}</span>}
      </div>
    </div>
  );
}

function SelectGroup({ label, children }: { label: string, children: React.ReactNode }) {
  return (
    <div className="group">
       <label className="block text-xs uppercase tracking-wider text-zinc-500 mb-2">
        {label}
      </label>
      {children}
    </div>
  );
}

function Toggle({ label, value, onChange }: { label: string, value: boolean, onChange: (v: boolean) => void }) {
   return (
      <div className="flex items-center justify-between p-4 rounded-xl border border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer" onClick={() => onChange(!value)}>
         <span className="text-sm text-zinc-400">{label}</span>
         <div className={clsx("w-5 h-5 rounded-full border flex items-center justify-center transition-colors", value ? "bg-white border-white" : "border-zinc-700 bg-transparent")}>
            {value && <Check className="w-3 h-3 text-black" />}
         </div>
      </div>
   )
}
