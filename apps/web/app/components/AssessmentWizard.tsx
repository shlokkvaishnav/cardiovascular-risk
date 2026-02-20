"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BadgeCheck,
  Ban,
  CircleHelp,
  Cigarette,
  HeartPulse,
  LoaderCircle,
  ShieldCheck,
  Timer,
  UserRound,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { appendAssessmentHistory } from "../hooks/useAssessmentHistory";
import {
  type ActivityLevel,
  type AssessmentData,
  type FamilyHistoryType,
  type SmokingStatus,
  createEmptyAssessment,
  scoreRisk,
} from "../lib/risk";

const STORAGE_KEY = "cardio_assessment_v1";
const DRAFT_KEY = "cardio_assessment_draft_v1";

type StepConfig = {
  title: string;
  subtitle: string;
};

const steps: StepConfig[] = [
  { title: "Welcome", subtitle: "Know your heart in under 4 minutes." },
  { title: "Demographics", subtitle: "Age and sex help anchor baseline risk." },
  { title: "Vitals", subtitle: "Blood pressure and lipid context." },
  { title: "Lifestyle", subtitle: "Behavioral factors with major impact." },
  { title: "History", subtitle: "Medical history and treatment context." },
  { title: "Review", subtitle: "Confirm details before analysis." },
  { title: "Analysis", subtitle: "Generating your personalized risk profile." },
];

function numberValidity(value: number | null, min: number, max: number) {
  if (value === null) return false;
  return value >= min && value <= max;
}

function round(value: number, places = 1) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function FloatingInput(props: {
  label: string;
  id: string;
  value: number | null;
  unit?: string;
  min?: number;
  max?: number;
  disabled?: boolean;
  helper?: string;
  onChange: (value: number | null) => void;
}) {
  const { label, id, value, unit, min, max, disabled, helper, onChange } = props;
  const isValid = value !== null && (min === undefined || max === undefined || numberValidity(value, min, max));

  return (
    <div className="field">
      <div className="field-head">
        <label htmlFor={id} className="field-label">
          {label}
        </label>
        {min !== undefined && max !== undefined && (
          <span className="field-range">
            {min} - {max}
          </span>
        )}
      </div>
      <div className={`floating ${disabled ? "disabled" : ""}`}>
        <input
          id={id}
          className="floating-input"
          type="number"
          value={value ?? ""}
          min={min}
          max={max}
          disabled={disabled}
          aria-label={label}
          onChange={(event) => {
            const next = event.target.value;
            if (!next) {
              onChange(null);
              return;
            }
            onChange(Number(next));
          }}
        />
        <span className={`floating-text ${value !== null ? "raised" : ""}`}>{label}</span>
        {unit ? <span className="floating-suffix">{unit}</span> : null}
        {isValid ? <BadgeCheck className="floating-check" size={16} aria-hidden="true" /> : null}
      </div>
      {helper ? <p className="helper-inline">{helper}</p> : null}
    </div>
  );
}

function OptionGrid<T extends string>(props: {
  label: string;
  value: T | null;
  options: { value: T; title: string; description: string }[];
  onChange: (next: T) => void;
}) {
  return (
    <div className="field">
      <div className="field-head">
        <span className="field-label">{props.label}</span>
      </div>
      <div className="option-grid" role="radiogroup" aria-label={props.label}>
        {props.options.map((option) => {
          const active = props.value === option.value;
          return (
            <button
              type="button"
              key={option.value}
              className={`option-card ${active ? "active" : ""}`}
              role="radio"
              aria-checked={active}
              onClick={() => props.onChange(option.value)}
            >
              <span className="option-title">{option.title}</span>
              <span className="option-desc">{option.description}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function SmokingCards(props: {
  value: SmokingStatus | null;
  onChange: (value: SmokingStatus) => void;
}) {
  const cards: Array<{ value: SmokingStatus; title: string; desc: string; icon: JSX.Element }> = [
    { value: "never", title: "Never", desc: "No smoking history", icon: <Ban size={16} /> },
    { value: "former", title: "Former", desc: "Stopped smoking", icon: <Cigarette size={16} /> },
    { value: "current", title: "Current", desc: "Current smoking exposure", icon: <Cigarette size={16} /> },
  ];
  return (
    <div className="field">
      <div className="field-head">
        <span className="field-label">Smoking status</span>
      </div>
      <div className="option-grid smoking-grid">
        {cards.map((card) => {
          const active = props.value === card.value;
          return (
            <button
              type="button"
              key={card.value}
              className={`option-card smoking-card ${active ? "active" : ""}`}
              onClick={() => props.onChange(card.value)}
            >
              <span className="smoking-icon">{card.icon}</span>
              <span className="option-title">{card.title}</span>
              <span className="option-desc">{card.desc}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function RangeIndicator(props: { min: number; max: number; healthyMin: number; healthyMax: number; value: number | null }) {
  const { min, max, healthyMin, healthyMax, value } = props;
  const position = value === null ? 0 : ((value - min) / (max - min)) * 100;
  const start = ((healthyMin - min) / (max - min)) * 100;
  const width = ((healthyMax - healthyMin) / (max - min)) * 100;
  return (
    <div className="range-indicator">
      <div className="range-track">
        <span className="range-good" style={{ left: `${start}%`, width: `${width}%` }} />
        {value !== null ? <span className="range-dot" style={{ left: `${Math.max(0, Math.min(100, position))}%` }} /> : null}
      </div>
      <div className="range-labels">
        <span>{min}</span>
        <span>Healthy zone</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export function AssessmentWizard() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [errors, setErrors] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisTipIndex, setAnalysisTipIndex] = useState(0);
  const [bpUnit, setBpUnit] = useState<"mmHg" | "kPa">("mmHg");
  const [cholUnit, setCholUnit] = useState<"mg/dL" | "mmol/L">("mg/dL");
  const [data, setData] = useState<AssessmentData>(createEmptyAssessment());

  const analysisTips = [
    "Cardiovascular risk estimates are probabilities, not diagnoses.",
    "Lowering systolic BP by 10 mmHg can meaningfully improve outcomes.",
    "Smoking cessation is one of the fastest ways to reduce risk trajectory.",
    "Regular physical activity improves both lipid profile and vascular health.",
  ];
  const analysisTipCount = analysisTips.length;

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const raw = window.sessionStorage.getItem(DRAFT_KEY);
      if (!raw) return;
      try {
        const parsed = JSON.parse(raw) as { step: number; data: AssessmentData };
        if (parsed?.data) {
          setData(parsed.data);
          setStep(Math.max(0, Math.min(5, parsed.step ?? 0)));
        }
      } catch {
        // Ignore invalid drafts
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    window.sessionStorage.setItem(DRAFT_KEY, JSON.stringify({ step, data }));
  }, [data, step]);

  const completion = useMemo(
    () =>
      Math.round(
        ((Number(data.age !== null) +
          Number(data.sex !== null) +
          Number(data.smokingStatus !== null) +
          Number(data.activityLevel !== null) +
          Number(data.diabetes !== null) +
          Number(data.familyHistoryType !== null) +
          Number(data.onBpMedication !== null)) /
          7) *
          100,
      ),
    [data],
  );

  const update = <K extends keyof AssessmentData>(key: K, value: AssessmentData[K]) => {
    setData((prev) => ({ ...prev, [key]: value }));
  };

  const toBpDisplay = (value: number | null) => {
    if (value === null) return null;
    return bpUnit === "mmHg" ? value : round(value / 7.50062, 1);
  };
  const fromBpDisplay = (value: number | null) => {
    if (value === null) return null;
    return bpUnit === "mmHg" ? value : round(value * 7.50062);
  };
  const toCholDisplay = (value: number | null) => {
    if (value === null) return null;
    return cholUnit === "mg/dL" ? value : round(value / 38.67, 2);
  };
  const fromCholDisplay = (value: number | null) => {
    if (value === null) return null;
    return cholUnit === "mg/dL" ? value : round(value * 38.67);
  };

  const stepValid = useCallback(() => {
    if (step === 0) return true;
    if (step === 1) return numberValidity(data.age, 20, 79) && data.sex !== null;
    if (step === 2) {
      const bpValid = data.unknownVitals.systolicBp || numberValidity(data.systolicBp, 80, 240);
      const totalCholValid = data.unknownVitals.totalCholesterol || numberValidity(data.totalCholesterol, 100, 400);
      const hdlValid = data.unknownVitals.hdlCholesterol || numberValidity(data.hdlCholesterol, 20, 130);
      return bpValid && totalCholValid && hdlValid;
    }
    if (step === 3) return data.smokingStatus !== null && data.activityLevel !== null;
    if (step === 4) return data.diabetes !== null && data.familyHistoryType !== null && data.onBpMedication !== null;
    if (step === 5) return true;
    return false;
  }, [data, step]);

  const moveNext = useCallback(() => {
    if (!stepValid()) {
      setErrors(["Please complete required fields before continuing."]);
      return;
    }
    setErrors([]);
    setStep((prev) => Math.min(prev + 1, steps.length - 1));
  }, [stepValid]);

  const moveBack = useCallback(() => {
    setErrors([]);
    setStep((prev) => Math.max(0, prev - 1));
  }, []);

  const startAnalysis = useCallback(() => {
    if (!stepValid()) {
      setErrors(["Please review the entries before analysis."]);
      return;
    }
    setStep(6);
    setIsAnalyzing(true);
    const normalized: AssessmentData = {
      ...data,
      smoking: data.smokingStatus === "current",
      familyHistory: data.familyHistoryType !== null && data.familyHistoryType !== "none",
    };
    const result = scoreRisk(normalized);
    if (typeof window !== "undefined") {
      window.sessionStorage.removeItem(DRAFT_KEY);
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ data: normalized, result }));
      appendAssessmentHistory({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        timestamp: new Date().toISOString(),
        score: result.score,
        category: result.category,
        age: normalized.age,
        systolicBp: normalized.systolicBp,
        totalCholesterol: normalized.totalCholesterol,
        hdlCholesterol: normalized.hdlCholesterol,
        smoking: normalized.smoking,
        activityLevel: normalized.activityLevel,
      });
    }

    const tipTimer = window.setInterval(() => {
      setAnalysisTipIndex((prev) => (prev + 1) % analysisTipCount);
    }, 900);

    window.setTimeout(() => {
      window.clearInterval(tipTimer);
      router.push("/results");
    }, 3200);
  }, [analysisTipCount, data, router, stepValid]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName.toLowerCase();
      const typing = tag === "input" || tag === "textarea" || tag === "select" || (target?.isContentEditable ?? false);

      if (event.key === "Enter" && !typing) {
        event.preventDefault();
        if (step === 0) moveNext();
        else if (step >= 1 && step < 5) moveNext();
        else if (step === 5) startAnalysis();
      }

      if (event.key === "Backspace" && !typing && step > 0 && step < 6) {
        event.preventDefault();
        moveBack();
      }

      if (!/^[1-9]$/.test(event.key)) return;
      const index = Number(event.key);
      if (step === 1) {
        if (index === 1) update("sex", "female");
        if (index === 2) update("sex", "male");
      }
      if (step === 3) {
        if (index === 1) {
          update("smokingStatus", "never");
          update("smoking", false);
        }
        if (index === 2) {
          update("smokingStatus", "former");
          update("smoking", false);
        }
        if (index === 3) {
          update("smokingStatus", "current");
          update("smoking", true);
        }
        if (index === 4) update("activityLevel", "low");
        if (index === 5) update("activityLevel", "moderate");
        if (index === 6) update("activityLevel", "high");
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, data, moveNext, startAnalysis, moveBack]);

  const renderStep = () => {
    if (step === 0) {
      return (
        <div className="step-block">
          <div className="welcome-hero">
            <motion.div
              className="welcome-heart"
              initial={{ scale: 0.95, opacity: 0.7 }}
              animate={{ scale: [1, 1.08, 1], opacity: [0.8, 1, 0.8] }}
              transition={{ duration: 1.8, repeat: Number.POSITIVE_INFINITY }}
            >
              <HeartPulse size={42} />
            </motion.div>
            <span className="chip">~ 4 minutes</span>
          </div>
          <h2 className="type-h2">Know your heart in 4 minutes.</h2>
          <p className="lead">
            This guided assessment uses established cardiovascular risk factors and keeps data in your current browser
            session only.
          </p>
          <div className="alert-row">
            <div className="trust-pill">
              <ShieldCheck size={16} />
              No data stored on our servers
            </div>
            <div className="trust-pill">
              <HeartPulse size={16} />
              Based on Framingham / Pooled Cohort principles
            </div>
          </div>
        </div>
      );
    }

    if (step === 1) {
      return (
        <div className="step-block">
          <h2 className="type-h2">Demographics</h2>
          <p className="lead">Baseline profile for age and sex assignment.</p>
          <FloatingInput id="age" label="Age" value={data.age} min={20} max={79} unit="years" onChange={(value) => update("age", value)} />
          <OptionGrid
            label="Sex at birth"
            value={data.sex}
            options={[
              { value: "female", title: "Female", description: "Reference female risk model (Shortcut: 1)" },
              { value: "male", title: "Male", description: "Reference male risk model (Shortcut: 2)" },
            ]}
            onChange={(value) => update("sex", value)}
          />
        </div>
      );
    }

    if (step === 2) {
      const showTooltip = "Clinical term explained in plain language.";
      const bpRange = bpUnit === "mmHg" ? { min: 80, max: 240 } : { min: 10.7, max: 32 };
      const cholRange = cholUnit === "mg/dL" ? { min: 100, max: 400 } : { min: 2.6, max: 10.35 };
      const hdlRange = cholUnit === "mg/dL" ? { min: 20, max: 130 } : { min: 0.52, max: 3.36 };
      return (
        <div className="step-block">
          <h2 className="type-h2">Vitals & Labs</h2>
          <p className="lead">Add values if known, or select I do not know.</p>

          <div className="unit-toggle-row">
            <button type="button" className={`chip ${bpUnit === "mmHg" ? "active" : ""}`} onClick={() => setBpUnit("mmHg")}>mmHg</button>
            <button type="button" className={`chip ${bpUnit === "kPa" ? "active" : ""}`} onClick={() => setBpUnit("kPa")}>kPa</button>
            <button type="button" className={`chip ${cholUnit === "mg/dL" ? "active" : ""}`} onClick={() => setCholUnit("mg/dL")}>mg/dL</button>
            <button type="button" className={`chip ${cholUnit === "mmol/L" ? "active" : ""}`} onClick={() => setCholUnit("mmol/L")}>mmol/L</button>
          </div>

          <div className="field-inline">
            <span className="field-label">
              Systolic BP <span title={showTooltip}><CircleHelp size={14} /></span>
            </span>
            <button
              type="button"
              className={`chip ${data.unknownVitals.systolicBp ? "active" : ""}`}
              onClick={() =>
                setData((prev) => ({
                  ...prev,
                  unknownVitals: { ...prev.unknownVitals, systolicBp: !prev.unknownVitals.systolicBp },
                  systolicBp: !prev.unknownVitals.systolicBp ? null : prev.systolicBp,
                }))
              }
            >
              I do not know
            </button>
          </div>
          <FloatingInput
            id="sbp"
            label="Systolic Blood Pressure"
            value={toBpDisplay(data.systolicBp)}
            min={bpRange.min}
            max={bpRange.max}
            unit={bpUnit}
            disabled={data.unknownVitals.systolicBp}
            helper="Top blood pressure number when your heart beats."
            onChange={(value) => update("systolicBp", fromBpDisplay(value))}
          />

          <div className="field-inline">
            <span className="field-label">
              Total Cholesterol <span title={showTooltip}><CircleHelp size={14} /></span>
            </span>
            <button
              type="button"
              className={`chip ${data.unknownVitals.totalCholesterol ? "active" : ""}`}
              onClick={() =>
                setData((prev) => ({
                  ...prev,
                  unknownVitals: { ...prev.unknownVitals, totalCholesterol: !prev.unknownVitals.totalCholesterol },
                  totalCholesterol: !prev.unknownVitals.totalCholesterol ? null : prev.totalCholesterol,
                }))
              }
            >
              I do not know
            </button>
          </div>
          <FloatingInput
            id="total-chol"
            label="Total Cholesterol"
            value={toCholDisplay(data.totalCholesterol)}
            min={cholRange.min}
            max={cholRange.max}
            unit={cholUnit}
            disabled={data.unknownVitals.totalCholesterol}
            helper="Total circulating cholesterol in your blood sample."
            onChange={(value) => update("totalCholesterol", fromCholDisplay(value))}
          />
          <RangeIndicator
            min={cholUnit === "mg/dL" ? 100 : 2.6}
            max={cholUnit === "mg/dL" ? 400 : 10.35}
            healthyMin={cholUnit === "mg/dL" ? 125 : 3.2}
            healthyMax={cholUnit === "mg/dL" ? 200 : 5.2}
            value={toCholDisplay(data.totalCholesterol)}
          />

          <div className="field-inline">
            <span className="field-label">
              HDL Cholesterol <span title={showTooltip}><CircleHelp size={14} /></span>
            </span>
            <button
              type="button"
              className={`chip ${data.unknownVitals.hdlCholesterol ? "active" : ""}`}
              onClick={() =>
                setData((prev) => ({
                  ...prev,
                  unknownVitals: { ...prev.unknownVitals, hdlCholesterol: !prev.unknownVitals.hdlCholesterol },
                  hdlCholesterol: !prev.unknownVitals.hdlCholesterol ? null : prev.hdlCholesterol,
                }))
              }
            >
              I do not know
            </button>
          </div>
          <FloatingInput
            id="hdl-chol"
            label="HDL Cholesterol"
            value={toCholDisplay(data.hdlCholesterol)}
            min={hdlRange.min}
            max={hdlRange.max}
            unit={cholUnit}
            disabled={data.unknownVitals.hdlCholesterol}
            helper="Protective cholesterol that helps remove LDL."
            onChange={(value) => update("hdlCholesterol", fromCholDisplay(value))}
          />
          <RangeIndicator
            min={cholUnit === "mg/dL" ? 20 : 0.52}
            max={cholUnit === "mg/dL" ? 130 : 3.36}
            healthyMin={cholUnit === "mg/dL" ? 45 : 1.16}
            healthyMax={cholUnit === "mg/dL" ? 80 : 2.07}
            value={toCholDisplay(data.hdlCholesterol)}
          />
        </div>
      );
    }

    if (step === 3) {
      return (
        <div className="step-block">
          <h2 className="type-h2">Lifestyle</h2>
          <p className="lead">Lifestyle risks are modifiable and high-impact.</p>
          <SmokingCards
            value={data.smokingStatus}
            onChange={(value) => {
              update("smokingStatus", value);
              update("smoking", value === "current");
            }}
          />
          <OptionGrid<ActivityLevel>
            label="Weekly activity level"
            value={data.activityLevel}
            options={[
              { value: "low", title: "Low", description: "< 60 minutes active/week (Shortcut: 4)" },
              { value: "moderate", title: "Moderate", description: "60-149 minutes/week (Shortcut: 5)" },
              { value: "high", title: "High", description: "150+ minutes/week (Shortcut: 6)" },
            ]}
            onChange={(value) => update("activityLevel", value)}
          />
        </div>
      );
    }

    if (step === 4) {
      return (
        <div className="step-block">
          <h2 className="type-h2">Medical History</h2>
          <p className="lead">These factors refine risk profile and urgency.</p>
          <OptionGrid
            label="Diabetes history"
            value={data.diabetes === null ? null : data.diabetes ? "yes" : "no"}
            options={[
              { value: "no", title: "No", description: "No known diabetes" },
              { value: "yes", title: "Yes", description: "Type 1, Type 2, or pre-existing diagnosis" },
            ]}
            onChange={(value) => update("diabetes", value === "yes")}
          />
          <OptionGrid<FamilyHistoryType>
            label="First-degree family history of early CVD"
            value={data.familyHistoryType}
            options={[
              { value: "none", title: "None", description: "No first-degree early CVD history" },
              { value: "male_lt55", title: "Male <55", description: "Father/brother before age 55" },
              { value: "female_lt65", title: "Female <65", description: "Mother/sister before age 65" },
              { value: "both", title: "Both", description: "Both thresholds apply" },
            ]}
            onChange={(value) => {
              update("familyHistoryType", value);
              update("familyHistory", value !== "none");
            }}
          />
          <OptionGrid
            label="Currently on blood pressure medication"
            value={data.onBpMedication === null ? null : data.onBpMedication ? "yes" : "no"}
            options={[
              { value: "no", title: "No", description: "Not currently prescribed antihypertensives" },
              { value: "yes", title: "Yes", description: "Currently prescribed BP medication" },
            ]}
            onChange={(value) => update("onBpMedication", value === "yes")}
          />
        </div>
      );
    }

    if (step === 5) {
      const rows: Array<{ label: string; value: string; extreme?: boolean }> = [
        { label: "Age", value: data.age ? `${data.age} years` : "Not set" },
        { label: "Sex", value: data.sex ?? "Not set" },
        {
          label: "Systolic BP",
          value: data.unknownVitals.systolicBp ? "I do not know" : data.systolicBp ? `${data.systolicBp} mmHg` : "Not set",
          extreme: data.systolicBp !== null && data.systolicBp >= 180,
        },
        {
          label: "Total Cholesterol",
          value: data.unknownVitals.totalCholesterol ? "I do not know" : data.totalCholesterol ? `${data.totalCholesterol} mg/dL` : "Not set",
          extreme: data.totalCholesterol !== null && data.totalCholesterol >= 280,
        },
        {
          label: "HDL Cholesterol",
          value: data.unknownVitals.hdlCholesterol ? "I do not know" : data.hdlCholesterol ? `${data.hdlCholesterol} mg/dL` : "Not set",
          extreme: data.hdlCholesterol !== null && data.hdlCholesterol <= 30,
        },
        { label: "Smoking", value: data.smokingStatus ?? "Not set" },
        { label: "Activity", value: data.activityLevel ?? "Not set" },
        { label: "Diabetes", value: data.diabetes === null ? "Not set" : data.diabetes ? "Yes" : "No" },
        {
          label: "Family History",
          value: data.familyHistoryType ? data.familyHistoryType.replaceAll("_", " ") : "Not set",
        },
      ];

      return (
        <div className="step-block">
          <h2 className="type-h2">Review & Confirm</h2>
          <p className="lead">Check details before generating risk analysis.</p>
          <div className="review-list">
            {rows.map((row) => (
              <div className={`review-row ${row.extreme ? "review-row-warn" : ""}`} key={row.label}>
                <span>{row.label}</span>
                <strong>{row.value}</strong>
                {row.extreme ? <em className="review-note">Confirm this value is correct</em> : null}
              </div>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="step-block">
        <h2 className="type-h2">Running Risk Analysis</h2>
        <p className="lead">Building your personalized dashboard and recommendations.</p>
        <div className="analysis-card" aria-live="polite">
          <LoaderCircle className="spin" size={28} />
          <p>{analysisTips[analysisTipIndex]}</p>
        </div>
      </div>
    );
  };

  const canGoBack = step > 0 && step < 6;
  const isReviewStep = step === 5;
  const canContinue = step > 0 && step < 5;
  const progressWidth = `${((step + 1) / 7) * 100}%`;

  return (
    <section className="section">
      <div className="container wizard-progress-shell card-panel">
        <div className="wizard-progress-head">
          <span className="mono">Step {step + 1}/7</span>
          <span>{steps[step]?.title}</span>
        </div>
        <div className="wizard-progress-track">
          <span className="wizard-progress-fill" style={{ width: progressWidth }} />
        </div>
        <div className="wizard-stage-list">
          {steps.map((item, index) => (
            <span key={item.title} className={`wizard-stage ${index === step ? "active" : ""} ${index < step ? "done" : ""}`}>
              {item.title}
            </span>
          ))}
        </div>
      </div>

      <div className="container wizard-grid">
        <aside className="wizard-side card-panel">
          <div className="mini-badge">Guided wizard</div>
          <h1 className="type-h1">Cardiovascular Assessment</h1>
          <p className="lead">Conversational flow built to reduce cognitive load and improve input confidence.</p>
          <div className="meter">
            <div className="meter-fill" style={{ width: `${completion}%` }} />
          </div>
          <p className="meter-label mono">{completion}% profile complete</p>
          <div className="step-list">
            {steps.map((item, index) => (
              <div key={item.title} className={`step-item ${index === step ? "active" : ""} ${index < step ? "done" : ""}`}>
                <span className="step-index">{index + 1}</span>
                <div>
                  <div className="step-title">{item.title}</div>
                  <div className="step-sub">{item.subtitle}</div>
                </div>
              </div>
            ))}
          </div>
        </aside>

        <motion.div className="card-panel wizard-main" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
          <div className="wizard-head">
            <div className="head-left">
              {step <= 1 ? <UserRound size={18} /> : step === 0 ? <Timer size={18} /> : <Activity size={18} />}
              <span>{steps[step]?.title}</span>
            </div>
            <span className="mono">Shortcuts: Enter / Backspace / 1-6</span>
          </div>

          <AnimatePresence mode="wait">
            <motion.div key={step} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
              {renderStep()}
            </motion.div>
          </AnimatePresence>

          {errors.length > 0 && (
            <div className="error-box" aria-live="assertive">
              {errors.map((error) => (
                <p key={error}>{error}</p>
              ))}
            </div>
          )}

          <div className="action-bar">
            {canGoBack ? (
              <button type="button" className="btn btn-subtle" onClick={moveBack}>
                Back
              </button>
            ) : (
              <span />
            )}
            {canContinue && (
              <button type="button" className="btn btn-primary" onClick={moveNext}>
                Continue
              </button>
            )}
            {isReviewStep && (
              <button type="button" className="btn btn-primary" onClick={startAnalysis} disabled={isAnalyzing}>
                Analyze Risk
              </button>
            )}
            {step === 0 && (
              <button type="button" className="btn btn-primary" onClick={moveNext}>
                Start
              </button>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
