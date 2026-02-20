"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BadgeCheck,
  CircleHelp,
  HeartPulse,
  LoaderCircle,
  ShieldCheck,
  UserRound,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  type ActivityLevel,
  type AssessmentData,
  createEmptyAssessment,
  scoreRisk,
} from "../lib/risk";

const STORAGE_KEY = "cardio_assessment_v1";

type StepConfig = {
  title: string;
  subtitle: string;
};

const steps: StepConfig[] = [
  { title: "Welcome", subtitle: "Know your heart in under 3 minutes." },
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

export function AssessmentWizard() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [errors, setErrors] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisTipIndex, setAnalysisTipIndex] = useState(0);
  const [data, setData] = useState<AssessmentData>(createEmptyAssessment());

  const analysisTips = [
    "Cardiovascular risk estimates are probabilities, not diagnoses.",
    "Blood pressure control can significantly shift long-term outcomes.",
    "Smoking cessation is one of the fastest ways to reduce risk trajectory.",
  ];

  const completion = useMemo(
    () =>
      Math.round(
        ((Number(data.age !== null) +
          Number(data.sex !== null) +
          Number(data.smoking !== null) +
          Number(data.activityLevel !== null) +
          Number(data.diabetes !== null) +
          Number(data.familyHistory !== null) +
          Number(data.onBpMedication !== null)) /
          7) *
          100,
      ),
    [data],
  );

  const update = <K extends keyof AssessmentData>(key: K, value: AssessmentData[K]) => {
    setData((prev) => ({ ...prev, [key]: value }));
  };

  const stepValid = () => {
    if (step === 0) return true;
    if (step === 1) return numberValidity(data.age, 20, 79) && data.sex !== null;
    if (step === 2) {
      const bpValid = data.unknownVitals.systolicBp || numberValidity(data.systolicBp, 80, 240);
      const totalCholValid =
        data.unknownVitals.totalCholesterol || numberValidity(data.totalCholesterol, 100, 400);
      const hdlValid = data.unknownVitals.hdlCholesterol || numberValidity(data.hdlCholesterol, 20, 130);
      return bpValid && totalCholValid && hdlValid;
    }
    if (step === 3) return data.smoking !== null && data.activityLevel !== null;
    if (step === 4) return data.diabetes !== null && data.familyHistory !== null && data.onBpMedication !== null;
    if (step === 5) return true;
    return false;
  };

  const moveNext = () => {
    if (!stepValid()) {
      setErrors(["Please complete required fields before continuing."]);
      return;
    }
    setErrors([]);
    setStep((prev) => Math.min(prev + 1, steps.length - 1));
  };

  const moveBack = () => {
    setErrors([]);
    setStep((prev) => Math.max(0, prev - 1));
  };

  const startAnalysis = () => {
    if (!stepValid()) {
      setErrors(["Please review the entries before analysis."]);
      return;
    }
    setStep(6);
    setIsAnalyzing(true);
    const result = scoreRisk(data);
    if (typeof window !== "undefined") {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ data, result }));
    }

    const tipTimer = window.setInterval(() => {
      setAnalysisTipIndex((prev) => (prev + 1) % analysisTips.length);
    }, 1100);

    window.setTimeout(() => {
      window.clearInterval(tipTimer);
      router.push("/results");
    }, 3200);
  };

  const renderStep = () => {
    if (step === 0) {
      return (
        <div className="step-block">
          <h2 className="type-h2">Know your heart in 3 minutes.</h2>
          <p className="lead">
            This guided assessment uses established cardiovascular risk factors and keeps data in session only.
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
          <FloatingInput
            id="age"
            label="Age"
            value={data.age}
            min={20}
            max={79}
            unit="years"
            onChange={(value) => update("age", value)}
          />
          <OptionGrid
            label="Sex at birth"
            value={data.sex}
            options={[
              { value: "female", title: "Female", description: "Reference female risk model" },
              { value: "male", title: "Male", description: "Reference male risk model" },
            ]}
            onChange={(value) => update("sex", value)}
          />
        </div>
      );
    }

    if (step === 2) {
      const showTooltip = "Clinical term explained in plain language.";
      return (
        <div className="step-block">
          <h2 className="type-h2">Vitals & Labs</h2>
          <p className="lead">Add values if known, or select the I do not know option.</p>

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
                  unknownVitals: {
                    ...prev.unknownVitals,
                    systolicBp: !prev.unknownVitals.systolicBp,
                  },
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
            value={data.systolicBp}
            min={80}
            max={240}
            unit="mmHg"
            disabled={data.unknownVitals.systolicBp}
            helper="Top blood pressure number when your heart beats."
            onChange={(value) => update("systolicBp", value)}
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
                  unknownVitals: {
                    ...prev.unknownVitals,
                    totalCholesterol: !prev.unknownVitals.totalCholesterol,
                  },
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
            value={data.totalCholesterol}
            min={100}
            max={400}
            unit="mg/dL"
            disabled={data.unknownVitals.totalCholesterol}
            helper="Total circulating cholesterol in your blood sample."
            onChange={(value) => update("totalCholesterol", value)}
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
                  unknownVitals: {
                    ...prev.unknownVitals,
                    hdlCholesterol: !prev.unknownVitals.hdlCholesterol,
                  },
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
            value={data.hdlCholesterol}
            min={20}
            max={130}
            unit="mg/dL"
            disabled={data.unknownVitals.hdlCholesterol}
            helper="Protective cholesterol that helps remove LDL."
            onChange={(value) => update("hdlCholesterol", value)}
          />
        </div>
      );
    }

    if (step === 3) {
      return (
        <div className="step-block">
          <h2 className="type-h2">Lifestyle</h2>
          <p className="lead">Lifestyle risks are modifiable and high-impact.</p>
          <OptionGrid
            label="Do you currently smoke?"
            value={data.smoking === null ? null : data.smoking ? "yes" : "no"}
            options={[
              { value: "no", title: "No", description: "No current smoking" },
              { value: "yes", title: "Yes", description: "Current smoking exposure" },
            ]}
            onChange={(value) => update("smoking", value === "yes")}
          />
          <OptionGrid<ActivityLevel>
            label="Weekly activity level"
            value={data.activityLevel}
            options={[
              { value: "low", title: "Low", description: "< 60 minutes active/week" },
              { value: "moderate", title: "Moderate", description: "60-149 minutes/week" },
              { value: "high", title: "High", description: "150+ minutes/week" },
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
          <OptionGrid
            label="Family history of early heart disease"
            value={data.familyHistory === null ? null : data.familyHistory ? "yes" : "no"}
            options={[
              { value: "no", title: "No", description: "No early family cardiac events" },
              { value: "yes", title: "Yes", description: "Parent or sibling with early CVD event" },
            ]}
            onChange={(value) => update("familyHistory", value === "yes")}
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
      const rows: Array<{ label: string; value: string }> = [
        { label: "Age", value: data.age ? `${data.age} years` : "Not set" },
        { label: "Sex", value: data.sex ?? "Not set" },
        {
          label: "Systolic BP",
          value: data.unknownVitals.systolicBp ? "I do not know" : data.systolicBp ? `${data.systolicBp} mmHg` : "Not set",
        },
        {
          label: "Total Cholesterol",
          value: data.unknownVitals.totalCholesterol
            ? "I do not know"
            : data.totalCholesterol
              ? `${data.totalCholesterol} mg/dL`
              : "Not set",
        },
        {
          label: "HDL Cholesterol",
          value: data.unknownVitals.hdlCholesterol
            ? "I do not know"
            : data.hdlCholesterol
              ? `${data.hdlCholesterol} mg/dL`
              : "Not set",
        },
        { label: "Smoking", value: data.smoking === null ? "Not set" : data.smoking ? "Yes" : "No" },
        { label: "Activity", value: data.activityLevel ?? "Not set" },
        { label: "Diabetes", value: data.diabetes === null ? "Not set" : data.diabetes ? "Yes" : "No" },
        {
          label: "Family History",
          value: data.familyHistory === null ? "Not set" : data.familyHistory ? "Yes" : "No",
        },
      ];

      return (
        <div className="step-block">
          <h2 className="type-h2">Review & Confirm</h2>
          <p className="lead">Check details before generating risk analysis.</p>
          <div className="review-list">
            {rows.map((row) => (
              <div className="review-row" key={row.label}>
                <span>{row.label}</span>
                <strong>{row.value}</strong>
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

  return (
    <section className="section">
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
              <div
                key={item.title}
                className={`step-item ${index === step ? "active" : ""} ${index < step ? "done" : ""}`}
              >
                <span className="step-index">{index + 1}</span>
                <div>
                  <div className="step-title">{item.title}</div>
                  <div className="step-sub">{item.subtitle}</div>
                </div>
              </div>
            ))}
          </div>
        </aside>

        <motion.div
          className="card-panel wizard-main"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div className="wizard-head">
            <div className="head-left">
              {step <= 1 ? <UserRound size={18} /> : <Activity size={18} />}
              <span>{steps[step]?.title}</span>
            </div>
            <span className="mono">Step {step + 1}/7</span>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
            >
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
