"use client";

import { useEffect, useMemo, useState } from "react";
import { runRiskCalc } from "../../lib/riskEngine";
import type { AssessmentData, RiskResult } from "../../lib/risk";

type Props = {
  baselineData: AssessmentData;
  baselineResult: RiskResult;
  onResultChange?: (result: RiskResult) => void;
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function RiskSimulator(props: Props) {
  const { baselineData, baselineResult, onResultChange } = props;
  const [simData, setSimData] = useState<AssessmentData>(baselineData);

  const simResult = useMemo(() => runRiskCalc(simData), [simData]);
  const delta = simResult.score - baselineResult.score;

  useEffect(() => {
    onResultChange?.(simResult);
  }, [onResultChange, simResult]);

  const setValue = (patch: Partial<AssessmentData>) => {
    setSimData((prev) => ({ ...prev, ...patch }));
  };

  const reset = () => setSimData(baselineData);

  const bestAchievable = () => {
    setSimData((prev) => ({
      ...prev,
      systolicBp: clamp(prev.systolicBp ?? 120, 110, 120),
      totalCholesterol: clamp(prev.totalCholesterol ?? 175, 150, 180),
      hdlCholesterol: clamp(prev.hdlCholesterol ?? 58, 55, 70),
      smoking: false,
      smokingStatus: "never",
      activityLevel: "high",
      unknownVitals: {
        systolicBp: false,
        totalCholesterol: false,
        hdlCholesterol: false,
      },
    }));
  };

  return (
    <div className="sim-wrap">
      <div className="sim-head">
        <h2 className="type-h3">What-If Risk Simulator</h2>
        <p className="lead">Adjust modifiable factors and view live risk impact.</p>
      </div>

      <div className="sim-grid">
        <label className="sim-field">
          <span>Systolic BP ({simData.systolicBp ?? "n/a"} mmHg)</span>
          <input
            type="range"
            min={90}
            max={180}
            value={simData.systolicBp ?? 120}
            onChange={(event) => setValue({ systolicBp: Number(event.target.value) })}
          />
        </label>

        <label className="sim-field">
          <span>Total Cholesterol ({simData.totalCholesterol ?? "n/a"} mg/dL)</span>
          <input
            type="range"
            min={120}
            max={320}
            value={simData.totalCholesterol ?? 200}
            onChange={(event) => setValue({ totalCholesterol: Number(event.target.value) })}
          />
        </label>

        <label className="sim-field">
          <span>HDL ({simData.hdlCholesterol ?? "n/a"} mg/dL)</span>
          <input
            type="range"
            min={20}
            max={90}
            value={simData.hdlCholesterol ?? 45}
            onChange={(event) => setValue({ hdlCholesterol: Number(event.target.value) })}
          />
        </label>
      </div>

      <div className="sim-toggle-row">
        <button
          type="button"
          className={`option-card ${simData.smoking ? "active" : ""}`}
          onClick={() => setValue({ smoking: true, smokingStatus: "current" })}
        >
          Current smoker
        </button>
        <button
          type="button"
          className={`option-card ${simData.smoking === false ? "active" : ""}`}
          onClick={() => setValue({ smoking: false, smokingStatus: "never" })}
        >
          Non-smoker
        </button>
        <button
          type="button"
          className={`option-card ${simData.activityLevel === "low" ? "active" : ""}`}
          onClick={() => setValue({ activityLevel: "low" })}
        >
          Low activity
        </button>
        <button
          type="button"
          className={`option-card ${simData.activityLevel === "moderate" ? "active" : ""}`}
          onClick={() => setValue({ activityLevel: "moderate" })}
        >
          Moderate activity
        </button>
        <button
          type="button"
          className={`option-card ${simData.activityLevel === "high" ? "active" : ""}`}
          onClick={() => setValue({ activityLevel: "high" })}
        >
          High activity
        </button>
      </div>

      <div className="sim-summary">
        <div className="sim-score mono">{simResult.score.toFixed(1)}%</div>
        <span className={`chip ${delta <= 0 ? "chip-success" : "chip-danger"}`}>
          {delta <= 0 ? "" : "+"}
          {delta.toFixed(1)}% vs current
        </span>
      </div>

      <div className="sim-actions">
        <button type="button" className="btn btn-subtle" onClick={bestAchievable}>
          Best Achievable
        </button>
        <button type="button" className="btn btn-subtle" onClick={reset}>
          Reset
        </button>
      </div>
    </div>
  );
}
