import type { AssessmentData, RiskCategory, RiskDriver, RiskResult } from "./risk";

/**
 * The backend's 11 cardiovascular-lifestyle features (Kaggle cardiovascular
 * disease dataset). Mirrors backend/src/api/schemas.py:PredictionRequest.
 * BMI is derived server-side from height/weight, not sent directly.
 */
export type BackendPredictionRequest = {
  age: number;
  sex: number;
  height: number;
  weight: number;
  ap_hi: number;
  ap_lo: number;
  cholesterol: number;
  gluc: number;
  smoke: number;
  alco: number;
  active: number;
};

export type BackendPredictionResponse = {
  prediction: number;
  probability: number;
  risk_level: "Low" | "Medium" | "High";
  confidence: number;
  timestamp: string;
  request_id?: string | null;
  top_contributors?: Array<Record<string, number>> | null;
  baseline_probability?: number | null;
};

export type MappedPrediction = {
  request: BackendPredictionRequest;
  defaultedFields: string[];
};

/** Standard clinical cholesterol categorization (mg/dL) matching the
 * training dataset's 1/2/3 (normal/above normal/well above normal) coding. */
function cholesterolCategory(totalCholesterol: number): number {
  if (totalCholesterol < 200) return 1;
  if (totalCholesterol < 240) return 2;
  return 3;
}

export function mapAssessmentToPredictionRequest(data: AssessmentData): MappedPrediction {
  const defaultedFields: string[] = [];

  const age = data.age ?? 50;
  if (data.age === null) defaultedFields.push("age");

  const sex = data.sex === "male" ? 1 : data.sex === "female" ? 0 : 1;
  if (data.sex === null) defaultedFields.push("sex");

  const height = data.heightCm ?? 165;
  if (data.heightCm === null) defaultedFields.push("height");

  const weight = data.weightKg ?? 75;
  if (data.weightKg === null) defaultedFields.push("weight");

  const ap_hi = !data.unknownVitals.systolicBp && data.systolicBp !== null ? data.systolicBp : 130;
  if (data.unknownVitals.systolicBp || data.systolicBp === null) defaultedFields.push("ap_hi (systolic blood pressure)");

  // Diastolic isn't always collected; derive from systolic using a typical
  // pulse-pressure ratio rather than a flat constant, so it stays physiologically
  // consistent with whatever systolic value (real or defaulted) is being used.
  const ap_lo =
    !data.unknownVitals.diastolicBp && data.diastolicBp !== null
      ? data.diastolicBp
      : Math.round(ap_hi * 0.65);
  if (data.unknownVitals.diastolicBp || data.diastolicBp === null) defaultedFields.push("ap_lo (diastolic blood pressure)");

  const cholesterol =
    !data.unknownVitals.totalCholesterol && data.totalCholesterol !== null
      ? cholesterolCategory(data.totalCholesterol)
      : 1;
  if (data.unknownVitals.totalCholesterol || data.totalCholesterol === null) defaultedFields.push("cholesterol");

  // Glucose category isn't collected directly; a diabetes diagnosis is a
  // reasonable (if imperfect) proxy for an above-normal glucose reading.
  const gluc = data.diabetes === true ? 2 : 1;
  if (data.diabetes === null) defaultedFields.push("gluc (glucose category, via diabetes proxy)");

  const smoke = data.smokingStatus === "current" || data.smoking === true ? 1 : 0;
  if (data.smokingStatus === null && data.smoking === null) defaultedFields.push("smoke");

  const alco = data.alcohol === true ? 1 : 0;
  if (data.alcohol === null) defaultedFields.push("alco");

  const active = data.activityLevel === "low" ? 0 : data.activityLevel !== null ? 1 : 1;
  if (data.activityLevel === null) defaultedFields.push("active");

  return {
    request: { age, sex, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active },
    defaultedFields,
  };
}

const FEATURE_LABELS: Record<string, string> = {
  age: "Age",
  sex: "Sex at birth",
  height: "Height",
  weight: "Weight",
  bmi: "Body mass index",
  ap_hi: "Systolic blood pressure",
  ap_lo: "Diastolic blood pressure",
  cholesterol: "Cholesterol level",
  gluc: "Glucose level",
  smoke: "Smoking",
  alco: "Alcohol intake",
  active: "Physical activity",
};

/** Turns a raw sklearn ColumnTransformer output name (e.g. "num__oldpeak",
 * "cat__thal_2.0") into a human-readable label for display. */
export function humanizeContributorLabel(rawKey: string): string {
  const stripped = rawKey.replace(/^(num__|cat__)/, "");
  // Numeric passthrough features keep their exact column name (e.g. "ap_hi").
  if (FEATURE_LABELS[stripped]) return FEATURE_LABELS[stripped];
  // One-hot encoded categorical features append "_<category-value>" (e.g.
  // "cholesterol_2.0") -- strip that suffix and look up the base column name.
  const base = stripped.replace(/_[^_]+$/, "");
  return FEATURE_LABELS[base] ?? stripped;
}

function riskLevelToCategory(level: BackendPredictionResponse["risk_level"]): RiskCategory {
  if (level === "Low") return "low";
  if (level === "High") return "high";
  return "moderate";
}

/**
 * Converts a real backend prediction (with signed SHAP contributions) into
 * the same RiskResult shape the client-side heuristic produces, so it can
 * slot into the existing dashboard UI as the authoritative "current" result.
 */
export function backendResponseToRiskResult(
  response: BackendPredictionResponse,
  fallbackRecommendations: RiskResult["recommendations"],
): RiskResult {
  const score = Math.max(0, Math.min(100, response.probability * 100));
  const category = riskLevelToCategory(response.risk_level);

  const contributors = response.top_contributors ?? [];
  const drivers: RiskDriver[] = contributors
    .map((entry) => {
      const [rawKey, value] = Object.entries(entry)[0] ?? ["", 0];
      return { label: humanizeContributorLabel(rawKey), impact: value };
    })
    .filter((driver) => driver.impact > 0)
    .sort((a, b) => b.impact - a.impact);

  const summary =
    category === "low"
      ? "Clinical model estimate suggests a lower cardiovascular risk profile."
      : category === "moderate"
        ? "Clinical model estimate indicates meaningful risk that benefits from targeted intervention."
        : "Clinical model estimate indicates elevated risk requiring proactive clinical follow-up.";

  return {
    score,
    category,
    summary,
    drivers,
    recommendations: fallbackRecommendations,
  };
}

export class PredictionApiError extends Error {}

/**
 * Calls the real FastAPI backend for an authoritative, SHAP-explained
 * prediction. Uses the existing Next.js rewrite (/api/* -> backend) so this
 * works both in local dev and production without hardcoding a backend origin.
 * Times out rather than hanging indefinitely if the backend is unreachable
 * (e.g. cold start on a free-tier host).
 */
export async function fetchBackendPrediction(
  data: AssessmentData,
  opts: { timeoutMs?: number } = {},
): Promise<{ response: BackendPredictionResponse; defaultedFields: string[] }> {
  const { request, defaultedFields } = mapAssessmentToPredictionRequest(data);
  const timeoutMs = opts.timeoutMs ?? 8000;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const apiKey = process.env.NEXT_PUBLIC_API_KEY;
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "X-API-Key": apiKey } : {}),
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new PredictionApiError(`Backend prediction failed with status ${res.status}`);
    }

    const response = (await res.json()) as BackendPredictionResponse;
    return { response, defaultedFields };
  } catch (err) {
    if (err instanceof PredictionApiError) throw err;
    throw new PredictionApiError(err instanceof Error ? err.message : "Unknown prediction error");
  } finally {
    clearTimeout(timer);
  }
}
