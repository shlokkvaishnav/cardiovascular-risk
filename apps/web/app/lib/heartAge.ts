import type { AssessmentData } from "./risk";

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

// Heuristic heart-age mapping layered on top of current risk score.
export function computeHeartAge(age: number, score: number, data: AssessmentData): number {
  const baselineRisk = clamp((age - 20) * 0.22, 1, 22);
  let delta = (score - baselineRisk) * 0.85;

  if (data.smoking) delta += 2;
  if (data.diabetes) delta += 3;
  if (data.activityLevel === "high") delta -= 1;
  if (data.systolicBp !== null && data.systolicBp >= 150) delta += 1.5;
  if (data.hdlCholesterol !== null && data.hdlCholesterol >= 60) delta -= 1;

  return Math.round(clamp(age + delta, 20, 90));
}

