import { scoreRisk, type AssessmentData, type RiskResult } from "./risk";

export function runRiskCalc(input: AssessmentData): RiskResult {
  return scoreRisk(input);
}

