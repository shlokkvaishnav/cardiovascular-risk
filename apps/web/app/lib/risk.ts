export type Sex = "female" | "male";
export type ActivityLevel = "low" | "moderate" | "high";
export type SmokingStatus = "never" | "former" | "current";
export type FamilyHistoryType = "none" | "male_lt55" | "female_lt65" | "both";

export type AssessmentData = {
  age: number | null;
  sex: Sex | null;
  systolicBp: number | null;
  totalCholesterol: number | null;
  hdlCholesterol: number | null;
  smoking: boolean | null;
  smokingStatus: SmokingStatus | null;
  activityLevel: ActivityLevel | null;
  diabetes: boolean | null;
  familyHistory: boolean | null;
  familyHistoryType: FamilyHistoryType | null;
  onBpMedication: boolean | null;
  unknownVitals: {
    systolicBp: boolean;
    totalCholesterol: boolean;
    hdlCholesterol: boolean;
  };
};

export type RiskCategory = "low" | "moderate" | "high";

export type RiskDriver = {
  label: string;
  impact: number;
};

export type RiskResult = {
  score: number;
  category: RiskCategory;
  summary: string;
  drivers: RiskDriver[];
  recommendations: string[];
};

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

export function createEmptyAssessment(): AssessmentData {
  return {
    age: null,
    sex: null,
    systolicBp: null,
    totalCholesterol: null,
    hdlCholesterol: null,
    smoking: null,
    smokingStatus: null,
    activityLevel: null,
    diabetes: null,
    familyHistory: null,
    familyHistoryType: null,
    onBpMedication: null,
    unknownVitals: {
      systolicBp: false,
      totalCholesterol: false,
      hdlCholesterol: false,
    },
  };
}

export function getRiskCategory(score: number): RiskCategory {
  if (score <= 7.5) return "low";
  if (score <= 20) return "moderate";
  return "high";
}

export function scoreRisk(data: AssessmentData): RiskResult {
  let rawScore = 0.035;
  const drivers: RiskDriver[] = [];

  if (data.age !== null) {
    const ageImpact = Math.max(0, (data.age - 35) * 0.004);
    rawScore += ageImpact;
    drivers.push({ label: "Age profile", impact: ageImpact });
  }

  if (data.sex === "male") {
    rawScore += 0.03;
    drivers.push({ label: "Sex at birth (male)", impact: 0.03 });
  }

  if (data.systolicBp !== null && !data.unknownVitals.systolicBp) {
    const bpImpact = Math.max(0, (data.systolicBp - 120) * 0.0014);
    rawScore += bpImpact;
    drivers.push({ label: "Systolic blood pressure", impact: bpImpact });
  } else if (data.unknownVitals.systolicBp) {
    rawScore += 0.02;
  }

  if (
    data.totalCholesterol !== null &&
    data.hdlCholesterol !== null &&
    !data.unknownVitals.totalCholesterol &&
    !data.unknownVitals.hdlCholesterol
  ) {
    const ratio = data.totalCholesterol / Math.max(20, data.hdlCholesterol);
    const cholImpact = Math.max(0, (ratio - 3.5) * 0.016);
    rawScore += cholImpact;
    drivers.push({ label: "Cholesterol ratio", impact: cholImpact });
  } else if (data.unknownVitals.totalCholesterol || data.unknownVitals.hdlCholesterol) {
    rawScore += 0.018;
  }

  if (data.smokingStatus === "current" || data.smoking) {
    rawScore += 0.06;
    drivers.push({ label: "Current smoking", impact: 0.06 });
  } else if (data.smokingStatus === "former") {
    rawScore += 0.022;
    drivers.push({ label: "Former smoking history", impact: 0.022 });
  }

  if (data.activityLevel === "low") {
    rawScore += 0.04;
    drivers.push({ label: "Low physical activity", impact: 0.04 });
  } else if (data.activityLevel === "moderate") {
    rawScore += 0.015;
    drivers.push({ label: "Moderate activity", impact: 0.015 });
  } else if (data.activityLevel === "high") {
    rawScore -= 0.006;
  }

  if (data.diabetes) {
    rawScore += 0.08;
    drivers.push({ label: "Diabetes history", impact: 0.08 });
  }

  if (data.familyHistoryType && data.familyHistoryType !== "none") {
    const familyImpact = data.familyHistoryType === "both" ? 0.035 : 0.024;
    rawScore += familyImpact;
    drivers.push({ label: "Family cardiac history", impact: familyImpact });
  } else if (data.familyHistory) {
    rawScore += 0.03;
    drivers.push({ label: "Family cardiac history", impact: 0.03 });
  }

  if (data.onBpMedication) {
    rawScore += 0.018;
    drivers.push({ label: "Current BP treatment", impact: 0.018 });
  }

  const score = clamp(rawScore * 100, 1, 45);
  const category = getRiskCategory(score);
  const topDrivers = drivers
    .filter((driver) => driver.impact > 0)
    .sort((a, b) => b.impact - a.impact)
    .slice(0, 4);

  const recommendations =
    category === "low"
      ? [
          "Maintain weekly aerobic activity and repeat screening yearly.",
          "Keep blood pressure below 120/80 mmHg with diet and sleep consistency.",
          "Continue smoke-free habits and routine preventive follow-ups.",
        ]
      : category === "moderate"
        ? [
            "Schedule a primary care review for lipid and blood pressure optimization.",
            "Target 150+ minutes of moderate exercise weekly and reduce sodium intake.",
            "Review smoking cessation support if applicable.",
          ]
        : [
            "Discuss intensive risk-reduction planning with a clinician within 2-4 weeks.",
            "Confirm labs (lipids, glucose) and evaluate BP treatment adherence.",
            "Create a structured lifestyle plan with measurable milestones.",
          ];

  const summary =
    category === "low"
      ? "Current profile suggests a lower short-term cardiovascular event probability."
      : category === "moderate"
        ? "Profile indicates meaningful risk that benefits from targeted intervention."
        : "Profile indicates elevated risk requiring proactive clinical follow-up.";

  return {
    score,
    category,
    summary,
    drivers: topDrivers,
    recommendations,
  };
}
