
import { describe, it, expect } from "vitest";
import { scoreRisk, getRiskCategory, createEmptyAssessment, type AssessmentData } from "./risk";

describe("Risk Assessment Logic", () => {
  it("creates an empty assessment correctly", () => {
    const empty = createEmptyAssessment();
    expect(empty.age).toBeNull();
    expect(empty.sex).toBeNull();
    expect(empty.unknownVitals.systolicBp).toBe(false);
  });

  it("calculates low risk for a healthy profile", () => {
    const data: AssessmentData = {
      ...createEmptyAssessment(),
      age: 30,
      sex: "female",
      systolicBp: 110,
      totalCholesterol: 160,
      hdlCholesterol: 60,
      smoking: false,
      activityLevel: "high",
      diabetes: false,
      familyHistory: false,
      onBpMedication: false,
    };

    const result = scoreRisk(data);
    expect(result.category).toBe("low");
    expect(result.score).toBeLessThan(5);
  });

  it("identifies high risk factors correctly", () => {
    const data: AssessmentData = {
      ...createEmptyAssessment(),
      age: 65,
      sex: "male",
      systolicBp: 160,
      totalCholesterol: 240,
      hdlCholesterol: 35,
      smoking: true,
      activityLevel: "low",
      diabetes: true,
      familyHistory: true,
      onBpMedication: true,
    };

    const result = scoreRisk(data);
    expect(result.category).toBe("high");
    expect(result.score).toBeGreaterThan(20);
    
    // Check drivers
    const driverLabels = result.drivers.map(d => d.label);
    expect(driverLabels).toContain("Age profile");
    expect(driverLabels).toContain("Diabetes history");
    expect(driverLabels).toContain("Current smoking");
  });

  it("handles unknown vitals by applying population averages/penalties", () => {
    const data: AssessmentData = {
      ...createEmptyAssessment(),
      age: 50,
      sex: "female",
      smoking: false,
      activityLevel: "moderate",
      diabetes: false,
      familyHistory: false,
      onBpMedication: false,
      unknownVitals: {
        systolicBp: true,
        totalCholesterol: true,
        hdlCholesterol: true,
      }
    };

    const result = scoreRisk(data);
    // Should still calculate a score
    expect(result.score).toBeGreaterThan(0);
    expect(result.score).toBeLessThan(45);
  });

  it("correctly categorizes risk scores", () => {
    expect(getRiskCategory(1)).toBe("low");
    expect(getRiskCategory(7)).toBe("low");
    expect(getRiskCategory(7.6)).toBe("moderate");
    expect(getRiskCategory(20)).toBe("moderate");
    expect(getRiskCategory(20.1)).toBe("high");
    expect(getRiskCategory(40)).toBe("high");
  });
});
