import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createEmptyAssessment, type AssessmentData } from "./risk";
import {
  backendResponseToRiskResult,
  fetchBackendPrediction,
  humanizeContributorLabel,
  mapAssessmentToPredictionRequest,
  PredictionApiError,
  type BackendPredictionResponse,
} from "./predictApi";

function fullAssessment(overrides: Partial<AssessmentData> = {}): AssessmentData {
  return {
    ...createEmptyAssessment(),
    age: 58,
    sex: "male",
    systolicBp: 145,
    diastolicBp: 90,
    totalCholesterol: 220,
    hdlCholesterol: 45,
    heightCm: 175,
    weightKg: 85,
    smoking: false,
    smokingStatus: "never",
    alcohol: false,
    activityLevel: "moderate",
    diabetes: false,
    familyHistory: false,
    familyHistoryType: "none",
    onBpMedication: false,
    ...overrides,
  };
}

describe("mapAssessmentToPredictionRequest", () => {
  it("maps a fully-filled assessment with no defaulted fields for collected values", () => {
    const { request, defaultedFields } = mapAssessmentToPredictionRequest(fullAssessment());

    expect(request).toEqual({
      age: 58,
      sex: 1,
      height: 175,
      weight: 85,
      ap_hi: 145,
      ap_lo: 90,
      cholesterol: 2, // 220 mg/dL -> "above normal"
      gluc: 1,
      smoke: 0,
      alco: 0,
      active: 1,
    });
    expect(defaultedFields).toEqual([]);
  });

  it("defaults missing fields and reports them", () => {
    const { request, defaultedFields } = mapAssessmentToPredictionRequest(createEmptyAssessment());

    expect(request.age).toBe(50);
    expect(request.height).toBe(165);
    expect(request.weight).toBe(75);
    expect(defaultedFields).toContain("age");
    expect(defaultedFields).toContain("height");
    expect(defaultedFields).toContain("weight");
  });

  it("derives diastolic BP from systolic when not collected", () => {
    const { request, defaultedFields } = mapAssessmentToPredictionRequest(
      fullAssessment({ diastolicBp: null }),
    );
    expect(request.ap_hi).toBe(145);
    expect(request.ap_lo).toBe(Math.round(145 * 0.65));
    expect(defaultedFields.some((f) => f.startsWith("ap_lo"))).toBe(true);
  });

  it("maps diabetes to an above-normal glucose proxy", () => {
    const { request } = mapAssessmentToPredictionRequest(fullAssessment({ diabetes: true }));
    expect(request.gluc).toBe(2);
  });

  it("maps low activity level to active=0, everything else to active=1", () => {
    expect(mapAssessmentToPredictionRequest(fullAssessment({ activityLevel: "low" })).request.active).toBe(0);
    expect(mapAssessmentToPredictionRequest(fullAssessment({ activityLevel: "high" })).request.active).toBe(1);
  });

  it("respects the unknownVitals 'I do not know' flags by defaulting", () => {
    const { request, defaultedFields } = mapAssessmentToPredictionRequest(
      fullAssessment({ unknownVitals: { systolicBp: true, diastolicBp: false, totalCholesterol: false, hdlCholesterol: false } }),
    );
    expect(request.ap_hi).toBe(130);
    expect(defaultedFields.some((f) => f.startsWith("ap_hi"))).toBe(true);
  });
});

describe("humanizeContributorLabel", () => {
  it("resolves numeric passthrough feature names exactly", () => {
    expect(humanizeContributorLabel("num__ap_hi")).toBe("Systolic blood pressure");
    expect(humanizeContributorLabel("num__age")).toBe("Age");
  });

  it("strips one-hot category suffixes for categorical features", () => {
    expect(humanizeContributorLabel("cat__cholesterol_2.0")).toBe("Cholesterol level");
    expect(humanizeContributorLabel("cat__smoke_1.0")).toBe("Smoking");
  });

  it("falls back to the raw key for unrecognized names", () => {
    expect(humanizeContributorLabel("totally_unknown_feature")).toBe("totally_unknown_feature");
  });
});

describe("backendResponseToRiskResult", () => {
  const baseResponse: BackendPredictionResponse = {
    prediction: 1,
    probability: 0.72,
    risk_level: "High",
    confidence: 0.72,
    timestamp: "2026-01-01T00:00:00Z",
    top_contributors: [
      { num__ap_hi: 0.3 },
      { num__age: -0.1 },
      { "cat__cholesterol_2.0": 0.15 },
    ],
    baseline_probability: 0.4,
  };

  it("converts probability to a 0-100 score and maps risk level to category", () => {
    const result = backendResponseToRiskResult(baseResponse, ["rec1"]);
    expect(result.score).toBeCloseTo(72);
    expect(result.category).toBe("high");
    expect(result.recommendations).toEqual(["rec1"]);
  });

  it("keeps only positive (risk-increasing) contributors as drivers, sorted descending", () => {
    const result = backendResponseToRiskResult(baseResponse, []);
    expect(result.drivers.map((d) => d.label)).toEqual(["Systolic blood pressure", "Cholesterol level"]);
    expect(result.drivers[0].impact).toBeGreaterThan(result.drivers[1].impact);
  });

  it("maps Medium risk level to the moderate category", () => {
    const result = backendResponseToRiskResult({ ...baseResponse, risk_level: "Medium" }, []);
    expect(result.category).toBe("moderate");
  });

  it("handles a null top_contributors gracefully", () => {
    const result = backendResponseToRiskResult({ ...baseResponse, top_contributors: null }, []);
    expect(result.drivers).toEqual([]);
  });
});

describe("fetchBackendPrediction", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("resolves with the backend response and defaulted fields on success", async () => {
    const mockResponse: BackendPredictionResponse = {
      prediction: 0,
      probability: 0.1,
      risk_level: "Low",
      confidence: 0.9,
      timestamp: "2026-01-01T00:00:00Z",
      top_contributors: [],
      baseline_probability: 0.2,
    };
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const { response } = await fetchBackendPrediction(fullAssessment());
    expect(response).toEqual(mockResponse);
    expect(global.fetch).toHaveBeenCalledWith(
      "/api/predict",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("throws PredictionApiError on a non-ok response", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: false, status: 503 });

    await expect(fetchBackendPrediction(fullAssessment())).rejects.toThrow(PredictionApiError);
  });

  it("throws PredictionApiError when fetch itself rejects (network error)", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("network down"));

    await expect(fetchBackendPrediction(fullAssessment())).rejects.toThrow(PredictionApiError);
  });
});
