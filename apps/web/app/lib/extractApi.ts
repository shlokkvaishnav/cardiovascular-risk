import type { AssessmentData } from "./risk";

export type ExtractionResponse = {
  extracted_values: {
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
  confidence: Record<string, number>;
  patient_name: string;
  report_notes: string;
};

export class ExtractionApiError extends Error {}

export async function extractFromDocument(file: File): Promise<ExtractionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/api/extract", { method: "POST", body: formData });
  if (!res.ok) {
    let detail = `Extraction failed with status ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore
    }
    throw new ExtractionApiError(detail);
  }
  return res.json();
}

/** Cholesterol/glucose categories (1/2/3) don't have a single mg/dL value --
 * map to a representative midpoint of the standard clinical range so the
 * wizard's numeric cholesterol field has something reasonable to show,
 * clearly disclosed as estimated from the extracted category. */
const CHOLESTEROL_CATEGORY_TO_MGDL: Record<number, number> = { 1: 180, 2: 220, 3: 260 };

export function extractionToAssessmentPatch(extraction: ExtractionResponse): Partial<AssessmentData> {
  const v = extraction.extracted_values;
  return {
    age: v.age,
    sex: v.sex === 1 ? "male" : "female",
    heightCm: v.height,
    weightKg: v.weight,
    systolicBp: v.ap_hi,
    diastolicBp: v.ap_lo,
    totalCholesterol: CHOLESTEROL_CATEGORY_TO_MGDL[v.cholesterol] ?? 180,
    diabetes: v.gluc >= 2,
    smoking: v.smoke === 1,
    smokingStatus: v.smoke === 1 ? "current" : "never",
    alcohol: v.alco === 1,
    activityLevel: v.active === 1 ? "moderate" : "low",
    unknownVitals: {
      systolicBp: false,
      diastolicBp: false,
      totalCholesterol: false,
      hdlCholesterol: true, // not extractable from this document schema
    },
  };
}
