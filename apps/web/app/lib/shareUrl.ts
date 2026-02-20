import type { AssessmentData, RiskResult } from "./risk";

type SharedPayload = {
  data: AssessmentData;
  result: RiskResult;
};

function toBase64Url(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function fromBase64Url(input: string): string {
  const normalized = input.replace(/-/g, "+").replace(/_/g, "/");
  const padLength = (4 - (normalized.length % 4)) % 4;
  const padded = normalized + "=".repeat(padLength);
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export function encodeSharedResult(payload: SharedPayload): string {
  return toBase64Url(JSON.stringify(payload));
}

export function decodeSharedResult(encoded: string): SharedPayload | null {
  try {
    const raw = fromBase64Url(encoded);
    return JSON.parse(raw) as SharedPayload;
  } catch {
    return null;
  }
}

export function createShareableResultsUrl(payload: SharedPayload): string {
  if (typeof window === "undefined") return "";
  const encoded = encodeSharedResult(payload);
  const url = new URL(`${window.location.origin}/results`);
  url.searchParams.set("data", encoded);
  return url.toString();
}

