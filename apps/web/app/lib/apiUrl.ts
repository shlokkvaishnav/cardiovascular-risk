// Where the browser sends API calls.
//
// Normal builds call "/api/..." on the same host. Next.js rewrites forward
// those calls to the backend (see next.config.ts).
//
// The GitHub Pages build is a static site with no server, so it cannot rewrite.
// It sets NEXT_PUBLIC_DIRECT_API=true and calls the backend directly. The
// backend already accepts requests from any origin (CORS).
const DIRECT = process.env.NEXT_PUBLIC_DIRECT_API === "true";
const BACKEND = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

export function apiUrl(path: string): string {
  if (!DIRECT) return path;
  return BACKEND + path.replace(/^\/api/, "");
}
