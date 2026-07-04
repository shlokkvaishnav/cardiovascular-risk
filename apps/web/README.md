# CardioRisk Web App (`apps/web`)

Next.js App Router frontend for the Cardiovascular Risk Platform.

## Routes

- `/` Home
- `/assess` 7-step guided assessment wizard (supports uploading a lab report to auto-fill)
- `/results` interactive risk dashboard (heuristic preview → real SHAP-explained model result)
- `/about` methodology details
- `/privacy` data policy
- `/login` / `/register` optional account creation
- `/dashboard` saved report history + risk trend (requires login)
- `/dashboard/reports/[id]` saved report detail + PDF export

## Stack

- Next.js 16 (App Router)
- React 19
- TypeScript
- Tailwind CSS (global design tokens in `app/globals.css`)
- Framer Motion
- Lucide icons

## Run Locally

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:3000`.

## Scripts

```bash
npm run dev
npm run lint
npm run build
npm run start
```

## Key Files

- `app/components/AssessmentWizard.tsx`
- `app/components/ResultsDashboard.tsx`
- `app/components/SiteShell.tsx`
- `app/lib/risk.ts`
- `app/globals.css`

## Containerization

This app is containerized (`Dockerfile`, using Next.js `output: 'standalone'`) rather than
deployed to Vercel, so the whole stack (frontend + backend + Postgres) can run as one
docker-compose project or as sibling services on the same host.

`NEXT_PUBLIC_*` environment variables are baked in at Docker build time (not read at
runtime), so `NEXT_PUBLIC_API_URL` must be passed as a `--build-arg`, not just set on the
running container.

