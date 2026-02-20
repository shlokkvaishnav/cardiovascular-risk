# CardioRisk Web App (`apps/web`)

Next.js App Router frontend for the Cardiovascular Risk Platform.

## Routes

- `/` Home
- `/assess` 7-step guided assessment wizard
- `/results` interactive risk dashboard
- `/about` methodology details
- `/privacy` data policy

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

## Deployment (Vercel)

This is a monorepo sub-app. Set project Root Directory to:

```txt
apps/web
```

Do not use `web` as root directory.

