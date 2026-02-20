# Cardiovascular Risk Platform

Full-stack cardiovascular risk project with:
- `backend/`: FastAPI + ML pipeline
- `apps/web/`: Next.js medical UX (guided assessment + results dashboard)

## Repository Layout

- `backend/` Python API, training scripts, tests, data/config
- `apps/web/` Next.js frontend
- `docker-compose.yml` local multi-service orchestration

## Web Experience (Current)

The frontend implements a modern, guided flow:
- `/` Home with trust signals and primary CTA
- `/assess` 7-step conversational wizard
- `/results` visual risk dashboard (gauge + risk drivers + action plan)
- `/about` methodology/trust content
- `/privacy` zero-storage policy

Design system highlights:
- Palette: `#C0392B`, `#922B21`, `#1A1A2E`, `#F8F9FA`
- Typography: Inter + JetBrains Mono
- Accessibility: keyboard-friendly controls, `aria-live` messaging, visible focus states

## Local Development

### 1) Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -e ".[dev,viz]"
uvicorn src.api.app:app --reload
```

### 2) Web

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:3000`.

## Quality Checks

### Backend tests

```bash
cd backend
pytest tests/ -v
```

### Web lint/build

```bash
cd apps/web
npm run lint
npm run build
```

## Deployment Notes (Vercel)

For this monorepo, set Vercel project **Root Directory** to:

```txt
apps/web
```

If Root Directory is set to `web`, deployment fails with:
`The specified Root Directory "web" does not exist.`

Also ensure `apps/web/app/lib/risk.ts` is tracked in git (it is required by wizard/dashboard imports).

## API Endpoints (Backend)

- `POST /predict`
- `POST /batch-predict`
- `GET /health`
- `POST /model/reload`
- `GET /model/info`

## License

MIT (`LICENSE`)

