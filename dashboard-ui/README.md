# PerfLens Dashboard (M7 stretch)

A small React + TypeScript client for `perflens/dashboard/api.py` -- lists
runs, shows findings against baseline for a selected run, and charts metric
history for a selected finding. Read-only, no auth: this is meant for local
use against your own `perflens.db`, not as a public-facing app.

## Run it locally

```bash
# from the repo root: start the API (needs the `dashboard` extra)
pip install -e ".[dashboard]"
perflens dashboard --db perflens.db

# in another terminal, from dashboard-ui/
cp .env.example .env   # points at http://localhost:8000 by default
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

## Build

```bash
npm run build   # tsc -b && vite build -> dist/
npm run lint
```
