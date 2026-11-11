# NEXUS 5G local application

The NEXUS 5G dashboard runs the selected model bundles in `results/models` via
a local FastAPI service. Its charts and counters describe analyses submitted to
this application; it does not ingest live network telemetry.

## Run locally

Install the optional API dependencies and start the backend from the repository
root:

```powershell
.\myenv\Scripts\python.exe -m pip install -e ".[app]"
.\myenv\Scripts\ina-app.exe
```

In another terminal, start the React development server:

```powershell
Set-Location web
npm install
npm run dev
```

Open the Vite URL shown in the terminal. The development server proxies `/api`
requests to `http://127.0.0.1:8000`.

To serve the production frontend from FastAPI, build it first and then start the
API:

```powershell
Set-Location web
npm run build
Set-Location ..
.\myenv\Scripts\ina-app.exe
```

## Input and local data

Submit one JSON window containing a `KPIs` object with all 18 channel names
listed in `web/src/api.ts`. Every channel must have exactly 128 values;
`DL_Protocol` and `UL_Protocol` accept strings or null, and numeric channels
accept numbers, numeric strings, or null. A template is available from the
Anomaly Detection screen. The REST endpoint is `POST /api/predictions`.

Prediction history, including submitted KPI values, is stored in
`results/app/predictions.sqlite3`. Set `INA_DATABASE_PATH` to use a different
local database file. Remove that file to clear the local history.

## Scope and security

Model monitoring reports local application activity and model evaluation
metrics, not operational network telemetry. Diagnostic explanations are
investigation hypotheses, not confirmed causes. The API binds to localhost by
default and has no authentication; do not expose it to a shared network or
production users without adding authentication, TLS, and access controls.
