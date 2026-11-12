# NEXUS 5G local application

The NEXUS 5G dashboard runs the selected model bundles in `results/models` via
a local FastAPI service. Its charts and counters describe analyses submitted to
this application; it does not ingest live network telemetry.

## Run locally

Install the optional API dependencies and start the backend from the repository
root:

```powershell
.\myenv\Scripts\python.exe -m pip install -e ".[app,data,models]"
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

## Public portfolio demo on Render

The repository includes a Dockerfile and Render Blueprint. Push the repository
to GitHub, then in Render choose **New → Blueprint** and connect that
repository. Render builds the frontend and Python API into one service, provides
an HTTPS URL, and uses the platform-assigned `PORT`. The Blueprint selects the
Free plan; review its memory and CPU limits before deploying.

The container runs as a non-root user and starts in `INA_PUBLIC_DEMO=true` mode.
It installs the CPU-only XGBoost package to avoid bundling GPU communication
libraries in the Render image.
In this mode:

- Submitted KPI windows and predictions are returned to the browser but are
  not written to SQLite. Prediction history is intentionally empty.
- API documentation routes are disabled, API responses are marked `no-store`,
  and errors avoid returning local file paths.
- Prediction requests are limited to 2 MiB, to 10 requests/minute, and to one
  inference at a time. Random-sample requests are limited to 6/minute.
- These in-memory limits are shared by all visitors and apply per service
  process. Keep a single instance for this demo; multiple instances require a
  shared rate limiter at the edge or in a centralized store.

The sample endpoint may download the public TelecomTS dataset on its first
request and cache it temporarily in the container. That cache is not user
input, and it is not persisted across container replacement. History and
prediction details are unavailable in public-demo mode.

The Free plan has 512 MB of memory. A local end-to-end inference used about
332 MiB, so the Free instance may run slowly or run out of memory depending on
its runtime overhead. Treat the Free plan as a best-effort portfolio demo; if
it restarts or times out, move to a larger plan.

## Input and local data

Submit one JSON window containing a `KPIs` object with all 18 channel names
listed in `web/src/api.ts`. Every channel must have exactly 128 values;
`DL_Protocol` and `UL_Protocol` accept strings or null, and numeric channels
accept numbers, numeric strings, or null. A template is available from the
Anomaly Detection screen. The REST endpoint is `POST /api/predictions`.

The Anomaly Detection screen can also download a random real window from the
internal held-out test partition recorded in
`results/models/binary_detection/comparison.json`. It uses the original
TelecomTS `train` source split and the saved test-row positions; this is the
project's internal holdout, not a separate public test split. Choose any,
normal, or anomaly reference class. The label shown in the UI is the dataset
annotation, kept separate from model predictions; the downloaded JSON contains
only the `KPIs` object. The first request may need internet access to cache
TelecomTS if it is not already available locally.

Prediction history, including submitted KPI values, is stored in
`results/app/predictions.sqlite3`. Set `INA_DATABASE_PATH` to use a different
local database file. Remove that file to clear the local history.

## Scope and security

Model monitoring reports local application activity and model evaluation
metrics, not operational network telemetry. Diagnostic explanations are
investigation hypotheses, not confirmed causes. The API binds to localhost by
default. Public-demo mode is intended only for a portfolio showcase: it has no
login, stores no submitted samples, and uses modest shared request limits. Do
not send real or sensitive network telemetry to it, and do not use it as an
operational or multi-user production service. Render terminates TLS for its
public HTTPS URL.
