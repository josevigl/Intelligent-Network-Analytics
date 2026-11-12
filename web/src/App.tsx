import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Antenna,
  ArrowDownToLine,
  Bell,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Cpu,
  FileJson2,
  Gauge,
  LayoutDashboard,
  LoaderCircle,
  Map as MapIcon,
  Menu,
  Radar,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
  type LucideIcon,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  Pie,
  PieChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  type ButtonHTMLAttributes,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Link,
  NavLink,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { api, CATEGORICAL_CHANNELS, KPI_CHANNELS, makeTemplate } from "./api";
import type {
  KpiInput,
  ModelInfo,
  Overview,
  Prediction,
  PredictionSummary,
  RandomTestSample,
  ScreenId,
} from "./types";

const palette = [
  "#004D98",
  "#A50044",
  "#6CABDD",
  "#4C9F70",
  "#FFC857",
  "#7A68A6",
  "#FF7F50",
  "#37A2DA",
  "#E6B422",
  "#00A6A6",
  "#D45087",
];

const screenMeta: Record<ScreenId, { title: string; subtitle: string }> = {
  overview: {
    title: "Network Overview",
    subtitle: "Overview of network performance and anomaly activity.",
  },
  detection: {
    title: "Anomaly Detection",
    subtitle: "Submit a KPI window to the selected network models.",
  },
  classification: {
    title: "Anomaly Classification",
    subtitle: "Inspect the predicted anomaly class and model confidence.",
  },
  localization: {
    title: "Temporal Localization",
    subtitle: "Review anomalous timesteps across the input window.",
  },
  diagnosis: {
    title: "Root Cause Analysis",
    subtitle: "Observed model outputs and investigation hypotheses.",
  },
  monitoring: {
    title: "Model Monitoring",
    subtitle: "Availability, recorded inference latency, and evaluation metrics.",
  },
  history: {
    title: "Prediction History",
    subtitle: "Search and review analyses saved on this workstation.",
  },
  settings: {
    title: "Settings",
    subtitle: "Application endpoints, models, and local data configuration.",
  },
};

const primaryNavigation: Array<{ id: ScreenId; label: string; icon: LucideIcon }> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "detection", label: "Anomaly Detection", icon: Radar },
  { id: "classification", label: "Classification", icon: BrainCircuit },
  { id: "localization", label: "Localization", icon: MapIcon },
  { id: "diagnosis", label: "Root Cause Analysis", icon: Sparkles },
];

const secondaryNavigation: Array<{ id: ScreenId; label: string; icon: LucideIcon }> = [
  { id: "monitoring", label: "Model Monitoring", icon: Activity },
  { id: "history", label: "Prediction History", icon: Clock3 },
  { id: "settings", label: "Settings", icon: Settings },
];

const screenPath: Record<ScreenId, string> = {
  overview: "/",
  detection: "/detection",
  classification: "/classification",
  localization: "/localization",
  diagnosis: "/diagnosis",
  monitoring: "/monitoring",
  history: "/history",
  settings: "/settings",
};

function currentScreen(pathname: string): ScreenId {
  const entry = Object.entries(screenPath).find(
    ([, path]) => path === pathname,
  );
  return (entry?.[0] as ScreenId | undefined) ?? "overview";
}

function formatPercent(value: number | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function shortId(value: string): string {
  return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-5)}` : value;
}

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const screen = currentScreen(location.pathname);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [activePrediction, setActivePrediction] = useState<Prediction | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [noticeOpen, setNoticeOpen] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    const [overviewResult, healthResult] = await Promise.allSettled([
      api.overview(),
      api.health(),
    ]);
    setOverview(overviewResult.status === "fulfilled" ? overviewResult.value : null);
    setApiOnline(healthResult.status === "fulfilled" ? healthResult.value.status === "ok" : false);
    setConnectionError(
      overviewResult.status === "rejected"
        ? overviewResult.reason instanceof Error
          ? overviewResult.reason.message
          : "Could not load overview data."
        : "",
    );
    setRefreshing(false);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onPrediction = useCallback(
    (prediction: Prediction) => {
      setActivePrediction(prediction);
      void refresh();
      navigate("/detection");
    },
    [navigate, refresh],
  );

  const selectPrediction = useCallback((prediction: Prediction) => {
    setActivePrediction(prediction);
  }, []);
  const openOverviewPrediction = useCallback(async (prediction: PredictionSummary) => {
    try {
      const details = await api.prediction(prediction.id);
      setActivePrediction(details);
      setSelectionError("");
      navigate("/diagnosis");
    } catch (reason) {
      setSelectionError(reason instanceof Error ? reason.message : "Could not load prediction details.");
    }
  }, [navigate]);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
        <Link className="brand" to="/" onClick={() => setMobileMenu(false)}>
          <span className="brand-mark"><Antenna size={22} strokeWidth={1.8} /></span>
          <span>
            <strong>NEXUS <b>5G</b></strong>
            <small>NETWORK INTELLIGENCE</small>
          </span>
        </Link>
        <div className="sidebar-rule" />
        <div className="nav-caption">WORKSPACE</div>
        <nav className="nav-list" aria-label="Primary navigation">
          {primaryNavigation.map((item) => (
            <NavItem
              key={item.id}
              {...item}
              active={screen === item.id}
              close={() => setMobileMenu(false)}
            />
          ))}
        </nav>
        <div className="nav-caption nav-caption-secondary">OPERATIONS</div>
        <nav className="nav-list" aria-label="Operations navigation">
          {secondaryNavigation.map((item) => (
            <NavItem
              key={item.id}
              {...item}
              active={screen === item.id}
              close={() => setMobileMenu(false)}
            />
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="cluster-card">
          <div className="cluster-heading">
            <span className="status-dot online" />
            <span>LOCAL ANALYTICS</span>
          </div>
          <div className="cluster-name">Model workspace</div>
          <div className="cluster-foot"><span>API</span><b>{apiOnline ? "CONNECTED" : apiOnline === false ? "OFFLINE" : "CHECKING"}</b></div>
          <div className="cluster-progress"><i /></div>
        </div>
        <div className="sidebar-footer">
          <span className="avatar">OP</span>
          <span className="operator-label"><strong>Network Operator</strong><small>Local workspace</small></span>
          <ChevronDown size={15} />
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <button
            className="icon-button mobile-menu-button"
            aria-label="Open navigation"
            onClick={() => setMobileMenu((open) => !open)}
          >
            <Menu size={19} />
          </button>
          <div className="breadcrumb">
            <span>Workspace</span><span className="crumb-slash">/</span>
            <strong>{screenMeta[screen].title}</strong>
          </div>
          <div className="topbar-actions">
            <div className={`system-status ${apiOnline ? "is-online" : apiOnline === false ? "is-offline" : ""}`}>
              <span className="status-dot" />
              {apiOnline ? "SYSTEM OPERATIONAL" : apiOnline === false ? "API UNAVAILABLE" : "CHECKING SYSTEM"}
            </div>
            <button
              className={`icon-button notification-button ${noticeOpen ? "active" : ""}`}
              aria-label="Notifications"
              onClick={() => setNoticeOpen((open) => !open)}
            >
              <Bell size={18} />
              <span className="notification-mark" />
            </button>
            <div className="topbar-avatar" title="Network Operator">OP</div>
          </div>
          {noticeOpen && (
            <div className="notification-popover">
              <strong>Workspace notices</strong>
              <p>{apiOnline ? "Local inference API is responding." : "Connect the local API to enable model inference."}</p>
              <button onClick={() => { setNoticeOpen(false); navigate("/settings"); }}>View system status</button>
            </div>
          )}
        </header>

        <main className="main-content">
          {connectionError && screen !== "settings" && (
            <div className="inline-error connection-error">
              <AlertCircle size={16} />
              <span>Cannot load dashboard data. Confirm the local API is running, then refresh. {connectionError}</span>
            </div>
          )}
          {selectionError && (
            <div className="inline-error connection-error">
              <AlertCircle size={16} /><span>Could not open this prediction: {selectionError}</span>
              <button aria-label="Dismiss error" onClick={() => setSelectionError("")}><X size={14} /></button>
            </div>
          )}
          <Routes>
            <Route path="/" element={<OverviewPage data={overview} refreshing={refreshing} refresh={refresh} onSelect={openOverviewPrediction} />} />
            <Route path="/detection" element={<DetectionPage active={activePrediction} onPrediction={onPrediction} onClearPrediction={() => setActivePrediction(null)} />} />
            <Route path="/classification" element={<ClassificationPage prediction={activePrediction} />} />
            <Route path="/localization" element={<LocalizationPage prediction={activePrediction} />} />
            <Route path="/diagnosis" element={<DiagnosisPage prediction={activePrediction} />} />
            <Route path="/monitoring" element={<MonitoringPage data={overview} />} />
            <Route path="/history" element={<HistoryPage onSelect={selectPrediction} />} />
            <Route path="/settings" element={<SettingsPage models={overview?.models ?? []} apiOnline={apiOnline} />} />
            <Route path="*" element={<OverviewPage data={overview} refreshing={refreshing} refresh={refresh} onSelect={openOverviewPrediction} />} />
          </Routes>
        </main>
      </div>
      {mobileMenu && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobileMenu(false)} />}
    </div>
  );
}

function NavItem({
  id,
  label,
  icon: Icon,
  active,
  close,
}: {
  id: ScreenId;
  label: string;
  icon: LucideIcon;
  active: boolean;
  close: () => void;
}) {
  return (
    <NavLink
      to={screenPath[id]}
      onClick={close}
      className={`nav-item ${active ? "nav-item-active" : ""}`}
      aria-current={active ? "page" : undefined}
    >
      <Icon size={17} strokeWidth={active ? 2 : 1.7} />
      <span>{label}</span>
      {id === "history" && <span className="nav-count">LOCAL</span>}
    </NavLink>
  );
}

function PageHeader({
  screen,
  eyebrow,
  action,
  children,
}: {
  screen: ScreenId;
  eyebrow?: string;
  action?: ReactNode;
  children?: ReactNode;
}) {
  const meta = screenMeta[screen];
  return (
    <div className="page-header">
      <div>
        <div className="eyebrow">{eyebrow ?? "NEXUS 5G / NETWORK OPERATIONS"}</div>
        <h1>{meta.title}</h1>
        <p>{meta.subtitle}</p>
        {children}
      </div>
      {action && <div className="page-actions">{action}</div>}
    </div>
  );
}

function Button({
  children,
  variant = "secondary",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger";
}) {
  return <button className={`button button-${variant}`} {...props}>{children}</button>;
}

function StatCard({
  label,
  value,
  change,
  icon: Icon,
  accent = "blue",
  footnote,
  loading = false,
}: {
  label: string;
  value: string;
  change?: string;
  icon: LucideIcon;
  accent?: string;
  footnote?: string;
  loading?: boolean;
}) {
  return (
    <article className={`stat-card accent-${accent}`}>
      <div className="stat-top"><span>{label}</span><span className="stat-icon"><Icon size={17} /></span></div>
      <div className="stat-value">{loading ? <span className="skeleton skeleton-number" /> : value}</div>
      <div className="stat-foot"><span>{footnote ?? change}</span>{change && <span className="stat-change">{change}</span>}</div>
    </article>
  );
}

function Panel({
  title,
  subtitle,
  action,
  className = "",
  children,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>
        {action && <div className="panel-action">{action}</div>}
      </div>
      {children}
    </section>
  );
}

function EmptyState({ title, text }: { title: string; text: string }) {
  return <div className="empty-state"><span className="empty-icon"><CircleHelp size={20} /></span><strong>{title}</strong><p>{text}</p></div>;
}

function ChartSkeleton({ rows = false }: { rows?: boolean }) {
  return (
    <div className={`chart-skeleton ${rows ? "chart-skeleton-rows" : ""}`} aria-label="Loading dashboard data">
      {rows
        ? Array.from({ length: 4 }, (_, index) => <i key={index} />)
        : <><i /><i /><i /><i /></>}
    </div>
  );
}

function OverviewPage({
  data,
  refreshing,
  refresh,
  onSelect,
}: {
  data: Overview | null;
  refreshing: boolean;
  refresh: () => void;
  onSelect: (prediction: PredictionSummary) => void;
}) {
  return (
    <>
      <PageHeader
        screen="overview"
        action={
          <>
            <span className="period-chip"><Clock3 size={14} /> Local analysis</span>
            <Button onClick={refresh} disabled={refreshing}><RefreshCw className={refreshing ? "spin" : ""} size={15} /> Refresh</Button>
          </>
        }
      >
        <div className="data-source-note"><span className="status-dot info-dot" /> Dataset-backed models · predictions are generated on demand, not live telemetry</div>
      </PageHeader>
      <div className="stat-grid">
        <StatCard label="ANALYSES RUN" value={data ? data.total_predictions.toLocaleString() : "—"} icon={Activity} footnote="Saved in local history" loading={!data && refreshing} />
        <StatCard label="ANOMALIES DETECTED" value={data ? data.anomalies.toLocaleString() : "—"} icon={AlertTriangle} accent="magenta" footnote="Across submitted windows" loading={!data && refreshing} />
        <StatCard label="AVG INFERENCE TIME" value={data ? `${data.average_inference_ms.toFixed(0)} ms` : "—"} icon={Gauge} accent="cyan" footnote="Measured by this app" loading={!data && refreshing} />
        <StatCard label="AVAILABLE MODELS" value={data ? `${data.models_available} / 4` : "—"} icon={BrainCircuit} accent="green" footnote="Selected local bundles" loading={!data && refreshing} />
      </div>
      <div className="notice-strip"><CircleHelp size={16} /><span><b>Dataset baseline:</b> {data?.dataset_baseline.name ?? "TelecomTS"} / {data?.dataset_baseline.split ?? "train"} · {data?.dataset_baseline.rows.toLocaleString() ?? "32,000"} source rows. Activity cards above show analyses run from this app, not network traffic.</span></div>
      <div className="dashboard-grid">
        <Panel title="Anomaly Distribution" subtitle="Predicted classes from local analyses">
          {data?.categories.length ? (
            <div className="chart-row">
              <div className="donut-wrap">
                <ResponsiveContainer width="100%" height={230}>
                  <PieChart>
                    <Pie data={data.categories} dataKey="count" nameKey="name" innerRadius={66} outerRadius={92} paddingAngle={3} stroke="none">
                      {data.categories.map((entry, index) => <Cell key={entry.name} fill={palette[index % palette.length]} />)}
                    </Pie>
                    <Tooltip content={<ChartTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="donut-center"><strong>{data.anomalies}</strong><span>ANOMALIES</span></div>
              </div>
              <div className="chart-legend">
                {data.categories.slice(0, 6).map((item, index) => (
                  <div className="legend-row" key={item.name}><span className="legend-swatch" style={{ background: palette[index % palette.length] }} /><span>{item.name.replaceAll("_", " ")}</span><b>{item.count}</b></div>
                ))}
              </div>
            </div>
          ) : refreshing && !data ? <ChartSkeleton /> : <EmptyState title="No prediction data yet" text="Submit a KPI window in Anomaly Detection to populate this chart." />}
        </Panel>
        <Panel title="Anomaly Categories" subtitle="Ranked by count · local prediction history">
          {data?.categories.length ? (
            <div className="ranked-chart">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={data.categories.slice(0, 8)} layout="vertical" margin={{ left: 8, right: 16, top: 3, bottom: 3 }}>
                  <CartesianGrid horizontal={false} stroke="#20324a" />
                  <XAxis type="number" hide />
                  <YAxis type="category" dataKey="name" width={112} tick={{ fill: "#98a8bb", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "#ffffff08" }} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={13}>
                    {data.categories.slice(0, 8).map((entry, index) => <Cell key={entry.name} fill={palette[index % palette.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : refreshing && !data ? <ChartSkeleton /> : <EmptyState title="Category ranking unavailable" text="Run an analysis that predicts an anomaly to build the ranking." />}
        </Panel>
      </div>
      <Panel title="Recent Predictions" subtitle="Latest analyses stored in this workspace" action={<Link className="text-link" to="/history">View history <span>→</span></Link>}>
        {data?.recent_predictions.length ? <PredictionTable predictions={data.recent_predictions} onSelect={onSelect} compact /> : refreshing && !data ? <ChartSkeleton rows /> : <EmptyState title="No recent predictions" text="Your analysis history will appear here after the first run." />}
      </Panel>
    </>
  );
}

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip"><strong>{label ?? payload[0].name}</strong><span>{payload[0].value}</span></div>;
}

function PredictionTable({
  predictions,
  onSelect,
  compact = false,
}: {
  predictions: PredictionSummary[];
  onSelect: (prediction: PredictionSummary) => void;
  compact?: boolean;
}) {
  return (
    <div className="table-scroll">
      <table className={`data-table ${compact ? "table-compact" : ""}`}>
        <thead><tr><th>Prediction ID</th><th>Date</th><th>Status</th><th>Predicted type</th><th>Confidence</th><th>{compact ? "" : "Affected KPIs"}</th><th /></tr></thead>
        <tbody>
          {predictions.map((item) => (
            <tr key={item.id}>
              <td><code>{shortId(item.id)}</code></td>
              <td>{formatDate(item.created_at)}</td>
              <td><StatusBadge status={item.anomaly_status} /></td>
              <td>{item.predicted_type.replaceAll("_", " ")}</td>
              <td><span className="confidence-cell">{formatPercent(item.confidence)}</span></td>
              {!compact && <td><div className="kpi-tags">{item.affected_kpis.slice(0, 3).map((kpi) => <span key={kpi}>{kpi}</span>)}</div></td>}
              <td><button className="table-view" onClick={() => onSelect(item)}>View <span>→</span></button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusBadge({ status }: { status: "normal" | "anomaly" }) {
  return <span className={`status-badge ${status}`}><i />{status === "anomaly" ? "Anomaly" : "Normal"}</span>;
}

function DetectionPage({
  active,
  onPrediction,
  onClearPrediction,
}: {
  active: Prediction | null;
  onPrediction: (prediction: Prediction) => void;
  onClearPrediction: () => void;
}) {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [kpis, setKpis] = useState<KpiInput | null>(null);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [sampleStatus, setSampleStatus] = useState<"any" | "normal" | "anomaly">("any");
  const [sampleReference, setSampleReference] = useState<RandomTestSample | null>(null);
  const [dragging, setDragging] = useState(false);
  const [selectedKpis, setSelectedKpis] = useState(["RSRP", "UL_SNR"]);
  const prediction = active;

  const acceptFile = async (file?: File) => {
    if (!file) return;
    setError("");
    setSampleReference(null);
    onClearPrediction();
    try {
      if (!file.name.toLowerCase().endsWith(".json")) throw new Error("Please choose a .json file.");
      const parsed = JSON.parse(await file.text()) as { KPIs?: KpiInput };
      const value = parsed.KPIs;
      if (!value || typeof value !== "object") throw new Error("JSON must contain a `KPIs` object.");
      const missing = KPI_CHANNELS.filter((channel) => !Array.isArray(value[channel]));
      if (missing.length) throw new Error(`Missing KPI arrays: ${missing.join(", ")}`);
      const invalid = KPI_CHANNELS.find((channel) => value[channel].length !== 128);
      if (invalid) throw new Error(`${invalid} must contain exactly 128 values.`);
      for (const channel of KPI_CHANNELS) {
        const values = value[channel];
        const invalidValue = values.findIndex((entry) => {
          if (CATEGORICAL_CHANNELS.has(channel)) return entry !== null && typeof entry !== "string";
          if (entry === null) return false;
          if (typeof entry === "number") return !Number.isFinite(entry);
          return typeof entry !== "string" || entry.trim() === "" || !Number.isFinite(Number(entry));
        });
        if (invalidValue !== -1) throw new Error(`${channel}[${invalidValue}] has an invalid value.`);
      }
      setKpis(value);
      setFileName(file.name);
    } catch (reason) {
      setKpis(null);
      setFileName("");
      setError(reason instanceof Error ? reason.message : "Could not read the file.");
    }
  };

  const runAnalysis = async () => {
    if (!kpis) return;
    setLoading(true);
    setError("");
    try {
      onPrediction(await api.predict(kpis));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Inference failed.");
    } finally {
      setLoading(false);
    }
  };

  const downloadRandomSample = async () => {
    setSampleLoading(true);
    setError("");
    onClearPrediction();
    try {
      const sample = await api.randomTestSample(sampleStatus);
      setKpis(sample.KPIs);
      setFileName(`TelecomTS held-out test window · row ${sample.source.row_position}`);
      setSampleReference(sample);
      downloadJson(
        `telecomts-test-window-${sample.source.row_position}.json`,
        { KPIs: sample.KPIs },
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load a TelecomTS test sample.");
    } finally {
      setSampleLoading(false);
    }
  };

  const chartData = useMemo(() => {
    if (!kpis) return [];
    return Array.from({ length: 128 }, (_, step) => {
      const values: Record<string, number | string> = { step };
      selectedKpis.forEach((channel) => {
        const value = kpis[channel]?.[step];
        values[channel] = typeof value === "number" ? value : Number(value) || 0;
      });
      return values;
    });
  }, [kpis, selectedKpis]);

  const toggleKpi = (kpi: string) => {
    setSelectedKpis((current) =>
      current.includes(kpi)
        ? current.length > 1 ? current.filter((value) => value !== kpi) : current
        : [...current, kpi].slice(-3),
    );
  };

  return (
    <>
      <PageHeader screen="detection" action={<Button onClick={() => { setKpis(null); setFileName(""); setError(""); setSampleReference(null); onClearPrediction(); }}><X size={15} /> New analysis</Button>} />
      <div className="input-layout">
        <Panel title="KPI Data Input" subtitle="Upload one 128-timestep network window in NEXUS JSON format" className="input-panel">
          <div
            className={`drop-zone ${dragging ? "drag-active" : ""} ${kpis ? "file-ready" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); void acceptFile(event.dataTransfer.files[0]); }}
          >
            <input ref={inputRef} className="visually-hidden" type="file" accept=".json,application/json" onChange={(event) => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ""; void acceptFile(file); }} />
            <span className="upload-icon">{kpis ? <Check size={24} /> : <UploadCloud size={25} />}</span>
            <strong>{kpis ? fileName : "Drag and drop your KPI file here"}</strong>
            <p>{kpis ? "Window format validated · ready to analyze" : "or select a file from your device"}</p>
            <Button onClick={() => inputRef.current?.click()}><FileJson2 size={15} /> Browse files</Button>
            <div className="file-format"><span>.JSON</span> 18 KPI channels · 128 values per channel</div>
          </div>
          {error && <div className="inline-error"><AlertCircle size={16} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}><X size={14} /></button></div>}
          <div className="validation-row"><span className={`validation-icon ${kpis ? "valid" : ""}`}>{kpis ? <Check size={13} /> : <AlertCircle size={13} />}</span><div><b>{kpis ? "Input validated" : "Waiting for input"}</b><small>{kpis ? `${KPI_CHANNELS.length} channels · 128 timesteps` : "Upload a JSON sample to validate its structure."}</small></div><span className={`validation-state ${kpis ? "valid" : ""}`}>{kpis ? "READY" : "PENDING"}</span></div>
          {kpis && (
            <div className="sample-summary">
              <div><strong>128</strong><span>timesteps</span></div>
              <div><strong>18</strong><span>KPI channels</span></div>
              <div><strong>16</strong><span>numerical</span></div>
              <div><strong>2</strong><span>categorical</span></div>
            </div>
          )}
          {sampleReference && (
            <div className="sample-provenance">
              <span className="eyebrow">DATASET REFERENCE · NOT MODEL OUTPUT</span>
              <div>
                <StatusBadge status={sampleReference.reference.anomaly_status} />
                <strong>{sampleReference.reference.anomaly_type.replaceAll("_", " ")}</strong>
              </div>
              <p>
                {sampleReference.source.dataset} · source <code>{sampleReference.source.source_split}</code>,
                internal held-out <code>{sampleReference.source.partition}</code> · row {sampleReference.source.row_position}
              </p>
              {sampleReference.reference.affected_kpis.length > 0 && (
                <div className="kpi-tags">{sampleReference.reference.affected_kpis.map((kpi) => <span key={kpi}>{kpi}</span>)}</div>
              )}
            </div>
          )}
          <div className="input-actions">
            <Button variant="primary" disabled={!kpis || loading} onClick={() => void runAnalysis()}>
              {loading ? <LoaderCircle className="spin" size={16} /> : <Radar size={16} />}
              {loading ? "Analyzing window…" : "Run detection"}
            </Button>
            <button className="text-link template-link" onClick={() => downloadTemplate()}><ArrowDownToLine size={14} /> Download JSON template</button>
          </div>
          <div className="real-sample-actions">
            <label className="filter-select"><span>REFERENCE</span><select aria-label="Sample reference class" value={sampleStatus} onChange={(event) => setSampleStatus(event.target.value as "any" | "normal" | "anomaly")}><option value="any">Any class</option><option value="normal">Normal</option><option value="anomaly">Anomaly</option></select></label>
            <Button disabled={sampleLoading || loading} onClick={() => void downloadRandomSample()}>
              {sampleLoading ? <LoaderCircle className="spin" size={14} /> : <ArrowDownToLine size={14} />}
              {sampleLoading ? "Selecting test sample…" : "Download real test sample"}
            </Button>
          </div>
          <p className="sample-source-note">Random real KPI window from the saved internal test holdout. The reference label is shown separately from the model prediction.</p>
          <p className="privacy-note"><ShieldCheck size={14} /> KPI windows are stored locally in SQLite to support prediction history.</p>
        </Panel>
        <aside className="analysis-aside">
          <div className="aside-card"><div className="aside-icon"><Cpu size={18} /></div><div className="eyebrow">INFERENCE PIPELINE</div><h3>Four selected models</h3><p>Each submitted window runs through detection, classification, localization, and affected-KPI models.</p><div className="pipeline-steps"><span>01 DETECT</span><i /><span>02 CLASSIFY</span><i /><span>03 LOCALIZE</span></div></div>
          <div className="aside-card data-contract"><div className="aside-icon cyan"><FileJson2 size={18} /></div><div className="eyebrow">INPUT CONTRACT</div><h3>One window per file</h3><p>Top-level JSON object with a <code>KPIs</code> map. Each of the 18 channel keys must contain exactly 128 values.</p><button className="code-sample" onClick={() => downloadTemplate()}><code>{'{"KPIs": { "RSRP": […] }}'}</code><ArrowDownToLine size={14} /></button></div>
        </aside>
      </div>
      {prediction && (
        <section className="result-section">
          <div className="result-header"><div><div className="eyebrow">LATEST ANALYSIS</div><h2>Prediction result <code>{shortId(prediction.id)}</code></h2></div><span className="result-time">{formatDate(prediction.created_at)}</span></div>
          <div className="result-grid">
            <div className={`result-status-card ${prediction.anomaly_status}`}>
              <div className="result-label">{prediction.anomaly_status === "anomaly" ? "ANOMALY DETECTED" : "NORMAL"}</div>
              <strong>{prediction.predicted_type.replaceAll("_", " ")}</strong>
              <div className="result-subtitle">{prediction.anomaly_status === "anomaly" ? "Model classification output" : "No anomaly predicted for this window"}</div>
              <div className="result-meta"><span>Prediction ID</span><code>{prediction.id}</code></div>
            </div>
            <div className="score-card">
              <div className="eyebrow">DETECTION SCORE</div>
              <ScoreGauge value={prediction.detection_score} />
              <div className="score-legend"><span><i className="anomaly-dot" /> Anomaly probability</span><b>{formatPercent(prediction.confidence)} confidence</b></div>
            </div>
          </div>
          <KpiSeries input={prediction.input?.KPIs ?? kpis ?? undefined} selected={selectedKpis} onToggle={toggleKpi} data={chartData.length ? chartData : undefined} />
          <div className="result-shortcuts">
            <button onClick={() => navigate("/classification")}><BrainCircuit size={15} /> Classification details →</button>
            <button onClick={() => navigate("/localization")}><MapIcon size={15} /> View timeline →</button>
            <button onClick={() => navigate("/diagnosis")}><Sparkles size={15} /> Diagnostic insights →</button>
          </div>
        </section>
      )}
    </>
  );
}

function downloadJson(fileName: string, value: unknown) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadTemplate() {
  downloadJson("nexus5g-kpi-template.json", { KPIs: makeTemplate() });
}

function ScoreGauge({ value }: { value: number }) {
  const percent = Math.max(0, Math.min(1, value));
  return (
    <div className="score-gauge">
      <svg viewBox="0 0 160 100" role="img" aria-label={`Anomaly score ${formatPercent(percent)}`}>
        <path d="M 20 82 A 60 60 0 0 1 140 82" pathLength="100" className="gauge-track" />
        <path d="M 20 82 A 60 60 0 0 1 140 82" pathLength="100" strokeDasharray={`${percent * 100} 100`} className="gauge-value" />
      </svg>
      <div className="gauge-text"><strong>{formatPercent(percent)}</strong><small>MODEL SCORE</small></div>
    </div>
  );
}

function KpiSeries({
  input,
  selected,
  onToggle,
  data,
}: {
  input?: KpiInput;
  selected: string[];
  onToggle: (kpi: string) => void;
  data?: Array<Record<string, number | string>>;
}) {
  const chartData = useMemo(() => data ?? (input ? Array.from({ length: 128 }, (_, step) => {
    const values: Record<string, number | string> = { step };
    selected.forEach((channel) => {
      const value = input[channel]?.[step];
      values[channel] = typeof value === "number" ? value : Number(value) || 0;
    });
    return values;
  }) : []), [data, input, selected]);
  return (
    <Panel title="KPI Signal Trends" subtitle="Observed input values · select up to three numeric channels" className="signal-panel">
      <div className="kpi-selectors">
        {KPI_CHANNELS.filter((name) => !CATEGORICAL_CHANNELS.has(name)).map((kpi) => (
          <button key={kpi} className={selected.includes(kpi) ? "selected" : ""} onClick={() => onToggle(kpi)}>{kpi}</button>
        ))}
      </div>
      {chartData.length ? (
        <div className="signal-chart">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{ left: 0, right: 14, top: 10, bottom: 0 }}>
              <CartesianGrid stroke="#20324a" strokeDasharray="3 5" />
              <XAxis dataKey="step" tick={{ fill: "#718198", fontSize: 10 }} tickLine={false} axisLine={false} ticks={[0, 32, 64, 96, 127]} />
              <YAxis tick={{ fill: "#718198", fontSize: 10 }} tickLine={false} axisLine={false} width={44} />
              <Tooltip content={<ChartTooltip />} />
              {selected.map((kpi, index) => <Line key={kpi} type="monotone" dataKey={kpi} stroke={palette[index]} strokeWidth={1.7} dot={false} activeDot={{ r: 4 }} />)}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : <EmptyState title="Signal chart awaits a sample" text="Upload a KPI window to inspect its timeseries." />}
    </Panel>
  );
}

function PredictionRequired({ screen }: { screen: ScreenId }) {
  const navigate = useNavigate();
  return (
    <>
      <PageHeader screen={screen} />
      <div className="standalone-empty">
        <span className="empty-icon large"><Radar size={24} /></span>
        <h2>No analysis selected</h2>
        <p>Run a KPI window through the inference pipeline to see {screenMeta[screen].title.toLowerCase()} results here.</p>
        <Button variant="primary" onClick={() => navigate("/detection")}><UploadCloud size={15} /> Start an analysis</Button>
      </div>
    </>
  );
}

function ClassificationPage({ prediction }: { prediction: Prediction | null }) {
  if (!prediction) return <PredictionRequired screen="classification" />;
  const task = prediction.details.tasks.anomaly_type;
  const scores = task?.scores ?? [];
  return (
    <>
      <PageHeader screen="classification" eyebrow="MODEL OUTPUT / ANOMALY TYPE">
        <div className="sample-id-line">SAMPLE <code>{prediction.id}</code><span>·</span><StatusBadge status={prediction.anomaly_status} /></div>
      </PageHeader>
      <section className="classification-hero">
        <div className="class-hero-copy"><div className="eyebrow">PREDICTED CLASS</div><h2>{prediction.predicted_type.replaceAll("_", " ")}</h2><p>{prediction.anomaly_status === "normal" ? "The binary model found no anomaly in this window. Class output shown for completeness." : "Selected anomaly-classification model output."}</p></div>
        <div className="class-confidence"><span>CLASS CONFIDENCE</span><strong>{formatPercent(task?.confidence)}</strong><div className="confidence-track"><i style={{ width: `${(task?.confidence ?? 0) * 100}%` }} /></div></div>
      </section>
      <Panel title="Category Probability Scores" subtitle={`Model: ${task?.model_name ?? "unavailable"} · normalized model outputs`}>
        {scores.length ? (
          <div className="horizontal-scores">
            {scores.map((item, index) => (
              <div className={`score-row ${item.label === prediction.predicted_type ? "score-row-active" : ""}`} key={item.label}>
                <span className="score-name">{item.label.replaceAll("_", " ")}</span>
                <div className="score-track"><i style={{ width: `${item.score * 100}%`, background: palette[index % palette.length] }} /></div>
                <strong>{formatPercent(item.score)}</strong>
              </div>
            ))}
          </div>
        ) : <EmptyState title="Class scores unavailable" text="This estimator does not expose a probability distribution for its classes." />}
      </Panel>
      <Panel title="Prediction Metadata" subtitle="Provenance for this inference">
        <div className="metadata-grid">
          <MetaItem label="Predicted class" value={prediction.predicted_type.replaceAll("_", " ")} />
          <MetaItem label="Model family" value={task?.model_name ?? "—"} />
          <MetaItem label="Model runtime version" value={task?.model_version ?? "—"} mono />
          <MetaItem label="Prediction ID" value={prediction.id} mono />
          <MetaItem label="Inference confidence" value={formatPercent(task?.confidence)} />
          <MetaItem label="Inference time" value={`${prediction.inference_ms.toFixed(0)} ms`} />
        </div>
      </Panel>
    </>
  );
}

function MetaItem({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <div className="meta-item"><span>{label}</span><strong className={mono ? "mono" : ""}>{value}</strong></div>;
}

function LocalizationPage({ prediction }: { prediction: Prediction | null }) {
  const [selected, setSelected] = useState("RSRP");
  const [hoverStep, setHoverStep] = useState<number | null>(null);
  if (!prediction) return <PredictionRequired screen="localization" />;
  const localization = prediction.details.localization;
  const input = prediction.input?.KPIs;
  const series = input?.[selected] ?? [];
  const chart = series.map((value, step) => ({ step, value: Number(value) || 0, anomaly: localization.timeline[step] }));
  const stats = [
    { title: "First anomalous timestep", value: localization.first_anomalous_timestep === null ? "—" : localization.first_anomalous_timestep.toString().padStart(3, "0"), tone: "magenta" },
    { title: "Last anomalous timestep", value: localization.last_anomalous_timestep === null ? "—" : localization.last_anomalous_timestep.toString().padStart(3, "0"), tone: "magenta" },
    { title: "Anomalous timesteps", value: localization.anomalous_timesteps.toString(), tone: "blue" },
    { title: "Anomalous segments", value: localization.segments.length.toString(), tone: "cyan" },
  ];
  return (
    <>
      <PageHeader screen="localization" eyebrow="MODEL OUTPUT / TIME WINDOW">
        <div className="sample-id-line">WINDOW <code>{prediction.id}</code><span>·</span>TYPE <b>{prediction.predicted_type.replaceAll("_", " ")}</b></div>
      </PageHeader>
      <Panel title="Anomaly Timeline" subtitle="128 timesteps · highlighted segments are localization-model predictions" action={<span className="legend-inline"><i className="legend-normal" /> Normal <i className="legend-anomaly" /> Anomalous</span>} className="timeline-panel">
        {((prediction.anomaly_status === "normal" && localization.anomalous_timesteps > 0) || (prediction.anomaly_status === "anomaly" && localization.anomalous_timesteps === 0)) && (
          <div className="model-disagreement"><AlertTriangle size={15} /><span>The binary detector and temporal localization model disagree for this window. These are independent model outputs; review the sample before acting.</span></div>
        )}
        <div className="timeline-labels"><span>t = 0</span><span>128 TIMESTEPS</span><span>t = 127</span></div>
        <div className="timeline-grid">
          {localization.timeline.map((isAnomaly, index) => (
            <button
              key={index}
              className={`timeline-step ${isAnomaly ? "step-anomaly" : "step-normal"} ${hoverStep === index ? "step-hover" : ""}`}
              title={`Timestep ${index}: ${isAnomaly ? "anomalous" : "normal"}`}
              onMouseEnter={() => setHoverStep(index)}
              onMouseLeave={() => setHoverStep(null)}
              onFocus={() => setHoverStep(index)}
              onBlur={() => setHoverStep(null)}
            />
          ))}
        </div>
        <div className="timeline-hover">{hoverStep === null ? "Hover over the timeline to inspect a timestep." : `Timestep ${hoverStep} · ${localization.timeline[hoverStep] ? "Anomalous" : "Normal"}`}</div>
      </Panel>
      <div className="four-stat-grid">
        {stats.map((item) => <div className={`localization-stat ${item.tone}`} key={item.title}><span>{item.title}</span><strong>{item.value}</strong></div>)}
      </div>
      <Panel title="KPI Signal Detail" subtitle="Input signal aligned with localized model output" action={<select className="dark-select" value={selected} onChange={(event) => setSelected(event.target.value)}>{KPI_CHANNELS.filter((kpi) => !CATEGORICAL_CHANNELS.has(kpi)).map((kpi) => <option key={kpi}>{kpi}</option>)}</select>}>
        {chart.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <ComposedChart data={chart} margin={{ left: 0, right: 12, top: 10 }}>
              <CartesianGrid stroke="#20324a" strokeDasharray="3 5" />
              <XAxis dataKey="step" tick={{ fill: "#718198", fontSize: 10 }} ticks={[0, 32, 64, 96, 127]} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#718198", fontSize: 10 }} width={45} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              {localization.segments.map((segment) => <ReferenceArea key={`${segment.start}-${segment.end}`} x1={segment.start} x2={segment.end} fill="#A50044" fillOpacity={0.17} strokeOpacity={0} />)}
              <Line dataKey="value" stroke="#6CABDD" strokeWidth={1.8} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        ) : <EmptyState title="Input signals not available" text="The selected analysis did not include stored KPI values." />}
      </Panel>
      <Panel title="Localized Intervals" subtitle="Contiguous anomalous ranges">
        {localization.segments.length ? <div className="interval-list">{localization.segments.map((segment, index) => <div className="interval-row" key={`${segment.start}-${segment.end}`}><span className="interval-index">SEGMENT {String(index + 1).padStart(2, "0")}</span><div><strong>{segment.start} — {segment.end}</strong><small>{segment.end - segment.start + 1} timesteps</small></div><span className="interval-bar"><i style={{ width: `${((segment.end - segment.start + 1) / 128) * 100}%` }} /></span></div>)}</div> : <EmptyState title="No anomalous interval" text="The localization model marked the full window as normal." />}
      </Panel>
    </>
  );
}

function DiagnosisPage({ prediction }: { prediction: Prediction | null }) {
  if (!prediction) return <PredictionRequired screen="diagnosis" />;
  const diagnosis = prediction.details.diagnostic;
  const localization = prediction.details.localization;
  return (
    <>
      <PageHeader screen="diagnosis" eyebrow="INCIDENT REVIEW / MODEL-ASSISTED" />
      <div className="incident-summary">
        <div className="incident-top"><div><span className="eyebrow">INCIDENT SUMMARY</span><h2>{prediction.anomaly_status === "anomaly" ? "Anomaly predicted" : "No anomaly predicted"}</h2></div><StatusBadge status={prediction.anomaly_status} /></div>
        <div className="incident-facts">
          <MetaItem label="Predicted type" value={prediction.predicted_type.replaceAll("_", " ")} />
          <MetaItem label="Detection score" value={formatPercent(prediction.detection_score)} />
          <MetaItem label="Localized duration" value={`${localization.anomalous_timesteps} / 128 steps`} />
          <MetaItem label="Prediction ID" value={prediction.id} mono />
        </div>
      </div>
      <Panel title="Affected KPIs" subtitle="Outputs from the multi-label identification model">
        {prediction.affected_kpis.length ? <div className="affected-kpi-grid">{prediction.affected_kpis.map((kpi, index) => <div className="affected-kpi" key={kpi}><span className="affected-symbol">{kpi.slice(0, 2)}</span><div><strong>{kpi}</strong><small>Model-identified indicator</small></div><span className="affected-index">0{index + 1}</span></div>)}</div> : <EmptyState title="No KPIs identified" text="No affected KPI crossed the selected model's learned threshold." />}
      </Panel>
      <div className="diagnostic-grid">
        <Panel title="Diagnostic Insights" subtitle="Model prediction and observations from the submitted window">
          <div className="diagnostic-summary"><Sparkles size={17} /><p>{diagnosis.summary}</p></div>
          <div className="diagnostic-section-label">OBSERVED</div>
          <ul className="insight-list observed-list">{diagnosis.observed.map((item) => <li key={item}><span className="observed-bullet" />{item}</li>)}</ul>
          <div className="diagnostic-caveat"><AlertCircle size={15} /><span>{diagnosis.notice}</span></div>
        </Panel>
        <Panel title="Investigation Hypotheses" subtitle="Suggested checks · not confirmed root causes">
          {diagnosis.hypotheses.length ? <ol className="hypothesis-list">{diagnosis.hypotheses.map((item, index) => <li key={item}><span>{String(index + 1).padStart(2, "0")}</span><p>{item}</p></li>)}</ol> : <EmptyState title="No investigation prompts" text="No anomaly was predicted for this window." />}
          <div className="hypothesis-footer"><ShieldCheck size={14} /> Operator verification required before remediation</div>
        </Panel>
      </div>
    </>
  );
}

function MonitoringPage({ data }: { data: Overview | null }) {
  const [predictions, setPredictions] = useState<PredictionSummary[]>([]);
  const [historyError, setHistoryError] = useState("");
  useEffect(() => {
    void api.history("limit=30")
      .then(setPredictions)
      .catch((reason: unknown) => setHistoryError(reason instanceof Error ? reason.message : "Monitoring history is unavailable."));
  }, []);
  const latency = predictions.slice().reverse().map((item, index) => ({ run: index + 1, ms: item.inference_ms }));
  const requestCounts = useMemo(() => {
    const counts = new globalThis.Map<string, number>();
    predictions.forEach((item) => {
      const key = item.created_at.slice(0, 10);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return [...counts].map(([day, count]) => ({ day: day.slice(5), count }));
  }, [predictions]);
  const modelMetrics = (model: ModelInfo) => model.metrics.f1_macro;
  return (
    <>
      <PageHeader screen="monitoring">
        <div className="data-source-note"><span className="status-dot info-dot" /> Local telemetry only · no live serving or error collector configured</div>
      </PageHeader>
      {historyError && <div className="inline-error"><AlertCircle size={16} /><span>Could not load latency history: {historyError}</span></div>}
      <div className="stat-grid">
        <StatCard label="TOTAL MODELS" value="4" icon={BrainCircuit} footnote="Configured objectives" />
        <StatCard label="AVAILABLE MODELS" value={`${data?.models_available ?? "—"} / 4`} icon={ShieldCheck} accent="green" footnote="Local selected bundles" />
        <StatCard label="AVG INFERENCE TIME" value={data ? `${data.average_inference_ms.toFixed(0)} ms` : "—"} icon={Gauge} accent="cyan" footnote="App-recorded per window" />
        <StatCard label="TOTAL PREDICTIONS" value={data?.total_predictions.toLocaleString() ?? "—"} icon={Activity} accent="magenta" footnote="Persisted in SQLite" />
      </div>
      <Panel title="Registered Models" subtitle="Selected models and held-out evaluation metrics">
        <div className="table-scroll"><table className="data-table model-table"><thead><tr><th>MODEL</th><th>TASK</th><th>FAMILY</th><th>VERSION</th><th>TEST F1 (MACRO)</th><th>STATUS</th></tr></thead><tbody>
          {(data?.models ?? []).map((model) => <tr key={model.task}><td><span className="model-name"><span className="model-symbol"><BrainCircuit size={16} /></span>{model.title}</span></td><td><code>{model.task}</code></td><td>{model.model_name ?? "—"}</td><td><code>{model.version ?? "—"}</code></td><td>{modelMetrics(model) === undefined ? "—" : formatPercent(modelMetrics(model))}<small className="metric-source">{model.metric_source === "test" ? "held-out test" : model.metric_source ?? ""}</small></td><td><span className={`model-status ${model.available ? "available" : "unavailable"}`}><i />{model.available ? "Available" : "Unavailable"}</span></td></tr>)}
        </tbody></table></div>
      </Panel>
      <div className="dashboard-grid monitoring-charts">
        <Panel title="Inference Latency" subtitle="Measured duration per saved analysis">
          {latency.length ? <ResponsiveContainer width="100%" height={210}><LineChart data={latency}><CartesianGrid stroke="#20324a" strokeDasharray="3 5" /><XAxis dataKey="run" tick={{ fill: "#718198", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "#718198", fontSize: 10 }} width={42} axisLine={false} tickLine={false} /><Tooltip content={<ChartTooltip />} /><Line dataKey="ms" name="Latency (ms)" stroke="#6CABDD" strokeWidth={2} dot={{ r: 2, fill: "#6CABDD" }} /></LineChart></ResponsiveContainer> : <EmptyState title="No latency samples yet" text="Latency is measured when analyses are submitted through this app." />}
        </Panel>
        <Panel title="Analysis Volume" subtitle="Saved predictions grouped by date">
          {requestCounts.length ? <ResponsiveContainer width="100%" height={210}><BarChart data={requestCounts}><CartesianGrid vertical={false} stroke="#20324a" strokeDasharray="3 5" /><XAxis dataKey="day" tick={{ fill: "#718198", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "#718198", fontSize: 10 }} width={30} axisLine={false} tickLine={false} /><Tooltip content={<ChartTooltip />} /><Bar dataKey="count" name="Analyses" fill="#004D98" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer> : <EmptyState title="No analysis volume yet" text="Request counts are local app analyses, not network traffic." />}
        </Panel>
      </div>
      <div className="dashboard-grid">
        <Panel title="Prediction Outcomes" subtitle="Normal vs anomaly · recorded app analyses">
          {data?.total_predictions ? <div className="outcome-bars"><OutcomeBar label="Anomaly" count={data.anomalies} total={data.total_predictions} color="#A50044" /><OutcomeBar label="Normal" count={data.total_predictions - data.anomalies} total={data.total_predictions} color="#4C9F70" /></div> : <EmptyState title="No outcome data" text="Run an analysis to record model outcomes." />}
        </Panel>
        <Panel title="Error Distribution" subtitle="Runtime error telemetry">
          <EmptyState title="Error collection is not enabled" text="Failed requests are returned to the user but are not persisted as monitoring events in this local release." />
        </Panel>
      </div>
    </>
  );
}

function OutcomeBar({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  return <div className="outcome-row"><div><span>{label}</span><b>{count}</b><small>{formatPercent(total ? count / total : 0)}</small></div><div className="outcome-track"><i style={{ width: `${total ? count / total * 100 : 0}%`, background: color }} /></div></div>;
}

function HistoryPage({ onSelect }: { onSelect: (prediction: Prediction) => void }) {
  const [items, setItems] = useState<PredictionSummary[]>([]);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [type, setType] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [selected, setSelected] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const load = useCallback(async () => {
    const params = new URLSearchParams({ limit: "500" });
    if (search) params.set("search", search);
    if (status) params.set("status", status);
    if (type) params.set("predicted_type", type);
    if (dateFrom) params.set("date_from", dateFrom);
    if (dateTo) params.set("date_to", dateTo);
    try { setItems(await api.history(params.toString())); setError(""); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "History unavailable."); }
  }, [search, status, type, dateFrom, dateTo]);
  useEffect(() => { void load(); }, [load]);
  const openDetails = async (item: PredictionSummary) => {
    setDetailLoading(true);
    setDetailError("");
    try {
      const details = await api.prediction(item.id);
      setSelected(details);
      onSelect(details);
    } catch (reason) {
      setDetailError(reason instanceof Error ? reason.message : "Could not load prediction details.");
    } finally {
      setDetailLoading(false);
    }
  };
  const types = [...new Set(items.map((item) => item.predicted_type))];
  return (
    <>
      <PageHeader screen="history" action={<Button onClick={() => void load()}><RefreshCw size={15} /> Refresh</Button>} />
      <Panel title="Analysis Records" subtitle={`${items.length} saved prediction${items.length === 1 ? "" : "s"} · stored locally in SQLite`} className="history-panel">
        <div className="filter-toolbar">
          <label className="search-field"><Search size={16} /><input placeholder="Search prediction ID or type…" value={search} onChange={(event) => setSearch(event.target.value)} /></label>
          <label className="filter-select"><span>STATE</span><select value={status} onChange={(event) => setStatus(event.target.value)}><option value="">All states</option><option value="anomaly">Anomaly</option><option value="normal">Normal</option></select></label>
          <label className="filter-select"><span>TYPE</span><select value={type} onChange={(event) => setType(event.target.value)}><option value="">All types</option>{types.map((item) => <option key={item} value={item}>{item.replaceAll("_", " ")}</option>)}</select></label>
          <label className="filter-select date-filter"><span>FROM</span><input aria-label="Filter from date" type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} /></label>
          <label className="filter-select date-filter"><span>TO</span><input aria-label="Filter to date" type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} /></label>
        </div>
        {error && <div className="inline-error"><AlertCircle size={16} />{error}</div>}
        {detailError && <div className="inline-error"><AlertCircle size={16} /><span>{detailError}</span></div>}
        {items.length ? <PredictionTable predictions={items} onSelect={(item) => void openDetails(item)} /> : <EmptyState title="No matching records" text="Submitted analyses are saved locally and will appear here." />}
      </Panel>
      {detailLoading && <div className="loading-indicator"><LoaderCircle className="spin" size={15} /> Loading prediction details…</div>}
      {selected && <PredictionDrawer prediction={selected} onClose={() => setSelected(null)} />}
    </>
  );
}

function PredictionDrawer({ prediction, onClose }: { prediction: Prediction; onClose: () => void }) {
  const navigate = useNavigate();
  return (
    <div className="drawer-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <aside className="prediction-drawer">
        <div className="drawer-header"><div><div className="eyebrow">PREDICTION DETAIL</div><h2>{prediction.predicted_type.replaceAll("_", " ")}</h2></div><button className="icon-button" aria-label="Close details" onClick={onClose}><X size={18} /></button></div>
        <StatusBadge status={prediction.anomaly_status} />
        <div className="drawer-metadata"><MetaItem label="Prediction ID" value={prediction.id} mono /><MetaItem label="Date" value={formatDate(prediction.created_at)} /><MetaItem label="Detection score" value={formatPercent(prediction.detection_score)} /><MetaItem label="Inference time" value={`${prediction.inference_ms.toFixed(0)} ms`} /></div>
        <div className="drawer-section"><span className="eyebrow">AFFECTED KPIs</span><div className="kpi-tags">{prediction.affected_kpis.length ? prediction.affected_kpis.map((kpi) => <span key={kpi}>{kpi}</span>) : <small>No KPIs identified</small>}</div></div>
        <div className="drawer-section"><span className="eyebrow">LOCALIZED TIMESTEPS</span><div className="timeline-grid small-timeline">{prediction.details.localization.timeline.map((value, index) => <i key={index} className={value ? "step-anomaly" : "step-normal"} />)}</div><p>{prediction.details.localization.anomalous_timesteps} of 128 timesteps flagged · {prediction.details.localization.segments.length} segments</p></div>
        <div className="drawer-section"><span className="eyebrow">DIAGNOSTIC NOTE</span><p className="drawer-note">{prediction.details.diagnostic.summary}</p></div>
        <div className="drawer-actions"><Button onClick={() => { onClose(); navigate("/diagnosis"); }}>Open diagnostic view →</Button></div>
      </aside>
    </div>
  );
}

function SettingsPage({ models, apiOnline }: { models: ModelInfo[]; apiOnline: boolean | null }) {
  const [health, setHealth] = useState("checking");
  useEffect(() => { void api.health().then(() => setHealth("Healthy")).catch(() => setHealth("Unavailable")); }, []);
  return (
    <>
      <PageHeader screen="settings" />
      <div className="settings-layout">
        <Panel title="Application" subtitle="Runtime and service status">
          <div className="settings-list">
            <SettingRow label="Application" value="NEXUS 5G Network Intelligence" />
            <SettingRow label="Inference API" value={health} status={apiOnline ? "green" : apiOnline === false ? "red" : undefined} />
            <SettingRow label="Prediction history" value="Local SQLite database" />
            <SettingRow label="Telemetry mode" value="Offline / sample-based" />
            <SettingRow label="Network ingestion" value="Not configured" />
          </div>
        </Panel>
        <Panel title="Selected Model Bundles" subtitle="Loaded from the local results/models directory">
          <div className="settings-list">{models.map((model) => <SettingRow key={model.task} label={model.title} value={model.model_name ?? model.error ?? "Not available"} status={model.available ? "green" : "amber"} />)}</div>
        </Panel>
        <Panel title="Input Contract" subtitle="Expected by the REST prediction endpoint">
          <div className="contract-details"><div><span>Format</span><b>JSON</b></div><div><span>Shape</span><b>18 KPI channels × 128 timesteps</b></div><div><span>Endpoint</span><code>POST /api/predictions</code></div><button className="code-sample" onClick={() => downloadTemplate()}><code>Download JSON template</code><ArrowDownToLine size={14} /></button></div>
        </Panel>
        <div className="security-note"><ShieldCheck size={17} /><p><b>Local-first setup.</b> This development server binds to localhost by default and has no authentication. Do not expose it to a shared network or production users without adding authentication, TLS, and access controls.</p></div>
      </div>
    </>
  );
}

function SettingRow({ label, value, status }: { label: string; value: string; status?: "green" | "red" | "amber" }) {
  return <div className="setting-row"><span>{label}</span><strong>{status && <i className={`status-dot ${status}`} />}{value}</strong></div>;
}

export default App;
