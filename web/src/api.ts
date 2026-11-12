import type {
  KpiInput,
  ModelInfo,
  Overview,
  Prediction,
  PredictionSummary,
  RandomTestSample,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as
      | { detail?: string }
      | null;
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  overview: () => request<Overview>("/api/overview"),
  randomTestSample: (status: "any" | "normal" | "anomaly" = "any") =>
    request<RandomTestSample>(`/api/samples/random?status=${status}`),
  models: () => request<ModelInfo[]>("/api/models"),
  history: (query = "") =>
    request<PredictionSummary[]>(`/api/predictions${query ? `?${query}` : ""}`),
  prediction: (id: string) =>
    request<Prediction>(`/api/predictions/${encodeURIComponent(id)}`),
  predict: (KPIs: KpiInput) =>
    request<Prediction>("/api/predictions", {
      method: "POST",
      body: JSON.stringify({ KPIs }),
    }),
};

export const KPI_CHANNELS = [
  "DL_BLER",
  "DL_MCS",
  "DL_NumberOfPackets",
  "DL_Protocol",
  "Estimated_UL_Buffer",
  "PRB_Utilization_DL",
  "PRB_Utilization_UL",
  "PRBs_DL_Current",
  "PRBs_UL_Current",
  "RSRP",
  "RX_Bytes",
  "TX_Bytes",
  "UL_BLER",
  "UL_MCS",
  "UL_NPRB",
  "UL_NumberOfPackets",
  "UL_Protocol",
  "UL_SNR",
] as const;

export const CATEGORICAL_CHANNELS = new Set(["DL_Protocol", "UL_Protocol"]);

export function makeTemplate(): KpiInput {
  return Object.fromEntries(
    KPI_CHANNELS.map((channel) => [
      channel,
      Array.from({ length: 128 }, (_, step) =>
        CATEGORICAL_CHANNELS.has(channel)
          ? step < 64
            ? "TCP"
            : "UDP"
          : Number((Math.sin(step / 12) * 0.5 + 0.5).toFixed(4)),
      ),
    ]),
  );
}
