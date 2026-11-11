export type ScreenId =
  | "overview"
  | "detection"
  | "classification"
  | "localization"
  | "diagnosis"
  | "monitoring"
  | "history"
  | "settings";

export type KpiInput = Record<string, Array<number | string | null>>;

export interface ScoreItem {
  label: string;
  score: number;
}

export interface TaskPrediction {
  model_name: string;
  model_version: string;
  prediction: string | number | number[] | string[];
  confidence: number;
  score?: number;
  scores: ScoreItem[];
}

export interface PredictionSummary {
  id: string;
  created_at: string;
  anomaly_status: "normal" | "anomaly";
  predicted_type: string;
  detection_score: number;
  confidence: number;
  affected_kpis: string[];
  inference_ms: number;
}

export interface Prediction extends PredictionSummary {
  details: {
    tasks: Record<string, TaskPrediction>;
    localization: {
      timeline: number[];
      first_anomalous_timestep: number | null;
      last_anomalous_timestep: number | null;
      anomalous_timesteps: number;
      segments: Array<{ start: number; end: number }>;
    };
    diagnostic: {
      summary: string;
      hypotheses: string[];
      observed: string[];
      notice: string;
    };
    input_channels: string[];
  };
  input?: { KPIs: KpiInput };
}

export interface ModelInfo {
  task: string;
  title: string;
  model_name: string | null;
  version: string | null;
  available: boolean;
  metrics: Record<string, number>;
  metric_source?: string;
  error?: string;
}

export interface Overview {
  data_source: string;
  live_telemetry: boolean;
  total_predictions: number;
  anomalies: number;
  average_inference_ms: number;
  models_available: number;
  models: ModelInfo[];
  categories: Array<{ name: string; count: number }>;
  recent_predictions: PredictionSummary[];
  dataset_baseline: { name: string; split: string; rows: number; note: string };
}
