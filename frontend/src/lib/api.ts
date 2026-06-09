// API client for BTC Prediction Backend
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001"

export interface DashboardData {
  timestamp: string
  price: number
  price_formatted: string
  change_pct: number
  pc_prediction: {
    tcn?: number | null
    tcn_available?: boolean
    catboost: number | null
    lightgbm: number | null
    adapter?: number | null
    ensemble: number | null
    direction: "UP" | "DOWN"
  }
  vps_prediction: {
    catboost: number | null
    lightgbm: number | null
    ensemble: number | null
    direction: "UP" | "DOWN"
    available: boolean
  }
  data_rows_used: number
  backend?: {
    state: string
    last_train_time: string | null
    last_train_accuracy: number | null
    train_count: number
    sync_count: number
  }
}

export interface ChartPoint {
  time: string
  price: number
  pc_prediction: number | null
  vps_prediction: number | null
  tcn_prediction: number | null
  direction: "UP" | "DOWN"
}

export interface ForecastPoint {
  time: string
  price: number
  price_upper: number
  price_lower: number
  pc_prediction: number | null
  vps_prediction: number | null
  tcn_prediction: number | null
  direction: "UP" | "DOWN"
}

export interface ChartDataResponse {
  actual: ChartPoint[]
  forecast: ForecastPoint[]
  total_actual: number
  total_forecast: number
}

export interface Trade {
  time: string
  action: string
  price: number
  btc_amount: number
  capital: number
}

export interface SimulateResponse {
  initial_capital: number
  final_capital: number
  pnl: number
  pnl_pct: number
  total_trades: number
  trades: Trade[]
}

export async function fetchDashboard(): Promise<DashboardData> {
  const res = await fetch(`${API_BASE}/api/dashboard`, { cache: "no-store" })
  if (!res.ok) throw new Error(`Dashboard fetch failed: ${res.status}`)
  return res.json()
}

export async function fetchChartData(limit = 200): Promise<ChartDataResponse> {
  const res = await fetch(`${API_BASE}/api/chart-data?limit=${limit}`, { cache: "no-store" })
  if (!res.ok) throw new Error(`Chart fetch failed: ${res.status}`)
  return res.json()
}

export async function runSimulation(capital: number): Promise<SimulateResponse> {
  const res = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ initial_capital: capital }),
  })
  if (!res.ok) throw new Error(`Simulation failed: ${res.status}`)
  return res.json()
}

// ── Settings ──
export interface SettingsData {
  sync_interval: number
  retrain_cooldown: number
  accuracy_threshold: number
  trade_buy_threshold: number
  trade_sell_threshold: number
  unrealized_loss_threshold: number
  max_wrong_examples: number
  max_acc_log: number
  consecutive_wrong_retrain: number
  tcn_weight: number
  multi_tf_5m_weight: number
  lgb_biased_min: number
  lgb_biased_max: number
  chart_hist_limit: number
  chart_forecast_bars: number
  tcn_sequence_length: number
  auto_train_min_rows: number
  ml_iterations_max: number
  ml_iterations_min: number
  ml_learning_rate: number
  ml_depth: number
}

export async function fetchSettings(): Promise<{ settings: SettingsData; defaults: SettingsData }> {
  const res = await fetch(`${API_BASE}/api/settings`, { cache: "no-store" })
  if (!res.ok) throw new Error(`Settings fetch failed: ${res.status}`)
  return res.json()
}

export async function updateSettings(changes: Partial<SettingsData>): Promise<{ status: string; settings: SettingsData }> {
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  })
  if (!res.ok) throw new Error(`Settings update failed: ${res.status}`)
  return res.json()
}
