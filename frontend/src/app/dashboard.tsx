"use client"

import { useState, useEffect, useCallback, useRef, type PointerEvent } from "react"
import {
  Activity,
  BarChart3,
  Bot,
  BrainCircuit,
  CircleDollarSign,
  Clock3,
  Cpu,
  Database,
  RefreshCw,
  Server,
  TrendingDown,
  TrendingUp,
  Wallet,
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarSeparator,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { fetchDashboard, fetchChartData, type ChartPoint, type DashboardData, type ForecastPoint, type Trade } from "@/lib/api"
import { SettingsDialog } from "./settings-dialog"

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001"
const REFRESH_INTERVAL = 60_000

interface PortfolioData {
  active: boolean
  initial_capital: number
  current_value: number
  pnl: number
  pnl_pct: number
  position: string | null
  entry_price: number
  btc_amount: number
  latest_price: number
  trade_count: number
  last_action: string
  started_at: string | null
  recent_trades: Trade[]
}

async function fetchPortfolio(): Promise<PortfolioData | null> {
  try {
    const res = await fetch(`${API_BASE}/api/portfolio`, { cache: "no-store" })
    return res.ok ? res.json() : null
  } catch {
    return null
  }
}

async function startAutoTrade(capital: number) {
  const res = await fetch(`${API_BASE}/api/auto-trade/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ initial_capital: capital }),
  })
  return res.json()
}

async function stopAutoTrade() {
  const res = await fetch(`${API_BASE}/api/auto-trade/stop`, { method: "POST" })
  return res.json()
}

function formatRp(v?: number | null) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—"
  return `Rp ${v.toLocaleString("id-ID")}`
}

function AppSidebar({
  onRefresh,
  portfolio,
  tradeLoading,
  onStart,
  onStop,
  backendState,
}: {
  onRefresh: () => void
  portfolio: PortfolioData | null
  tradeLoading: boolean
  onStart: (capital: number) => void
  onStop: () => void
  backendState?: string
}) {
  return (
    <Sidebar variant="inset" collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center gap-3 px-2 py-2">
          <div className="flex size-10 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
            <BarChart3 className="size-5" />
          </div>
          <div className="grid flex-1 text-left text-sm leading-tight">
            <span className="truncate font-semibold tracking-tight">BTC Predictor</span>
            <span className="truncate text-xs text-muted-foreground">Indodax ML</span>
          </div>
        </div>
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Dashboard</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton onClick={onRefresh} tooltip="Refresh">
                  <RefreshCw />
                  <span>Refresh data</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SettingsDialog />
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Trading</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <AutoTradeControl portfolio={portfolio} loading={tradeLoading} onStart={onStart} onStop={onStop} />
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <div className="px-2 pb-2">
          <Card className="border-border/70 bg-muted/30">
            <CardContent className="p-3">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span
                  className={`size-2 rounded-full ${
                    backendState === "training"
                      ? "bg-yellow-400 animate-pulse"
                      : backendState === "trading"
                        ? "bg-emerald-400"
                        : backendState === "syncing"
                          ? "bg-blue-400"
                          : "bg-emerald-500"
                  }`}
                />
                <span className="truncate capitalize">{backendState || "idle"}</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}

function AutoTradeControl({ portfolio, loading, onStart, onStop }: { portfolio: PortfolioData | null; loading: boolean; onStart: (capital: number) => void; onStop: () => void }) {
  const [capital, setCapital] = useState("10000000")

  if (portfolio?.active) {
    const profit = portfolio.pnl >= 0
    return (
      <div className="space-y-2 px-2">
        <Card className="bg-card/70">
          <CardContent className="p-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Bot className="size-4 text-primary" /> Active
              </div>
              <Badge variant={profit ? "default" : "destructive"}>{profit ? "Profit" : "Loss"}</Badge>
            </div>
            <div className="grid gap-1 text-xs text-muted-foreground">
              <div>Nilai: {formatRp(portfolio.current_value)}</div>
              <div className={profit ? "text-emerald-400" : "text-red-400"}>P/L: {profit ? "+" : ""}{portfolio.pnl_pct}%</div>
              <div>{portfolio.trade_count} trade · {portfolio.position || "wait"}</div>
            </div>
          </CardContent>
        </Card>
        <Button variant="destructive" size="sm" className="w-full" onClick={onStop} disabled={loading}>
          Stop Trading
        </Button>
      </div>
    )
  }

  return (
    <Dialog>
      <DialogTrigger>
        <div className="flex h-8 items-center gap-2 rounded-md px-2 text-sm hover:bg-accent hover:text-accent-foreground cursor-pointer">
          <Bot className="size-4" />
          <span>Auto Trade</span>
        </div>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Auto-Trading Bot</DialogTitle>
          <DialogDescription>Simulasi real-time. BUY ≥70%, SELL ≤30%.</DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <Input type="number" value={capital} onChange={(e) => setCapital(e.target.value)} placeholder="Modal simulasi (Rp)" />
          <Button className="w-full" onClick={() => onStart(Number(capital))} disabled={loading}>
            {loading ? "Menjalankan..." : "Mulai Trading"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function MetricCard({ icon: Icon, label, value, sub, tone = "default" }: { icon: any; label: string; value: string; sub?: string; tone?: "default" | "green" | "red" | "amber" | "blue" }) {
  const toneClass =
    tone === "green" ? "text-emerald-400" : tone === "red" ? "text-red-400" : tone === "amber" ? "text-amber-400" : tone === "blue" ? "text-blue-400" : "text-foreground"
  return (
    <Card className="bg-card/80 backdrop-blur">
      <CardContent className="p-4">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className={`truncate text-xl font-semibold tracking-tight ${toneClass}`}>{value}</p>
            {sub && <p className="truncate text-[11px] text-muted-foreground">{sub}</p>}
          </div>
          <div className="rounded-xl bg-muted p-2 text-muted-foreground">
            <Icon className="size-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function PredictionPanel({ title, description, prediction, icon: Icon }: { title: string; description: string; prediction: DashboardData["pc_prediction"]; icon: any }) {
  const up = prediction.direction === "UP"
  return (
    <Card className="overflow-hidden bg-card/80 backdrop-blur">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Icon className="size-4 text-primary" /> {title}
            </CardTitle>
            <CardDescription className="text-xs">{description}</CardDescription>
          </div>
          <Badge variant={up ? "default" : "destructive"} className="shrink-0">
            {up ? "UP" : "DOWN"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-end justify-between">
          <div>
            <p className="text-xs text-muted-foreground">Model confidence</p>
            <p className="text-4xl font-bold tracking-tight">{prediction.ensemble ?? "—"}%</p>
          </div>
          <div className={up ? "text-emerald-400" : "text-red-400"}>{up ? <TrendingUp className="size-9" /> : <TrendingDown className="size-9" />}</div>
        </div>
        <Separator />
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div>
            <p className="text-muted-foreground">Ensemble</p>
            <p className="font-mono font-medium">{prediction.ensemble ?? "—"}%</p>
          </div>
          <div>
            <p className="text-muted-foreground">CatBoost</p>
            <p className="font-mono font-medium">{prediction.catboost?.toFixed(1) ?? "—"}%</p>
          </div>
          <div>
            <p className="text-muted-foreground">LightGBM</p>
            <p className="font-mono font-medium">{prediction.lightgbm?.toFixed(1) ?? "—"}%</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function PriceChart({ points, forecast }: { points: ChartPoint[]; forecast: ForecastPoint[] }) {
  const [range, setRange] = useState<{ start: number; end: number } | null>(null)
  const dragRef = useRef<{ x: number; range: { start: number; end: number } } | null>(null)
  const [dragging, setDragging] = useState(false)
  const [pricePulse, setPricePulse] = useState<"up" | "down" | null>(null)
  const lastPriceRef = useRef<number | null>(null)

  const latestPointPrice = points?.[points.length - 1]?.price
  useEffect(() => {
    if (typeof latestPointPrice !== "number") return
    const prev = lastPriceRef.current
    if (prev !== null && latestPointPrice !== prev) {
      setPricePulse(latestPointPrice > prev ? "up" : "down")
      const id = window.setTimeout(() => setPricePulse(null), 1600)
      lastPriceRef.current = latestPointPrice
      return () => window.clearTimeout(id)
    }
    lastPriceRef.current = latestPointPrice
  }, [latestPointPrice])

  if (!points?.length) {
    return (
      <div className="flex h-72 items-center justify-center">
        <div className="space-y-3 text-center">
          <Skeleton className="mx-auto h-8 w-44" />
          <p className="text-sm text-muted-foreground">Menunggu data grafik...</p>
        </div>
      </div>
    )
  }

  const actual = points.map((p) => ({ time: p.time, price: p.price, forecast: null as number | null, pc: p.pc_prediction, vps: p.vps_prediction, tcn: p.tcn_prediction }))
  const pred = forecast.map((f) => ({ time: f.time, price: null as number | null, forecast: f.price, pc: f.pc_prediction, vps: f.vps_prediction, tcn: f.tcn_prediction }))
  const lastActual = actual[actual.length - 1]
  const bridge = lastActual
    ? { ...lastActual, forecast: lastActual.price }
    : null
  const data = bridge ? [...actual, bridge, ...pred] : [...actual, ...pred]
  const safeRange = range && range.start >= 0 && range.end < data.length && range.start < range.end ? range : null
  const visibleData = safeRange ? data.slice(safeRange.start, safeRange.end + 1) : data
  const visiblePrices = visibleData.flatMap((d) => [d.price, d.forecast]).filter((v): v is number => typeof v === "number" && Number.isFinite(v))
  const domain: [number, number] = visiblePrices.length
    ? [Math.min(...visiblePrices) * 0.999, Math.max(...visiblePrices) * 1.001]
    : [0, 1]
  const fmtPrice = (v: number) => (v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : `${(v / 1e6).toFixed(0)}M`)

  const zoomIn = () => {
    const current = safeRange ?? { start: 0, end: data.length - 1 }
    const len = current.end - current.start + 1
    if (len <= 8) return
    const cut = Math.max(1, Math.floor(len * 0.25))
    setRange({ start: current.start + cut, end: current.end - cut })
  }
  const zoomOut = () => {
    const current = safeRange ?? { start: 0, end: data.length - 1 }
    const len = current.end - current.start + 1
    const add = Math.max(2, Math.floor(len * 0.5))
    const next = { start: Math.max(0, current.start - add), end: Math.min(data.length - 1, current.end + add) }
    if (next.start === 0 && next.end === data.length - 1) setRange(null)
    else setRange(next)
  }
  const resetZoom = () => setRange(null)

  const startPan = (e: PointerEvent<HTMLDivElement>) => {
    // First drag opens a movable window, then mouse/touch can pan it.
    const defaultWindow = Math.max(20, Math.min(data.length, Math.floor(data.length * 0.6)))
    const current = safeRange ?? { start: Math.max(0, data.length - defaultWindow), end: data.length - 1 }
    if (!safeRange && data.length > defaultWindow) setRange(current)
    dragRef.current = { x: e.clientX, range: current }
    e.currentTarget.setPointerCapture?.(e.pointerId)
    setDragging(true)
  }

  const movePan = (e: PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) return
    e.preventDefault()
    const x = e.clientX
    const start = dragRef.current.x
    const current = dragRef.current.range
    const width = e.currentTarget?.clientWidth || 1
    const len = current.end - current.start + 1
    const shift = Math.round(((start - x) / width) * len)
    if (shift === 0) return
    const maxStart = Math.max(0, data.length - len)
    const nextStart = Math.max(0, Math.min(maxStart, current.start + shift))
    setRange({ start: nextStart, end: nextStart + len - 1 })
  }

  const endPan = (e?: PointerEvent<HTMLDivElement>) => {
    if (e) e.currentTarget.releasePointerCapture?.(e.pointerId)
    dragRef.current = null
    setDragging(false)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2 px-1">
        <p className="text-xs text-muted-foreground">
          {dragging ? "Geser chart..." : safeRange ? `Zoom: ${safeRange.end - safeRange.start + 1}/${data.length} titik • drag untuk geser` : `Full ${data.length} titik • zoom lalu drag`}
        </p>
        <div className="flex gap-1">
          <Button variant="outline" size="sm" className="h-7 px-2" onClick={zoomIn}>+</Button>
          <Button variant="outline" size="sm" className="h-7 px-2" onClick={zoomOut}>−</Button>
          <Button variant="outline" size="sm" className="h-7 px-2" onClick={resetZoom}>Reset</Button>
        </div>
      </div>
      <div
        className={`relative h-[330px] sm:h-[460px] w-full select-none overflow-hidden rounded-lg border border-transparent transition-all duration-500 ${
          dragging ? "cursor-grabbing border-primary/40 ring-2 ring-primary/15" : "cursor-grab"
        } ${pricePulse === "up" ? "bg-emerald-500/5 shadow-[0_0_28px_rgba(16,185,129,0.20)]" : pricePulse === "down" ? "bg-red-500/5 shadow-[0_0_28px_rgba(239,68,68,0.20)]" : ""}`}
        style={{ touchAction: "none" }}
        onPointerDown={startPan}
        onPointerMove={movePan}
        onPointerUp={endPan}
        onPointerCancel={endPan}
        onPointerLeave={endPan}
      >
      {pricePulse && (
        <div className={`pointer-events-none absolute inset-0 z-10 animate-pulse rounded-lg border ${pricePulse === "up" ? "border-emerald-400/70" : "border-red-400/70"}`} />
      )}
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={visibleData} margin={{ left: 0, right: 12, top: 12, bottom: 0 }}>
          <defs>
            <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.30} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
          <XAxis dataKey="time" fontSize={10} tickLine={false} axisLine={false} minTickGap={28} />
          <YAxis yAxisId="price" fontSize={10} tickLine={false} axisLine={false} width={56} tickFormatter={fmtPrice} domain={domain} />
          <YAxis yAxisId="confidence" orientation="right" hide domain={[0, 100]} />
          <Tooltip
            contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 12, fontSize: 12 }}
            formatter={(v: any, n: any) => {
              if (n === "pc") return [`${Number(v).toFixed(2)}%`, "CatBoost+LightGBM"]
              if (n === "tcn") return [`${Number(v).toFixed(2)}%`, "TCN"]
              return [`Rp ${Number(v).toLocaleString("id-ID")}`, n === "price" ? "Aktual" : "Prediksi"]
            }}
          />
          <Area
            yAxisId="price"
            type="monotone"
            dataKey="price"
            stroke="#3b82f6"
            fill="url(#priceFill)"
            strokeWidth={2.6}
            dot={false}
            isAnimationActive
            animationDuration={650}
            animationEasing="ease-out"
          />
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="forecast"
            stroke="#fbbf24"
            strokeWidth={2.1}
            strokeDasharray="6 5"
            dot={false}
            connectNulls
            isAnimationActive
            animationDuration={650}
            animationEasing="ease-out"
          />
          <Line
            yAxisId="confidence"
            type="monotone"
            dataKey="pc"
            stroke="#a78bfa"
            strokeWidth={1.9}
            strokeDasharray="8 4"
            dot={false}
            isAnimationActive
            animationDuration={650}
            animationEasing="ease-out"
          />
          <Line
            yAxisId="confidence"
            type="monotone"
            dataKey="tcn"
            stroke="#06b6d4"
            strokeWidth={1.8}
            strokeDasharray="2 6"
            dot={false}
            isAnimationActive
            animationDuration={650}
            animationEasing="ease-out"
          />
        </AreaChart>
      </ResponsiveContainer>
      </div>
    </div>
  )
}

function PortfolioCard({ portfolio }: { portfolio: PortfolioData | null }) {
  if (!portfolio) return null
  const profit = portfolio.pnl >= 0
  return (
    <Card className="bg-card/80 backdrop-blur">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Wallet className="size-4 text-primary" /> Portfolio Simulasi
        </CardTitle>
        <CardDescription>Status auto-trading saat ini</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-4">
        <MetricCard icon={Wallet} label="Modal" value={formatRp(portfolio.initial_capital)} />
        <MetricCard icon={CircleDollarSign} label="Nilai kini" value={formatRp(portfolio.current_value)} />
        <MetricCard icon={profit ? TrendingUp : TrendingDown} label="P/L" value={`${profit ? "+" : ""}${portfolio.pnl_pct}%`} sub={formatRp(portfolio.pnl)} tone={profit ? "green" : "red"} />
        <MetricCard icon={Bot} label="Posisi" value={portfolio.position || "WAIT"} sub={`${portfolio.trade_count} trade`} tone="blue" />
      </CardContent>
    </Card>
  )
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [chart, setChart] = useState<ChartPoint[]>([])
  const [forecast, setForecast] = useState<ForecastPoint[]>([])
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null)
  const [tradeLoading, setTradeLoading] = useState(false)
  const [error, setError] = useState("")
  const [wibTime, setWibTime] = useState("")

  useEffect(() => {
    const tick = () => {
      setWibTime(
        new Intl.DateTimeFormat("id-ID", {
          timeZone: "Asia/Jakarta",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          day: "2-digit",
          month: "short",
        }).format(new Date()) + " WIB"
      )
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  const refresh = useCallback(async () => {
    try {
      const [dashboard, chartData] = await Promise.all([fetchDashboard(), fetchChartData()])
      setData(dashboard)
      setChart(chartData.actual)
      setForecast(chartData.forecast || [])
      setError("")
      fetchPortfolio().then((p) => {
        if (p) setPortfolio(p)
      }).catch(() => {})
    } catch (e: any) {
      setError(e.message || "Fetch failed")
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, REFRESH_INTERVAL)
    return () => clearInterval(id)
  }, [refresh])

  const start = async (capital: number) => {
    setTradeLoading(true)
    try {
      await startAutoTrade(capital)
      await refresh()
    } finally {
      setTradeLoading(false)
    }
  }

  const stop = async () => {
    setTradeLoading(true)
    try {
      await stopAutoTrade()
      await refresh()
    } finally {
      setTradeLoading(false)
    }
  }

  const up = (data?.change_pct ?? 0) >= 0
  const tcn = data?.pc_prediction?.tcn
  const ensemble = data?.pc_prediction?.ensemble

  return (
    <SidebarProvider>
      <AppSidebar onRefresh={refresh} portfolio={portfolio} tradeLoading={tradeLoading} onStart={start} onStop={stop} backendState={data?.backend?.state} />
      <SidebarInset className="min-w-0 overflow-x-hidden">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b bg-background/85 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <SidebarTrigger />
          <Separator orientation="vertical" className="h-6" />
          <div className="flex min-w-0 flex-1 items-center justify-between gap-3">
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold">BTC/IDR Prediction Dashboard</h1>
              <p className="hidden truncate text-xs text-muted-foreground sm:block">TCN · CatBoost · LightGBM · Auto Trading</p>
            </div>
            <Badge variant="outline" className="hidden gap-1 whitespace-nowrap min-[420px]:inline-flex">
              <Clock3 className="size-3" /> {wibTime}
            </Badge>
          </div>
        </header>

        <main className="min-h-screen bg-[radial-gradient(circle_at_top_right,rgba(113,50,245,0.12),transparent_35%),radial-gradient(circle_at_bottom_left,rgba(20,158,97,0.08),transparent_30%)] p-3 sm:p-6 overflow-x-hidden">
          <div className="mx-auto max-w-7xl space-y-6">
            {error && (
              <Card className="border-red-500/30 bg-red-500/5">
                <CardContent className="p-3 text-sm text-red-400">{error}</CardContent>
              </Card>
            )}

            <section className="grid gap-4 lg:grid-cols-4">
              <Card className="overflow-hidden border-primary/20 bg-card/90 backdrop-blur lg:col-span-2">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <p className="text-sm text-muted-foreground">Harga aktual</p>
                      {data ? (
                        <h2 className="mt-1 break-words text-2xl font-bold tracking-tight sm:text-5xl">{data.price_formatted}</h2>
                      ) : (
                        <Skeleton className="mt-2 h-10 w-64" />
                      )}
                      <p className={`mt-3 flex items-center gap-1 text-sm font-medium ${up ? "text-emerald-400" : "text-red-400"}`}>
                        {up ? <TrendingUp className="size-4" /> : <TrendingDown className="size-4" />}
                        {Math.abs(data?.change_pct ?? 0).toFixed(4)}%
                      </p>
                    </div>
                    <div className="rounded-3xl bg-primary/10 p-4 text-primary shadow-inner">
                      <CircleDollarSign className="size-9" />
                    </div>
                  </div>
                </CardContent>
              </Card>

              <MetricCard icon={BrainCircuit} label="Model confidence" value={ensemble !== undefined && ensemble !== null ? `${ensemble}%` : "—"} sub={tcn !== undefined && tcn !== null ? "TCN + tree ensemble" : "CatBoost + LightGBM"} tone="amber" />
              <MetricCard icon={Database} label="Data rows" value={`${data?.data_rows_used ?? "—"}`} sub="Live cache" tone="blue" />
            </section>

            {portfolio?.active && <PortfolioCard portfolio={portfolio} />}

            <Tabs defaultValue="overview" className="space-y-4">
              <TabsList className="grid w-full grid-cols-3 sm:w-[420px]">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="models">Models</TabsTrigger>
                <TabsTrigger value="chart">Chart</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-4">
                <section className="grid gap-4 lg:grid-cols-3">
                  <MetricCard icon={Activity} label="Ensemble PC" value={ensemble !== undefined && ensemble !== null ? `${ensemble}%` : "—"} sub={data?.pc_prediction?.direction || "waiting"} tone={up ? "green" : "red"} />
                  <MetricCard icon={Server} label="Backend" value={data?.backend?.state || "idle"} sub={`train: ${data?.backend?.last_train_accuracy ?? "—"}%`} />
                  <MetricCard icon={Bot} label="Auto Trade" value={portfolio?.active ? "ACTIVE" : "OFF"} sub={portfolio?.active ? `${portfolio.trade_count} trade` : "manual start"} tone={portfolio?.active ? "green" : "default"} />
                </section>

                <Card className="bg-card/80 backdrop-blur">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <BarChart3 className="size-4 text-primary" /> Chart harga & prediksi
                    </CardTitle>
                    <CardDescription>Biru = harga, kuning = forecast harga, ungu putus = CatBoost+LightGBM, cyan putus = TCN</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <PriceChart points={chart} forecast={forecast} />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="models" className="space-y-4">
                {data ? (
                  <section className="grid gap-4 lg:grid-cols-2">
                    <PredictionPanel title="Model PC" description="TCN + CatBoost + LightGBM" prediction={data.pc_prediction} icon={Cpu} />
                    <PredictionPanel title="Model VPS" description="Live-trained Indodax model" prediction={data.vps_prediction} icon={Server} />
                  </section>
                ) : (
                  <div className="grid gap-4 lg:grid-cols-2">
                    <Skeleton className="h-64" />
                    <Skeleton className="h-64" />
                  </div>
                )}
              </TabsContent>

              <TabsContent value="chart">
                <Card className="bg-card/80 backdrop-blur">
                  <CardHeader>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <CardTitle className="flex items-center gap-2 text-base">
                          <BarChart3 className="size-4 text-primary" /> Full chart
                        </CardTitle>
                        <CardDescription>{forecast.length} forecast points</CardDescription>
                      </div>
                      <Button variant="outline" size="sm" onClick={refresh} className="gap-2">
                        <RefreshCw className="size-4" /> Refresh
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <PriceChart points={chart} forecast={forecast} />
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
