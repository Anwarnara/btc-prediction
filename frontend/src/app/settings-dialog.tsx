"use client"

import { useState, useEffect } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { fetchSettings, updateSettings, SettingsData } from "@/lib/api"

const LABELS: Record<string, string> = {
  sync_interval: "Sync interval (detik)",
  retrain_cooldown: "Retrain cooldown (detik)",
  accuracy_threshold: "Akurasi threshold (%)",
  trade_buy_threshold: "Threshold BUY (%)",
  trade_sell_threshold: "Threshold SELL (%)",
  unrealized_loss_threshold: "Unrealized loss (%)",
  max_wrong_examples: "Max wrong examples",
  max_acc_log: "Max accuracy log",
  consecutive_wrong_retrain: "Consecutive wrong → retrain",
  tcn_weight: "TCN weight",
  multi_tf_5m_weight: "5m TF weight",
  lgb_biased_min: "LGBM bias min (%)",
  lgb_biased_max: "LGBM bias max (%)",
  chart_hist_limit: "Chart history bars",
  chart_forecast_bars: "Chart forecast bars",
  tcn_sequence_length: "TCN sequence",
  auto_train_min_rows: "Min rows training",
  ml_iterations_max: "ML max iterations",
  ml_iterations_min: "ML min iterations",
  ml_learning_rate: "ML learning rate",
  ml_depth: "ML depth",
}

const CATEGORIES: Record<string, string> = {
  sync_interval: "Sync & Training", auto_train_min_rows: "Sync & Training",
  retrain_cooldown: "Sync & Training", accuracy_threshold: "Sync & Training",
  ml_iterations_max: "Sync & Training", ml_iterations_min: "Sync & Training",
  ml_learning_rate: "Sync & Training", ml_depth: "Sync & Training",
  trade_buy_threshold: "Trading", trade_sell_threshold: "Trading",
  unrealized_loss_threshold: "Trading", max_wrong_examples: "Trading",
  max_acc_log: "Trading", consecutive_wrong_retrain: "Trading",
  tcn_weight: "Ensemble", multi_tf_5m_weight: "Ensemble",
  lgb_biased_min: "Ensemble", lgb_biased_max: "Ensemble",
  chart_hist_limit: "Chart", chart_forecast_bars: "Chart",
  tcn_sequence_length: "Ensemble",
}

export function SettingsDialog() {
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [changed, setChanged] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState("")
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (open) fetchSettings().then(d => setSettings(d.settings)).catch(() => setMsg("Gagal load"))
  }, [open])

  const handleChange = (key: string, val: string) => {
    setChanged(prev => ({ ...prev, [key]: val }))
  }

  const handleSave = async () => {
    if (!Object.keys(changed).length) return
    setSaving(true)
    const updates: Partial<SettingsData> = {}
    for (const [k, v] of Object.entries(changed)) updates[k as keyof SettingsData] = parseFloat(v) as any
    try {
      const res = await updateSettings(updates)
      setSettings(res.settings)
      setChanged({})
      setMsg("✅ Tersimpan!")
    } catch { setMsg("❌ Gagal") }
    setSaving(false)
    setTimeout(() => setMsg(""), 3000)
  }

  const cats = new Set(Object.values(CATEGORIES))
  const groups: Record<string, string[]> = {}
  for (const cat of cats) groups[cat] = Object.keys(LABELS).filter(k => CATEGORIES[k] === cat)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger>
        <div className="flex items-center gap-2 h-9 px-4 text-sm rounded-md hover:bg-muted/50 transition-colors cursor-pointer">
          ⚙️ Settings
        </div>
      </DialogTrigger>
      <DialogContent className="max-w-lg max-h-[85vh]">
        <DialogHeader>
          <DialogTitle>⚙️ Settings</DialogTitle>
          <DialogDescription>Ubah parameter — real-time tanpa restart. {msg && <span className="block mt-1 text-green-400 text-xs">{msg}</span>}</DialogDescription>
        </DialogHeader>
        {!settings ? (
          <div className="text-center py-8 text-sm text-muted-foreground">Loading...</div>
        ) : (
          <div className="overflow-y-auto max-h-[55vh] pr-4">
            <div className="space-y-5">
              {Object.entries(groups).map(([cat, keys]) => (
                <div key={cat}>
                  <h4 className="text-[11px] font-semibold text-muted-foreground mb-2 uppercase tracking-wider">{cat}</h4>
                  <Separator className="mb-2" />
                  <div className="space-y-2.5">
                    {keys.map(key => (
                      <div key={key} className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground w-36 shrink-0">{LABELS[key]}</span>
                        <Input className="h-8 text-xs font-mono flex-1"
                          value={changed[key] !== undefined ? changed[key] : String(settings[key as keyof SettingsData])}
                          onChange={e => handleChange(key, e.target.value)} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="flex justify-end gap-2 pt-2">
          <Button variant="outline" size="sm" onClick={() => setChanged({})} disabled={!Object.keys(changed).length}>Reset</Button>
          <Button size="sm" onClick={handleSave} disabled={!Object.keys(changed).length || saving}>{saving ? "..." : "💾 Save"}</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
