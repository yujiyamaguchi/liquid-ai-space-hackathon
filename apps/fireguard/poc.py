"""
FireGuard /poc — 発火前スペクトル分離確認スクリプト (v2: NDMI 主指標版)
============================================================
科学的根拠 (2026-04-26 改訂):

  主指標: NDMI = (B8A - B11) / (B8A + B11)  [Gao 1996, 現代命名では NDMI]
    - 葉内液体水分量 (EWT) の最良リモートセンシングプロキシ
    - B11 (1610nm) は液体水の吸収帯に位置し LFMC 推定に最感度
    - チャパラル (Chamise) LFMC 推定: R²=0.76, MAE=9.68% [Myoung et al. 2018]
    - 発火前乾燥ストレス指標として最高推奨 [Gemini Deep Research 2026-04-26]

  副指標: NBR = (B8A - B12) / (B8A + B12)  [Key & Benson 2006 — pre-fire baseline]
    - 燃料構造 + 水分の複合指標。B12 は土壌・枯死燃料への反応があるため副次扱い

  LFMC 臨界値 (チャパラル): < 77-80% で大規模火災リスク急増 [Santa Monica Mts 研究]
  14-28日リードタイムの生理的根拠: クロロフィル 7-75%、アントシアニン 38-100% の減少

設計変更 vs v1:
  - 旧: NDWI = (B08 - B11)/(B08 + B11) → 新: NDMI = (B8A - B11)/(B8A + B11) (B8A 使用に修正)
  - 旧: NBR が主指標  → 新: NDMI が主指標、NBR は副指標
  - 旧: NEG = サクラメントバレー農地 (植生タイプ交絡)
       → 新: NEG = チャパラル優勢エリア限定 (南カリフォルニア / Bay Area hills)
  - NDMI seasonal anomaly を任意取得 (SimSat 過去データが存在する場合のみ)

CLAUDE.md /poc 完了条件:
  ① SimSat が指定座標・日時の Sentinel-2 データを返せること
  ② NDMI が POS(発火前14〜28日) と NEG(チャパラル非発火) で統計的に分離できること

実行:
  cd apps/fireguard
  uv run python poc.py

出力:
  data/poc/results.csv     — 各シーンの指標値
  data/poc/figures/        — NDMI / NBR 分布図
  data/poc/images/         — 疑似カラー画像 (SWIR22=R, B8A=G, SWIR16=B)
"""
from __future__ import annotations

import base64
import csv
import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
ENV_FILE = ROOT.parent.parent / ".env"
SAVE_DIR = ROOT / "data" / "poc"

SIMSAT_BASE = "http://localhost:9005"
SIZE_KM = 20.0
WINDOW_SECONDS = 5 * 86400   # Sentinel-2 revisit 3〜5日 → ±5日でシーン検索
BANDS = ["nir08", "nir", "swir22", "swir16", "red"]
# ch0=nir08(B8A,865nm), ch1=nir(B08,842nm), ch2=swir22(B12,2190nm),
# ch3=swir16(B11,1610nm), ch4=red(B04,665nm)

LEAD_DAYS = [14, 21, 28]
ANOMALY_YEARS = [1, 2]  # seasonal anomaly 用: 1年前・2年前の同日 NDMI を取得


# ─────────────────────────────────────────────────────────
# .env 読み込み
# ─────────────────────────────────────────────────────────

def _load_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()
FIRMS_API_KEY = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("FIRMS_API_KEY", "")


# ─────────────────────────────────────────────────────────
# GT データ定義 — カリフォルニア地中海性気候帯
# ─────────────────────────────────────────────────────────

# POS: 発火座標・発火日 (VIIRS/CAL FIRE データベース準拠)
# vegetation_type: "chaparral" = チャパラル/低木優勢, "conifer" = 針葉樹混合
POS_EVENTS: list[dict] = [
    {"name": "CZU Lightning Complex", "lat": 37.10, "lon": -122.30, "fire_date": "2020-08-16", "veg": "chaparral"},
    {"name": "SCU Lightning Complex", "lat": 37.50, "lon": -121.50, "fire_date": "2020-08-18", "veg": "chaparral"},
    {"name": "LNU Lightning Complex", "lat": 38.80, "lon": -122.40, "fire_date": "2020-08-17", "veg": "chaparral"},
    {"name": "Thomas Fire",           "lat": 34.40, "lon": -119.10, "fire_date": "2017-12-04", "veg": "chaparral"},
    {"name": "Kincade Fire",          "lat": 38.80, "lon": -122.70, "fire_date": "2019-10-23", "veg": "chaparral"},
    {"name": "Creek Fire",            "lat": 37.20, "lon": -119.30, "fire_date": "2020-09-04", "veg": "conifer"},
    {"name": "Carr Fire",             "lat": 40.60, "lon": -122.30, "fire_date": "2018-07-23", "veg": "conifer"},
    {"name": "Mendocino Complex",     "lat": 39.30, "lon": -122.90, "fire_date": "2018-07-27", "veg": "conifer"},
    {"name": "Dixie Fire",            "lat": 39.90, "lon": -121.20, "fire_date": "2021-07-13", "veg": "conifer"},
    {"name": "Caldor Fire",           "lat": 38.70, "lon": -120.00, "fire_date": "2021-08-14", "veg": "conifer"},
]

# NEG: チャパラル優勢エリア限定 (南カリフォルニア/Bay Area hills)
# 旧設計のサクラメントバレー農地は植生タイプ交絡のため廃止
# 同カレンダー月・火災非経験地点 (FIRMS SP fire-free で確認)
NEG_CANDIDATES: list[dict] = [
    # 南カリフォルニア — 夏〜秋 (POS の主要季節に合わせる)
    {"name": "Santa Monica Mts Aug2019",       "lat": 34.08, "lon": -118.75, "date": "2019-08-01"},
    {"name": "Santa Monica Mts Sep2019",       "lat": 34.09, "lon": -118.70, "date": "2019-09-01"},
    {"name": "Angeles NF foothills Jul2019",   "lat": 34.25, "lon": -117.75, "date": "2019-07-15"},
    {"name": "San Bernardino foothills Aug2022","lat": 34.10, "lon": -117.30, "date": "2022-08-01"},
    {"name": "Riverside chaparral Sep2022",    "lat": 33.75, "lon": -117.00, "date": "2022-09-01"},
    {"name": "Ventura hills Oct2022",          "lat": 34.30, "lon": -119.20, "date": "2022-10-15"},
    # Bay Area hills — チャパラル丘陵 (POS の北カリフォルニアに対応)
    {"name": "Diablo Range Aug2019",           "lat": 37.15, "lon": -121.50, "date": "2019-08-01"},
    {"name": "Diablo Range Sep2019",           "lat": 37.20, "lon": -121.55, "date": "2019-09-01"},
    {"name": "Mt Diablo Jul2022",              "lat": 37.88, "lon": -121.92, "date": "2022-07-15"},
    {"name": "Sonoma hills Oct2022",           "lat": 38.30, "lon": -122.50, "date": "2022-10-01"},
]


# ─────────────────────────────────────────────────────────
# SimSat クライアント
# ─────────────────────────────────────────────────────────

def simsat_health() -> bool:
    try:
        r = requests.get(f"{SIMSAT_BASE}/", timeout=5)
        return r.ok
    except Exception:
        return False


def fetch_scene(lat: float, lon: float, timestamp_iso: str) -> Optional[np.ndarray]:
    """SimSat から5バンド配列を取得。取得不可の場合 None を返す。

    Returns: (H, W, 5) float32 [0, 1]
      ch0=nir08(B8A), ch1=nir(B08), ch2=swir22(B12), ch3=swir16(B11), ch4=red(B04)
    """
    params = {
        "lat": lat,
        "lon": lon,
        "timestamp": timestamp_iso,
        "spectral_bands": BANDS,
        "size_km": SIZE_KM,
        "return_type": "array",
        "window_seconds": WINDOW_SECONDS,
    }
    try:
        r = requests.get(
            f"{SIMSAT_BASE}/data/image/sentinel",
            params=params,
            timeout=60,
        )
        if not r.ok:
            print(f"HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
    except Exception as e:
        print(f"error: {e}")
        return None

    meta = data.get("sentinel_metadata") or data
    if not meta.get("image_available", False):
        return None

    img_entry = data.get("image")
    if img_entry is None:
        return None

    if isinstance(img_entry, dict) and "image" in img_entry:
        img_meta = img_entry.get("metadata", {})
        shape = img_meta.get("shape")
        dtype = img_meta.get("dtype", "uint16")
        raw = base64.b64decode(img_entry["image"])
        arr = np.frombuffer(raw, dtype=np.dtype(dtype))
        if shape:
            arr = arr.reshape(shape)            # (C, H, W)
            arr = np.transpose(arr, (1, 2, 0))  # → (H, W, C)
        if np.issubdtype(np.dtype(dtype), np.integer):
            arr = arr.astype(np.float32) / np.iinfo(np.dtype(dtype)).max
        else:
            arr = arr.astype(np.float32)
        return arr

    return None


# ─────────────────────────────────────────────────────────
# スペクトル指標
# ─────────────────────────────────────────────────────────

def compute_indices(arr: np.ndarray) -> dict:
    """5バンド配列からスペクトル指標を計算する。

    arr: (H, W, 5)
      ch0=nir08(B8A,865nm), ch1=nir(B08,842nm),
      ch2=swir22(B12,2190nm), ch3=swir16(B11,1610nm), ch4=red(B04,665nm)

    NDMI = (B8A - B11) / (B8A + B11)   [Gao 1996; 現代命名 NDMI]
           主指標。葉内液体水分量 (EWT) の最良プロキシ。
           B11 (1610nm) は液体水吸収帯。低値 = 乾燥ストレス = 高火災リスク。
           チャパラル LFMC 推定 R²=0.76, MAE≈10% [Myoung et al. 2018]

    NBR  = (B8A - B12) / (B8A + B12)   [Key & Benson 2006 — pre-fire baseline]
           副指標。燃料構造 + 水分の複合指標。
           B12 は土壌・枯死燃料への反応が大きいためノイズが多い。

    NDVI = (B08 - B04) / (B08 + B04)
           植生活性度の参照指標。チャパラル識別 (0.2〜0.4) に使用。
    """
    nir08  = arr[:, :, 0].astype(np.float64)  # B8A (865nm)
    nir    = arr[:, :, 1].astype(np.float64)  # B08 (842nm)
    swir22 = arr[:, :, 2].astype(np.float64)  # B12 (2190nm)
    swir16 = arr[:, :, 3].astype(np.float64)  # B11 (1610nm)
    red    = arr[:, :, 4].astype(np.float64)  # B04 (665nm)

    eps = 1e-8

    # 主指標: NDMI (B8A/B11) — v1 では "NDWI" と誤称、かつ B08 を使っていたため修正
    ndmi_map = (nir08 - swir16) / (nir08 + swir16 + eps)

    # 副指標: NBR (B8A/B12)
    nbr_map  = (nir08 - swir22) / (nir08 + swir22 + eps)

    # 参照指標: NDVI (B08/B04)
    ndvi_map = (nir - red) / (nir + red + eps)

    # 有効ピクセル: 飽和・全黒を除外
    valid = (nir08 > 0.01) & (nir08 < 0.99) & (swir16 < 0.99) & (swir22 < 0.99)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return {}

    return {
        "ndmi_mean":   float(ndmi_map[valid].mean()),
        "ndmi_p10":    float(np.percentile(ndmi_map[valid], 10)),
        "nbr_mean":    float(nbr_map[valid].mean()),
        "nbr_p10":     float(np.percentile(nbr_map[valid], 10)),
        "ndvi_mean":   float(ndvi_map[valid].mean()),
        "swir16_mean": float(swir16[valid].mean()),
        "nir08_mean":  float(nir08[valid].mean()),
        "n_valid_px":  n_valid,
    }


# ─────────────────────────────────────────────────────────
# NDMI seasonal anomaly (任意)
# ─────────────────────────────────────────────────────────

def fetch_ndmi_anomaly(lat: float, lon: float, target_date_iso: str) -> Optional[float]:
    """対象日の NDMI から過去 1〜2 年同日平均を引いた seasonal anomaly を返す。
    SimSat に過去データがない場合は None を返す (PoC の主判定には使わない)。
    """
    target_dt = datetime.strptime(target_date_iso[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    historical_ndmi: list[float] = []

    for y in ANOMALY_YEARS:
        hist_dt = target_dt - timedelta(days=365 * y)
        hist_iso = hist_dt.strftime("%Y-%m-%dT12:00:00Z")
        arr = fetch_scene(lat, lon, hist_iso)
        if arr is None:
            continue
        idx = compute_indices(arr)
        if idx:
            historical_ndmi.append(idx["ndmi_mean"])

    if not historical_ndmi:
        return None

    # 対象日の NDMI
    arr_now = fetch_scene(lat, lon, target_date_iso)
    if arr_now is None:
        return None
    idx_now = compute_indices(arr_now)
    if not idx_now:
        return None

    baseline = float(np.mean(historical_ndmi))
    return idx_now["ndmi_mean"] - baseline


# ─────────────────────────────────────────────────────────
# FIRMS SP — NEG fire-free 確認
# ─────────────────────────────────────────────────────────

def firms_fire_free(lat: float, lon: float, date_str: str, margin_days: int = 28) -> bool:
    """指定座標・日付の ±margin_days 内に FIRMS VIIRS_SNPP_SP の nominal/high 検知がないか確認。"""
    if not FIRMS_API_KEY:
        return True

    r = 0.15
    bbox = f"{lon - r:.3f},{lat - r:.3f},{lon + r:.3f},{lat + r:.3f}"
    check_days = min(margin_days * 2, 365)
    start_dt = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=margin_days)
    start_str = start_dt.strftime("%Y-%m-%d")

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_API_KEY}/VIIRS_SNPP_SP/{bbox}/{check_days}/{start_str}"
    )
    try:
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            return True
        rows = list(csv.DictReader(io.StringIO(resp.text)))
        fires = [
            row for row in rows
            if str(row.get("confidence", "")).strip().lower() in ("nominal", "high")
        ]
        return len(fires) == 0
    except Exception:
        return True


# ─────────────────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────────────────

def save_composite(arr: np.ndarray, path: Path) -> None:
    """疑似カラー保存: R=SWIR22(B12), G=B8A, B=SWIR16(B11) — 乾燥/植生/水分の視覚化。
    arr: (H, W, 5) ch0=nir08(B8A), ch1=nir, ch2=swir22, ch3=swir16, ch4=red
    """
    rgb = np.stack([arr[:, :, 2], arr[:, :, 0], arr[:, :, 3]], axis=-1)  # SWIR22, B8A, SWIR16
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def plot_distributions(pos_records: list[dict], neg_records: list[dict], save_dir: Path) -> None:
    if not HAS_MPL or not neg_records:
        return

    fig_dir = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # lead 別 NEG リスト
    def _neg_lead(lead: int) -> list[float]:
        v = [r["ndmi_mean"] for r in neg_records if r["lead_days"] == lead]
        return v if v else [r["ndmi_mean"] for r in neg_records]

    def _neg_nbr_lead(lead: int) -> list[float]:
        v = [r["nbr_mean"] for r in neg_records if r["lead_days"] == lead]
        return v if v else [r["nbr_mean"] for r in neg_records]

    # ── NDMI 分布 (主指標) ──
    fig, axes = plt.subplots(1, len(LEAD_DAYS), figsize=(5 * len(LEAD_DAYS), 4), sharey=False)
    if len(LEAD_DAYS) == 1:
        axes = [axes]

    for ax, lead in zip(axes, LEAD_DAYS):
        pos_all  = [r["ndmi_mean"] for r in pos_records if r["lead_days"] == lead]
        pos_chap = [r["ndmi_mean"] for r in pos_records if r["lead_days"] == lead and r.get("veg") == "chaparral"]
        neg_l    = _neg_lead(lead)

        bins = np.linspace(-0.2, 0.7, 20)
        ax.hist(neg_l,    bins=bins, alpha=0.6, label=f"NEG -{lead}d (n={len(neg_l)})",      color="steelblue", density=True)
        if pos_all:
            ax.hist(pos_all,   bins=bins, alpha=0.4, label=f"POS all (n={len(pos_all)})",       color="salmon",    density=True)
        if pos_chap:
            ax.hist(pos_chap,  bins=bins, alpha=0.7, label=f"POS chaparral (n={len(pos_chap)})", color="tomato",    density=True)

        ax.axvline(np.mean(neg_l),  color="steelblue", ls="--", lw=1.5, label=f"NEG mu={np.mean(neg_l):.3f}")
        if pos_chap:
            ax.axvline(np.mean(pos_chap), color="tomato",    ls="--", lw=1.5, label=f"POS chap mu={np.mean(pos_chap):.3f}")

        ax.set_xlabel("NDMI (B8A/B11)")
        ax.set_title(f"NDMI POS -{lead}d vs NEG [主指標]")
        ax.legend(fontsize=7)

    fig.suptitle("FireGuard /poc v2 — NDMI 分布 (発火前 vs チャパラル非発火)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "ndmi_distributions.png", dpi=120)
    plt.close(fig)

    # ── NBR 分布 (副指標) ──
    fig2, axes2 = plt.subplots(1, len(LEAD_DAYS), figsize=(5 * len(LEAD_DAYS), 4), sharey=False)
    if len(LEAD_DAYS) == 1:
        axes2 = [axes2]

    for ax, lead in zip(axes2, LEAD_DAYS):
        pos_chap  = [r["nbr_mean"] for r in pos_records if r["lead_days"] == lead and r.get("veg") == "chaparral"]
        neg_nbr_l = _neg_nbr_lead(lead)
        bins = np.linspace(-0.2, 0.7, 20)
        ax.hist(neg_nbr_l,  bins=bins, alpha=0.6, label=f"NEG -{lead}d (n={len(neg_nbr_l)})", color="steelblue", density=True)
        if pos_chap:
            ax.hist(pos_chap, bins=bins, alpha=0.7, label=f"POS chap (n={len(pos_chap)})",     color="tomato",    density=True)
        ax.axvline(np.mean(neg_nbr_l), color="steelblue", ls="--", lw=1.5)
        if pos_chap:
            ax.axvline(np.mean(pos_chap), color="tomato", ls="--", lw=1.5)
        ax.set_xlabel("NBR (B8A/B12)")
        ax.set_title(f"NBR POS -{lead}d vs NEG [副指標]")
        ax.legend(fontsize=7)

    fig2.suptitle("FireGuard /poc v2 — NBR 分布 (副指標)", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "nbr_distributions.png", dpi=120)
    plt.close(fig2)

    # ── NDMI 時系列ライン: POS チャパラル vs NEG (per event) ──
    neg_names      = list(dict.fromkeys(r["name"] for r in neg_records))
    pos_chap_names = list(dict.fromkeys(r["name"] for r in pos_records if r.get("veg") == "chaparral"))

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    colors_neg = plt.cm.Blues(np.linspace(0.4, 0.85, max(len(neg_names), 1)))
    colors_pos = plt.cm.Reds(np.linspace(0.4, 0.85, max(len(pos_chap_names), 1)))

    for name, color in zip(neg_names, colors_neg):
        pts = sorted([(r["lead_days"], r["ndmi_mean"]) for r in neg_records if r["name"] == name], reverse=True)
        if pts:
            xs, ys = zip(*pts)
            ax3.plot(xs, ys, "o--", color=color, lw=1.2, ms=5, label=f"NEG {name[:22]}")

    for name, color in zip(pos_chap_names, colors_pos):
        pts = sorted([(r["lead_days"], r["ndmi_mean"]) for r in pos_records if r["name"] == name and r.get("veg") == "chaparral"], reverse=True)
        if pts:
            xs, ys = zip(*pts)
            ax3.plot(xs, ys, "^-", color=color, lw=1.5, ms=7, label=f"POS {name[:22]}")

    ax3.set_xlabel("Days before fire/reference (lead days)")
    ax3.set_ylabel("NDMI (B8A/B11)")
    ax3.set_title("NDMI time series: POS chaparral vs NEG chaparral")
    ax3.invert_xaxis()
    ax3.axhline(0, color="gray", ls=":", lw=0.8)
    ax3.legend(fontsize=6, loc="upper left", ncol=2)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "ndmi_timeseries.png", dpi=120)
    plt.close(fig3)

    # ── NDMI scatter: 全イベント -14d (veg type 色分け) ──
    neg_14    = _neg_lead(14)
    lead_plot = 14
    chap_pos  = [(r["ndmi_mean"], r["name"]) for r in pos_records if r["lead_days"] == lead_plot and r.get("veg") == "chaparral"]
    conif_pos = [(r["ndmi_mean"], r["name"]) for r in pos_records if r["lead_days"] == lead_plot and r.get("veg") == "conifer"]

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.scatter(range(len(neg_14)), neg_14, color="steelblue", s=60, label=f"NEG -{lead_plot}d (n={len(neg_14)})", zorder=3)
    offset = len(neg_14)
    if chap_pos:
        ax4.scatter(range(offset, offset + len(chap_pos)),
                    [v for v, _ in chap_pos], color="tomato", marker="^", s=80, label="POS chaparral", zorder=3)
    if conif_pos:
        ax4.scatter(range(offset + len(chap_pos), offset + len(chap_pos) + len(conif_pos)),
                    [v for v, _ in conif_pos], color="orange", marker="s", s=80, label="POS conifer", zorder=3)
    if neg_14:
        ax4.axhline(np.mean(neg_14), color="steelblue", ls="--", lw=1.2, label=f"NEG mean={np.mean(neg_14):.3f}")
    ax4.set_ylabel("NDMI (B8A/B11)")
    ax4.set_title(f"NDMI individual events (-{lead_plot}d) by veg type")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(fig_dir / "ndmi_by_vegtype.png", dpi=120)
    plt.close(fig4)

    print(f"  figures saved: {fig_dir}/")


# ─────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────

def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = SAVE_DIR / "images"
    img_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("FireGuard /poc v2 — 発火前 NDMI 分離確認 (チャパラル限定)")
    print("主指標: NDMI=(B8A-B11)/(B8A+B11)  副指標: NBR=(B8A-B12)/(B8A+B12)")
    print("=" * 60)

    # ── SimSat ヘルスチェック ──
    print("\n[1] SimSat 疎通確認 ...")
    if not simsat_health():
        print("  ERROR: http://localhost:9005/ に接続できません。SimSat を起動してください。")
        sys.exit(1)
    print("  OK: SimSat online")

    # ── POS シーン取得 ──
    print(f"\n[2] POS シーン取得 (イベント={len(POS_EVENTS)}, リードタイム={LEAD_DAYS}d)")
    pos_records: list[dict] = []

    for ev in POS_EVENTS:
        fire_dt = datetime.strptime(ev["fire_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        veg_tag = f"[{ev['veg']}]"
        print(f"\n  {ev['name']} {veg_tag} ({ev['fire_date']})")

        for lead in LEAD_DAYS:
            target_dt = fire_dt - timedelta(days=lead)
            ts_iso = target_dt.strftime("%Y-%m-%dT12:00:00Z")
            print(f"    -{lead}d [{ts_iso[:10]}] ... ", end="", flush=True)

            arr = fetch_scene(ev["lat"], ev["lon"], ts_iso)
            if arr is None:
                print("no image")
                continue

            # 画像は有効性に関わらず常に保存（雲・黒画像の目視確認のため）
            img_name = f"pos_{ev['name'].replace(' ', '_')}_-{lead}d.png"
            save_composite(arr, img_dir / img_name)

            idx = compute_indices(arr)
            if not idx:
                print(f"invalid (n_valid<100) — image saved: {img_name}")
                continue

            print(
                f"OK  NDMI={idx['ndmi_mean']:.3f}(p10={idx['ndmi_p10']:.3f})"
                f"  NBR={idx['nbr_mean']:.3f}  NDVI={idx['ndvi_mean']:.3f}"
            )

            pos_records.append({
                "label": "POS",
                "name": ev["name"],
                "veg": ev["veg"],
                "lat": ev["lat"],
                "lon": ev["lon"],
                "fire_date": ev["fire_date"],
                "lead_days": lead,
                "scene_date": ts_iso[:10],
                **idx,
            })

    # ── POS seasonal anomaly (任意) ──
    print(f"\n[2b] POS NDMI seasonal anomaly 取得 (SimSat 過去データがある場合のみ)")
    anomaly_map: dict[str, float] = {}  # key = "{name}-{lead}"
    for ev in POS_EVENTS[:3]:  # 最初の3イベントのみ (API 負荷軽減)
        fire_dt = datetime.strptime(ev["fire_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        target_dt = fire_dt - timedelta(days=14)
        ts_iso = target_dt.strftime("%Y-%m-%dT12:00:00Z")
        print(f"  {ev['name']} -14d anomaly ... ", end="", flush=True)
        anom = fetch_ndmi_anomaly(ev["lat"], ev["lon"], ts_iso)
        if anom is not None:
            key = f"{ev['name']}-14"
            anomaly_map[key] = anom
            print(f"NDMI anomaly = {anom:+.3f}")
        else:
            print("skip (no historical data)")

    # ── NEG シーン取得 (LEAD_DAYS 分の時系列) ──
    print(f"\n[3] NEG シーン取得 — チャパラル優勢エリア限定 (候補={len(NEG_CANDIDATES)}, 各 {LEAD_DAYS}d 分)")
    print("    ※ NEG の lead_days は参照日からの遡り日数 (fire はないがPOSと同窓で時系列比較)")
    neg_records: list[dict] = []

    for ev in NEG_CANDIDATES:
        ref_dt = datetime.strptime(ev["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        print(f"\n  {ev['name']} (ref={ev['date']})")

        print(f"    FIRMS SP fire-free 確認 ... ", end="", flush=True)
        if not firms_fire_free(ev["lat"], ev["lon"], ev["date"]):
            print("SKIP (fire detected nearby)")
            continue
        print("OK")

        for lead in LEAD_DAYS:
            target_dt = ref_dt - timedelta(days=lead)
            ts_iso = target_dt.strftime("%Y-%m-%dT12:00:00Z")
            print(f"    -{lead}d [{ts_iso[:10]}] ... ", end="", flush=True)

            arr = fetch_scene(ev["lat"], ev["lon"], ts_iso)
            if arr is None:
                print("no image")
                continue

            # 画像は有効性に関わらず常に保存
            img_name = f"neg_{ev['name'].replace(' ', '_')}_-{lead}d.png"
            save_composite(arr, img_dir / img_name)

            idx = compute_indices(arr)
            if not idx:
                print(f"invalid (n_valid<100) — image saved: {img_name}")
                continue

            print(
                f"OK  NDMI={idx['ndmi_mean']:.3f}(p10={idx['ndmi_p10']:.3f})"
                f"  NBR={idx['nbr_mean']:.3f}  NDVI={idx['ndvi_mean']:.3f}"
            )

            neg_records.append({
                "label": "NEG",
                "name": ev["name"],
                "veg": "chaparral",
                "lat": ev["lat"],
                "lon": ev["lon"],
                "ref_date": ev["date"],
                "lead_days": lead,
                "scene_date": ts_iso[:10],
                **idx,
            })

    # ── CSV 保存 ──
    print(f"\n[4] 結果保存 → {SAVE_DIR}/results.csv")
    all_records = pos_records + neg_records
    if all_records:
        # POS と NEG でフィールドが異なる (ref_date など) ため全レコードから収集
        fieldnames = list(dict.fromkeys(k for r in all_records for k in r.keys()))
        with open(SAVE_DIR / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_records)

    # ── 統計分析 ──
    print("\n[5] 統計分析 (主指標: NDMI, Mann-Whitney U, alternative='less': POS < NEG)")
    print("    仮説: 発火前のチャパラルは NEG チャパラルより NDMI が低い (= 乾燥している)")

    # NEG は lead 別に分けて比較 (POS と同じ lead_days で揃える)
    def neg_at(lead: int) -> list[float]:
        return [r["ndmi_mean"] for r in neg_records if r["lead_days"] == lead]
    def neg_nbr_at(lead: int) -> list[float]:
        return [r["nbr_mean"]  for r in neg_records if r["lead_days"] == lead]

    neg_ndmi_all = [r["ndmi_mean"] for r in neg_records]
    neg_nbr_all  = [r["nbr_mean"]  for r in neg_records]

    if not neg_ndmi_all:
        print("  WARNING: NEG サンプルなし → 分析不可")
    else:
        print(f"\n  NEG chaparral (全lead)  NDMI: mean={np.mean(neg_ndmi_all):.3f}  std={np.std(neg_ndmi_all):.3f}  n={len(neg_ndmi_all)}")
        for lead in LEAD_DAYS:
            nl = neg_at(lead)
            if nl:
                print(f"    NEG -{lead}d: mean={np.mean(nl):.3f}  n={len(nl)}")

    go_count_ndmi = 0
    go_count_nbr  = 0

    print("\n  [NDMI — 主指標] (NEG は同 lead で比較)")
    for lead in LEAD_DAYS:
        pos_all_ndmi  = [r["ndmi_mean"] for r in pos_records if r["lead_days"] == lead]
        pos_chap_ndmi = [r["ndmi_mean"] for r in pos_records if r["lead_days"] == lead and r.get("veg") == "chaparral"]
        neg_lead      = neg_at(lead)
        neg_use       = neg_lead if neg_lead else neg_ndmi_all  # lead 別がない場合は全体を使う

        if not pos_all_ndmi or not neg_use:
            print(f"  -{lead}d: データ不足")
            continue

        # 全イベント
        delta_all = np.mean(pos_all_ndmi) - np.mean(neg_use)
        p_all = 1.0
        if HAS_SCIPY and len(pos_all_ndmi) >= 3 and len(neg_use) >= 3:
            _, p_all = scipy_stats.mannwhitneyu(pos_all_ndmi, neg_use, alternative="less")

        sig_all = "★ p<0.05" if p_all < 0.05 else ("△ p<0.10" if p_all < 0.10 else "ns")
        print(f"  -{lead}d [全]:       mean={np.mean(pos_all_ndmi):.3f}  Δ={delta_all:+.3f}  p={p_all:.3f}  {sig_all}  (n_pos={len(pos_all_ndmi)}, n_neg={len(neg_use)})")

        # チャパラル限定
        if pos_chap_ndmi:
            delta_chap = np.mean(pos_chap_ndmi) - np.mean(neg_use)
            p_chap = 1.0
            if HAS_SCIPY and len(pos_chap_ndmi) >= 3 and len(neg_use) >= 3:
                _, p_chap = scipy_stats.mannwhitneyu(pos_chap_ndmi, neg_use, alternative="less")
            sig_chap = "★ p<0.05" if p_chap < 0.05 else ("△ p<0.10" if p_chap < 0.10 else "ns")
            print(f"  -{lead}d [チャパラル]: mean={np.mean(pos_chap_ndmi):.3f}  Δ={delta_chap:+.3f}  p={p_chap:.3f}  {sig_chap}  (n_pos={len(pos_chap_ndmi)}, n_neg={len(neg_use)})")
            if p_chap < 0.10:
                go_count_ndmi += 1

    print("\n  [NBR — 副指標] (NEG は同 lead で比較)")
    for lead in LEAD_DAYS:
        pos_chap_nbr = [r["nbr_mean"] for r in pos_records if r["lead_days"] == lead and r.get("veg") == "chaparral"]
        neg_nbr_lead = neg_nbr_at(lead) if neg_nbr_at(lead) else neg_nbr_all
        if not pos_chap_nbr or not neg_nbr_lead:
            continue
        delta = np.mean(pos_chap_nbr) - np.mean(neg_nbr_lead)
        p_val = 1.0
        if HAS_SCIPY and len(pos_chap_nbr) >= 3 and len(neg_nbr_lead) >= 3:
            _, p_val = scipy_stats.mannwhitneyu(pos_chap_nbr, neg_nbr_lead, alternative="less")
        sig = "★ p<0.05" if p_val < 0.05 else ("△ p<0.10" if p_val < 0.10 else "ns")
        print(f"  -{lead}d [チャパラル]: mean={np.mean(pos_chap_nbr):.3f}  Δ={delta:+.3f}  p={p_val:.3f}  {sig}  (n_pos={len(pos_chap_nbr)}, n_neg={len(neg_nbr_lead)})")
        if p_val < 0.10:
            go_count_nbr += 1

    # ── seasonal anomaly サマリ ──
    if anomaly_map:
        print(f"\n  [NDMI seasonal anomaly (参考)]")
        for key, anom in anomaly_map.items():
            print(f"  {key}: anomaly={anom:+.3f}")

    # ── 可視化 ──
    if HAS_MPL:
        plot_distributions(pos_records, neg_records, SAVE_DIR)

    # ── Go/No-Go 判断 ──
    print("\n" + "=" * 60)
    print("Go/No-Go 判断")
    print("=" * 60)
    n_pos_chap  = len([r for r in pos_records if r.get("veg") == "chaparral"])
    n_neg_sites = len(set(r["name"] for r in neg_records))
    n_neg_total = len(neg_records)
    print(f"  取得済み POS シーン (全):        {len(pos_records)} / {len(POS_EVENTS) * len(LEAD_DAYS)} 試行")
    print(f"  取得済み POS シーン (チャパラル): {n_pos_chap}")
    print(f"  取得済み NEG シーン (地点数):     {n_neg_sites} / {len(NEG_CANDIDATES)} 候補")
    print(f"  取得済み NEG シーン (time series): {n_neg_total} シーン (地点×{len(LEAD_DAYS)}lead)")
    print(f"  NDMI 有意差あり (p<0.10, チャパラル): {go_count_ndmi} / {len(LEAD_DAYS)} リードタイム")
    print(f"  NBR  有意差あり (p<0.10, チャパラル): {go_count_nbr}  / {len(LEAD_DAYS)} リードタイム (副指標)")

    cond1 = len(pos_records) > 0 and n_neg_total > 0
    cond2 = go_count_ndmi >= 1

    print(f"\n  ① SimSat データ取得:   {'✅ OK' if cond1 else '❌ NG'}")
    print(f"  ② NDMI スペクトル分離: {'✅ OK' if cond2 else '❌ NG (Plan B: seasonal anomaly 追加 or サンプル増加)'}")

    if cond1 and cond2:
        print("\n  → Go: /poc2 へ進める")
    elif cond1 and not cond2:
        print("\n  → Conditional: NDMI シグナル弱い → サンプル追加 (n≥10 チャパラル) で再試行推奨")
        print("     seasonal anomaly が取得できた場合はそちらも確認すること")
    else:
        print("\n  → No-Go: SimSat データ取得の根本的問題")

    print(f"\n  詳細結果: {SAVE_DIR}/results.csv")
    print(f"  画像:     {SAVE_DIR}/images/")
    if HAS_MPL:
        print(f"  図:       {SAVE_DIR}/figures/")
        print(f"    - ndmi_distributions.png  (主指標)")
        print(f"    - nbr_distributions.png   (副指標)")
        print(f"    - ndmi_by_vegtype.png     (植生タイプ別 scatter)")


if __name__ == "__main__":
    main()
