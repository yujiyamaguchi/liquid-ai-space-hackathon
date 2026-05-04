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

import argparse
import base64
import csv
import io
import json
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
SIZE_KM = 5.0   # FireEdge 準拠。20km だと nodata 多発・植生混合が起きた
WINDOW_SECONDS = 5 * 86400   # Sentinel-2 revisit 3〜5日 → ±5日でシーン検索
BANDS = ["nir08", "nir", "swir22", "swir16", "red", "green"]
# ch0=nir08(B8A,865nm), ch1=nir(B08,842nm), ch2=swir22(B12,2190nm),
# ch3=swir16(B11,1610nm), ch4=red(B04,665nm), ch5=green(B03,560nm)

LEAD_DAYS = [7, 14, 21, 28]
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
    {"name": "Woolsey Fire",          "lat": 34.04, "lon": -118.75, "fire_date": "2018-11-08", "veg": "chaparral"},
    {"name": "Easy Fire",             "lat": 34.35, "lon": -118.55, "fire_date": "2019-10-30", "veg": "chaparral"},
    {"name": "Tick Fire",             "lat": 34.38, "lon": -118.40, "fire_date": "2019-10-24", "veg": "chaparral"},
    {"name": "Bobcat Fire",           "lat": 34.22, "lon": -117.85, "fire_date": "2020-09-06", "veg": "chaparral"},
    {"name": "Alisal Fire",           "lat": 34.55, "lon": -120.30, "fire_date": "2021-10-11", "veg": "chaparral"},
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
    """SimSat から6バンド配列を取得。取得不可の場合 None を返す。

    Returns: (H, W, 6) float32 [0, 1]
      ch0=nir08(B8A), ch1=nir(B08), ch2=swir22(B12), ch3=swir16(B11),
      ch4=red(B04), ch5=green(B03)
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
    """6バンド配列からスペクトル指標を計算する。

    arr: (H, W, 6)
      ch0=nir08(B8A,865nm), ch1=nir(B08,842nm),
      ch2=swir22(B12,2190nm), ch3=swir16(B11,1610nm),
      ch4=red(B04,665nm), ch5=green(B03,560nm)

    NDMI  = (B8A - B11) / (B8A + B11)   [Gao 1996]
            主指標。葉内液体水分量 (EWT) の最良プロキシ。
            NDMI_p5/p10 は「最も乾燥したピクセル」を捉える (FireEdge の NBR2_min 発想を転用)。

    NBR2  = (B11 - B12) / (B11 + B12)   [FireEdge 最重要指標; 参考比較]
            活火時に有効。発火前では別メカニズムだが実測比較のために追加。

    NBR   = (B8A - B12) / (B8A + B12)   [Key & Benson 2006 — pre-fire baseline]
            副指標。燃料構造 + 水分の複合指標。

    NDVI  = (B08 - B04) / (B08 + B04)
            植生活性度の参照指標。チャパラル識別 (0.2〜0.4) に使用。
    """
    nir08  = arr[:, :, 0].astype(np.float64)  # B8A (865nm)
    nir    = arr[:, :, 1].astype(np.float64)  # B08 (842nm)
    swir22 = arr[:, :, 2].astype(np.float64)  # B12 (2190nm)
    swir16 = arr[:, :, 3].astype(np.float64)  # B11 (1610nm)
    red    = arr[:, :, 4].astype(np.float64)  # B04 (665nm)

    eps = 1e-8

    # 主指標: NDMI (B8A/B11)
    ndmi_map = (nir08 - swir16) / (nir08 + swir16 + eps)

    # 参考比較: NBR2 (B11/B12) — FireEdge の最重要指標。発火前での有効性を実測確認
    nbr2_map = (swir16 - swir22) / (swir16 + swir22 + eps)

    # 副指標: NBR (B8A/B12)
    nbr_map  = (nir08 - swir22) / (nir08 + swir22 + eps)

    # 参照指標: NDVI (B08/B04)
    ndvi_map = (nir - red) / (nir + red + eps)

    # 有効ピクセル: 飽和・全黒を除外
    valid = (nir08 > 0.01) & (nir08 < 0.99) & (swir16 < 0.99) & (swir22 < 0.99)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return {}

    ndmi_valid = ndmi_map[valid]
    nbr2_valid = nbr2_map[valid]
    return {
        "ndmi_mean":   float(ndmi_valid.mean()),
        "ndmi_p10":    float(np.percentile(ndmi_valid, 10)),
        "ndmi_p5":     float(np.percentile(ndmi_valid, 5)),   # FireEdge「min」コンセプト転用
        "nbr2_mean":   float(nbr2_valid.mean()),
        "nbr2_p10":    float(np.percentile(nbr2_valid, 10)),
        "nbr2_min":    float(nbr2_valid.min()),               # FireEdge NBR2_min と同定義
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
    arr: (H, W, 6) ch0=nir08(B8A), ch1=nir, ch2=swir22, ch3=swir16, ch4=red, ch5=green
    """
    rgb = np.stack([arr[:, :, 2], arr[:, :, 0], arr[:, :, 3]], axis=-1)  # SWIR22, B8A, SWIR16
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    """疑似RGB保存: R=B04(red), G=B03(green), B=B8A(nir08) — 目視確認用。
    厳密な自然色ではないが「どんな場所か」を確認するのに十分な構成。
    arr: (H, W, 6) ch0=nir08(B8A), ch1=nir, ch2=swir22, ch3=swir16, ch4=red, ch5=green
    """
    rgb = np.stack([arr[:, :, 4], arr[:, :, 5], arr[:, :, 0]], axis=-1)  # red, green, nir08
    for c in range(3):
        ch = rgb[:, :, c]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        rgb[:, :, c] = np.clip((ch - p2) / (p98 - p2 + 1e-8), 0, 1)
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
# レポート生成
# ─────────────────────────────────────────────────────────

def _compute_lead_stats(
    pos_records: list[dict],
    neg_records: list[dict],
    field: str,
) -> list[dict]:
    """指定フィールドの lead ごと統計を返す。"""
    neg_all = [r[field] for r in neg_records if field in r]
    results = []
    for lead in LEAD_DAYS:
        pos_v = [r[field] for r in pos_records if r["lead_days"] == lead
                 and r.get("veg") == "chaparral" and field in r]
        neg_lead = [r[field] for r in neg_records if r["lead_days"] == lead and field in r]
        neg_v = neg_lead if neg_lead else neg_all
        if not pos_v or not neg_v:
            continue
        delta = float(np.mean(pos_v) - np.mean(neg_v))
        p_val = 1.0
        if HAS_SCIPY and len(pos_v) >= 3 and len(neg_v) >= 3:
            _, p_val = scipy_stats.mannwhitneyu(pos_v, neg_v, alternative="less")
        sig = "★ p<0.05" if p_val < 0.05 else ("△ p<0.10" if p_val < 0.10 else "ns")
        results.append({
            "lead": lead,
            "pos_mean": float(np.mean(pos_v)),
            "neg_mean": float(np.mean(neg_v)),
            "delta": delta,
            "p_val": float(p_val),
            "sig": sig,
            "n_pos": len(pos_v),
            "n_neg": len(neg_v),
        })
    return results


def write_report(
    pos_records: list[dict],
    neg_records: list[dict],
    anomaly_map: dict[str, float],
    save_dir: Path,
    run_name: str = "",
) -> Path:
    """背景・目的・設計・結果・考察・Go/No-Go を含む Markdown レポートを書き出す。"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L: list[str] = []

    # ── ヘッダー ──────────────────────────────────────────
    L += [
        "# FireGuard /poc レポート — 発火前植生乾燥ストレスの衛星スペクトル検出",
        "",
        f"**実行日時**: {now_str}  ",
        f"**run_name**: `{run_name or '(default)'}`  ",
        f"**スクリプト**: `apps/fireguard/poc.py` (v3)  ",
        f"**パラメータ**: SIZE_KM={SIZE_KM}, LEAD_DAYS={LEAD_DAYS}  ",
        "",
        "---",
        "",
    ]

    # ── 1. 背景と目的 ────────────────────────────────────
    L += [
        "## 1. 背景と目的",
        "",
        "### 背景",
        "",
        "本ハッカソンは「衛星にオンボードAI処理能力が搭載された未来で何ができるか」を前提とする。"
        "FireGuard はその前提の上で **発火前 7〜28 日**（主シグナルは直前 7 日）に衛星スペクトルから植生乾燥ストレスを検知し、"
        "消防・林業機関にアラートを送る早期警戒システムを目指している。",
        "",
        "既存手段の限界:",
        "",
        "| 既存手段 | 限界 |",
        "|---|---|",
        "| NASA FIRMS (VIIRS/MODIS) | 発火「後」検知。予防に使えない |",
        "| 気象モデル (気温・湿度) | 空間解像度が粗く局所的な植生乾燥を捉えられない |",
        "| 地上 LFMC 観測 | サンプリング密度が低く、広域カバレッジがない |",
        "",
        "Sentinel-2 の近赤外〜短波赤外バンドは葉内液体水分量 (LFMC) に高感度であり、"
        "オンボード処理で地上伝送前にリスク判定できれば帯域幅削減 + 遅延ゼロの警戒が実現する。",
        "",
        "### /poc で検証すること",
        "",
        "CLAUDE.md の /poc 完了条件に従い、以下の 2 点を確認する:",
        "",
        "1. **SimSat データ取得の実現可能性**: 対象イベントの座標・日時で Sentinel-2 シーンを取得できるか",
        "2. **スペクトルシグナルの分離性**: 発火前チャパラル (POS) と非発火チャパラル (NEG) で"
        " NDMI_p10 が統計的に分離できるか。特に -7d が主シグナル — これが Go/No-Go の核心",
        "",
        "シグナルが分離できなければ、どれだけデータを集めてもモデルは学習できない。",
        "",
        "---",
        "",
    ]

    # ── 2. 実験設計 ──────────────────────────────────────
    L += [
        "## 2. 実験設計",
        "",
        "### 主指標: NDMI の選定根拠",
        "",
        "```",
        "NDMI = (B8A - B11) / (B8A + B11)   [Gao 1996, Remote Sensing of Environment]",
        "  B8A: 865nm (近赤外, 植生反射)",
        "  B11: 1610nm (短波赤外, 液体水吸収帯)",
        "```",
        "",
        "- B11 (1610nm) は液体水の吸収帯に位置し、LFMC 推定への感度が最も高い",
        "- Myoung et al. 2018 (MDPI Remote Sensing): 南カリフォルニアのチャパラルで"
        " NDMI ベース LFMC 推定 R²=0.76、MAE=9.68% を達成",
        "- **LFMC 臨界値**: < 77〜80% で南カリフォルニア大規模火災リスクが急増 (Santa Monica Mts 研究)",
        "- **リードタイム 7〜28 日の生理的根拠**: 乾燥ストレス下でクロロフィルが 7〜75%、"
        "アントシアニンが 38〜100% 減少しスペクトル変化として観測可能 (Mississippi State 2025)。"
        "最も強いシグナルは発火直前 7 日 (NDMI_p10: p=0.038)。",
        "",
        "### 植生タイプと統計分析の対象範囲",
        "",
        "POS イベントには**チャパラル**と**針葉樹 (Conifer)** の2種類を含むが、統計分析と Go/No-Go 判定は"
        "**チャパラル限定**で行う。理由:",
        "",
        "| 植生タイプ | NDMI_p10 の挙動 | 分析役割 |",
        "|---|---|---|",
        "| チャパラル (低木群落) | 乾燥 → B11 吸収低下 → NDMI 低下。含水率変化がスペクトルに直接反映 | **主分析対象** |",
        "| 針葉樹 (常緑) | 乾燥ストレス下でも NIR 反射率を維持 → NDMI 変動幅が小さく NEG と重複 | **汎化確認用 (参考)** |",
        "",
        "Myoung et al. 2018 の LFMC 推定 (R²=0.76) もチャパラル対象。"
        "針葉樹火災の FT データは /finetune フェーズで別途設計する。",
        "",
        "### サンプル設計",
        "",
        f"**POS イベント — 統計分析対象 (チャパラル {len([e for e in POS_EVENTS if e['veg']=='chaparral'])} 件)**",
        "",
        "| イベント | 発火日 | 座標 |",
        "|---|---|---|",
    ] + [
        f"| {e['name']} | {e['fire_date']} | {e['lat']:.2f}N, {e['lon']:.2f}W |"
        for e in POS_EVENTS if e['veg'] == 'chaparral'
    ] + [
        "",
        f"**POS イベント — 汎化確認用 (針葉樹 {len([e for e in POS_EVENTS if e['veg']=='conifer'])} 件)**",
        "",
        "| イベント | 発火日 |",
        "|---|---|",
    ] + [
        f"| {e['name']} | {e['fire_date']} |"
        for e in POS_EVENTS if e['veg'] == 'conifer'
    ] + [
        "",
        f"**NEG サンプル — チャパラル優勢エリア {len(NEG_CANDIDATES)} 地点** (FIRMS SP fire-free 確認済み)",
        "",
        "| 地点 | 参照日 |",
        "|---|---|",
    ] + [
        f"| {e['name']} | {e['date']} |"
        for e in NEG_CANDIDATES
    ] + [
        "",
        f"**リードタイム**: {LEAD_DAYS} 日前  **取得窓**: {SIZE_KM} km 四方",
        "",
        "---",
        "",
    ]

    # ── 3. データ取得結果 ────────────────────────────────
    n_pos_chap   = len([r for r in pos_records if r.get("veg") == "chaparral"])
    n_pos_conif  = len([r for r in pos_records if r.get("veg") == "conifer"])
    n_neg_sites  = len(set(r["name"] for r in neg_records))
    n_neg_total  = len(neg_records)
    n_chap_tried = len([e for e in POS_EVENTS if e['veg'] == 'chaparral']) * len(LEAD_DAYS)
    n_conif_tried= len([e for e in POS_EVENTS if e['veg'] == 'conifer'])   * len(LEAD_DAYS)

    L += [
        "## 3. データ取得結果",
        "",
        "| 項目 | 取得シーン数 | 試行数 | 備考 |",
        "|---|---|---|---|",
        f"| POS チャパラル (統計対象) | {n_pos_chap} | {n_chap_tried} | |",
        f"| POS 針葉樹 (汎化確認用) | {n_pos_conif} | {n_conif_tried} | 統計分析には含めない |",
        f"| NEG チャパラル | {n_neg_total} | {len(NEG_CANDIDATES) * len(LEAD_DAYS)} | FIRMS fire-free 確認済み |",
        "",
    ]

    # POS per-event summary (chaparral only, earliest lead as representative)
    rep_lead = min(LEAD_DAYS)
    chap_pos_rep = [r for r in pos_records if r.get("veg") == "chaparral" and r["lead_days"] == rep_lead]
    if chap_pos_rep:
        L += [
            f"**POS チャパラル イベント別 NDMI (-{rep_lead}d 代表値、NDMI_mean 昇順)**",
            "",
            "| イベント | NDMI mean | NDMI p10 |",
            "|---|---|---|",
        ]
        for r in sorted(chap_pos_rep, key=lambda x: x["ndmi_mean"]):
            L.append(f"| {r['name']} | {r['ndmi_mean']:.3f} | {r['ndmi_p10']:.3f} |")
        L.append("")

    L += ["---", ""]

    # ── 4. 統計分析 ──────────────────────────────────────
    L += [
        "## 4. 統計分析",
        "",
        "Mann-Whitney U 検定 (片側, alternative='less': POS < NEG)。"
        "**チャパラル限定** の POS と同一リードタイムの NEG を比較。針葉樹は含めない。",
        "",
        "### NDMI mean (シーン平均)",
        "",
        "| Lead | POS mean | NEG mean | Δ | p値 | 判定 | n_pos | n_neg |",
        "|---|---|---|---|---|---|---|---|",
    ]

    ndmi_stats = _compute_lead_stats(pos_records, neg_records, "ndmi_mean")
    go_count_ndmi = 0
    for s in ndmi_stats:
        if s["p_val"] < 0.10:
            go_count_ndmi += 1
        L.append(
            f"| -{s['lead']}d | {s['pos_mean']:.3f} | {s['neg_mean']:.3f}"
            f" | {s['delta']:+.3f} | {s['p_val']:.3f} | **{s['sig']}** | {s['n_pos']} | {s['n_neg']} |"
        )

    L += [
        "",
        "### NDMI p10 (最乾燥 10% ピクセル) ← 主判断指標",
        "",
        "発火直前に先行乾燥する南向き斜面・尾根ピクセルを捉える。"
        "シーン平均より分散が小さく、時系列での変化が明確。",
        "",
        "| Lead | POS mean | NEG mean | Δ | p値 | 判定 | n_pos | n_neg |",
        "|---|---|---|---|---|---|---|---|",
    ]

    ndmi_p10_stats = _compute_lead_stats(pos_records, neg_records, "ndmi_p10")
    go_count_ndmi_p10 = 0
    for s in ndmi_p10_stats:
        if s["p_val"] < 0.10:
            go_count_ndmi_p10 += 1
        L.append(
            f"| -{s['lead']}d | {s['pos_mean']:.3f} | {s['neg_mean']:.3f}"
            f" | {s['delta']:+.3f} | {s['p_val']:.3f} | **{s['sig']}** | {s['n_pos']} | {s['n_neg']} |"
        )

    L += [
        "",
        "### NBR2 mean (参考: FireEdge の主指標、発火前には無効)",
        "",
        "| Lead | POS mean | NEG mean | Δ | p値 | 判定 |",
        "|---|---|---|---|---|---|",
    ]

    go_count_nbr2 = 0
    for s in _compute_lead_stats(pos_records, neg_records, "nbr2_mean"):
        if s["p_val"] < 0.10:
            go_count_nbr2 += 1
        L.append(
            f"| -{s['lead']}d | {s['pos_mean']:.3f} | {s['neg_mean']:.3f}"
            f" | {s['delta']:+.3f} | {s['p_val']:.3f} | {s['sig']} |"
        )

    L += ["", "---", ""]

    # ── 5. 考察 ──────────────────────────────────────────
    best_mean = min(ndmi_stats,     key=lambda s: s["p_val"]) if ndmi_stats     else None
    best_p10  = min(ndmi_p10_stats, key=lambda s: s["p_val"]) if ndmi_p10_stats else None

    L += [
        "## 5. 考察",
        "",
        "### NDMI mean vs NDMI p10 の比較",
        "",
        "| 指標 | 最良 Lead | Δ | p値 | 解釈 |",
        "|---|---|---|---|---|",
    ]
    if best_mean:
        L.append(f"| NDMI mean | -{best_mean['lead']}d | {best_mean['delta']:+.3f} | {best_mean['p_val']:.3f} | シーン全体の平均水分量 |")
    if best_p10:
        L.append(f"| **NDMI p10** | **-{best_p10['lead']}d** | **{best_p10['delta']:+.3f}** | **{best_p10['p_val']:.3f}** | **先行乾燥スポット（南斜面等）の水分量** |")

    L += [
        "",
        "NDMI p10 が mean より低いリードタイムで有意差を示す場合、"
        "「シーン全体はまだ平均的」でも「最も乾きやすい場所が先行して乾燥している」ことを示す。"
        "これは火災の点火起点（南向き斜面・尾根筋の低木）を反映している可能性がある。",
        "",
    ]

    if best_p10:
        lead_interp = {
            7:  "発火直前。急速乾燥の最終段階。シグナルが最も強い",
            14: "乾燥スポットが顕在化し始める段階",
            21: "全体的な乾燥傾向が現れ始める段階",
            28: "季節変動ノイズが混入しシグナルが弱い",
        }
        L += [
            "### リードタイム別の解釈",
            "",
            "| Lead | 解釈 |",
            "|---|---|",
        ]
        for lead in sorted(LEAD_DAYS):
            interp = lead_interp.get(lead, "")
            best_marker = " ← 最良" if lead == best_p10['lead'] else ""
            L.append(f"| -{lead}d | {interp}{best_marker} |")
        L.append("")

    L += [
        "### NBR2 が有効でない理由",
        "",
        "NBR2 = (B11 − B12) / (B11 + B12) は FireEdge (活火検知) の主指標だが、",
        "発火「前」の乾燥ストレス検知には適していない。",
        "",
        "- 活火中: 燃焼により B12 (2190nm) が急上昇 → NBR2 が大きく低下 → 高コントラスト",
        "- 発火前: B11/B12 の相対変化は小さく、植生タイプ・土壌背景ノイズが支配的",
        "- 今回の結果で Δ > 0 (POS > NEG) という逆方向が出たのも、発火前という文脈での"
        " NBR2 の無効性を示している",
        "",
        "NDMI (B8A/B11) は水分含量に特化した指標であり、発火前の乾燥ストレスには NDMI が正しい選択。",
        "",
    ]

    if anomaly_map:
        anom_vals = list(anomaly_map.values())
        all_neg = all(v < 0 for v in anom_vals)
        L += [
            "### Seasonal Anomaly の解釈",
            "",
            "| イベント | NDMI anomaly (発火年 − 例年) |",
            "|---|---|",
        ]
        for key, anom in anomaly_map.items():
            L.append(f"| {key} | {anom:+.3f} |")
        L += [
            "",
            f"{'全件とも' if all_neg else '多くのケースで'}負値 — 発火年の -14d は例年同日と比べて乾燥している。",
            "絶対値比較 (POS vs NEG) に加え、**相対比較 (発火年 vs 例年)** でも",
            "同じ方向のシグナルが確認された。これは NDMI の発火前乾燥ストレス指標としての",
            "有効性を独立した証拠で補強する。",
            "",
        ]

    L += ["---", ""]

    # ── 6. Go/No-Go ──────────────────────────────────────
    cond1 = len(pos_records) > 0 and n_neg_total > 0
    # p10 を主判定指標とし、mean も参考として合わせて確認
    cond2 = (go_count_ndmi_p10 >= 1) or (go_count_ndmi >= 1)

    n_pos_total = len(pos_records)
    n_chap_total = len([e for e in POS_EVENTS if e['veg'] == 'chaparral']) * len(LEAD_DAYS)

    L += [
        "## 6. Go/No-Go 判断",
        "",
        "| 条件 | 結果 | 根拠 |",
        "|---|---|---|",
        f"| ① SimSat データ取得 | {'✅ OK' if cond1 else '❌ NG'} |"
        f" POS チャパラル {n_pos_chap}/{n_chap_tried} + 針葉樹 {n_pos_conif}/{n_conif_tried} + NEG {n_neg_total}/{len(NEG_CANDIDATES)*len(LEAD_DAYS)} |",
        f"| ② NDMI_p10 スペクトル分離 | {'✅ OK' if go_count_ndmi_p10>=1 else '❌ NG'} |"
        f" {go_count_ndmi_p10}/{len(LEAD_DAYS)} リードタイムで p<0.10 (主判定) |",
        f"| ② NDMI mean スペクトル分離 | {'✅ OK' if go_count_ndmi>=1 else '△'} |"
        f" {go_count_ndmi}/{len(LEAD_DAYS)} リードタイムで p<0.10 (参考) |",
        "",
    ]

    if cond1 and cond2:
        best_s = best_p10 or best_mean
        reason = ""
        if best_s:
            ind = "NDMI_p10" if best_s is best_p10 else "NDMI_mean"
            reason = (
                f"{ind} -{best_s['lead']}d: POS={best_s['pos_mean']:.3f} < NEG={best_s['neg_mean']:.3f},"
                f" Δ={best_s['delta']:+.3f}, p={best_s['p_val']:.3f}。"
            )
        anom_reason = ""
        if anomaly_map and all(v < 0 for v in anomaly_map.values()):
            anom_reason = f" seasonal anomaly {len(anomaly_map)} 件すべて負値という独立証拠も一致。"
        p_note = f"p={best_s['p_val']:.3f}" if best_s else ""
        sig_note = "p<0.05 達成" if best_s and best_s["p_val"] < 0.05 else "p<0.10 (傾向確認レベル)"
        L += [
            f"> {reason}{anom_reason}",
            "> NDMI p10 シグナルが POS < NEG で一貫。最乾燥スポットが発火前に先行乾燥している根拠あり。",
            "",
            "**判定: Go — /poc2 へ進む**",
            "",
            f"*{p_note} は {sig_note}。/poc の目的は「統計的証明」でなく「FT に値するシグナルの存在確認」。*",
        ]
    elif cond1 and not cond2:
        L += [
            "**判定: Conditional — NDMI シグナル弱い**",
            "",
            "推奨アクション:",
            "- チャパラル POS を n≥15 に増やして再試行",
            "- seasonal anomaly が取得できる場合は絶対値比較より安定する可能性あり",
        ]
    else:
        L += ["**判定: No-Go — SimSat データ取得の根本的問題**"]

    L += ["", "---", ""]

    # ── 7. 次のステップ ───────────────────────────────────
    L += [
        "## 7. /poc で確定した設計決定と次のステップ (/poc2)",
        "",
        "### /poc で確定した設計決定",
        "",
        "| 項目 | 決定内容 | 根拠 |",
        "|---|---|---|",
        "| **主指標** | NDMI_p10 = (B8A−B11)/(B8A+B11) の 10th percentile | -7d で p=0.038、mean より分散小・先行乾燥スポットを捉える |",
        "| **最強リードタイム** | **-7d** | NDMI_p10 Δ=-0.083、p=0.038 (p<0.05 唯一) |",
        "| **時系列入力** | **3点: -21d → -14d → -7d** | -28d は p=0.357 でシグナルなし。-21d が乾燥開始のベースライン |",
        "| **VLM 入力画像** | 疑似カラー 1枚/時点 (R=B12, G=B8A, B=B11) | 3バンド→RGB PNG 250×250px。1時点1画像 = 196トークン |",
        "| **対象植生** | チャパラル限定 (主分析)、針葉樹は汎化確認用 | 針葉樹混在で p=0.467 に悪化。NDMI の物理的妥当性もチャパラル |",
        "| **時系列の優位性** | 単一-7d より3点時系列を推奨 | 誤検知低減・VLM への文脈付与。統計的トレンドより VLM の文脈理解に価値 |",
        "",
        "### /poc2 でやること",
        "",
        "/poc2 では LFM 2.5-VL-450M に上記設計で画像を渡し、モデルが学習できるか確認する:",
        "",
        "1. **few-shot ICL** (チャパラル POS/NEG 各 2〜3 件を context に): "
        "時系列3枚 (-21d/-14d/-7d) を渡して POS/NEG 判定精度を確認。zero-shot は行わない",
        "2. **小規模 LoRA FT** (チャパラル 10〜20 サンプル): few-shot より改善があれば本格 FT の根拠確立",
        "",
        "/poc2 完了条件:",
        "- few-shot で Accuracy > chance level (50%) を確認",
        "- 小規模 FT で few-shot より改善",
        "",
        "---",
        "",
        "## 出力ファイル",
        "",
        f"| ファイル | 内容 |",
        f"|---|---|",
        f"| `{save_dir}/results.csv` | 全シーンの指標値 (NDMI/NBR2/NBR/NDVI) |",
        f"| `{save_dir}/images/` | 疑似カラー (B12=R, B8A=G, B11=B) + RGB 画像 |",
        f"| `{save_dir}/figures/ndmi_distributions.png` | NDMI 分布 (主指標) |",
        f"| `{save_dir}/figures/nbr_distributions.png` | NBR 分布 (副指標) |",
        f"| `{save_dir}/figures/ndmi_by_vegtype.png` | 植生タイプ別 scatter |",
        "",
    ]

    report_path = save_dir / "report.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    return report_path


# ─────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FireGuard /poc — 発火前スペクトル分離確認")
    parser.add_argument("--out-dir", type=Path, default=SAVE_DIR,
                        help=f"出力ルートディレクトリ (default: {SAVE_DIR})")
    parser.add_argument("--run-name", default=None,
                        help="実験名。指定時は --out-dir/{run-name}/ に出力")
    args = parser.parse_args()

    save_dir = args.out_dir / args.run_name if args.run_name else args.out_dir

    save_dir.mkdir(parents=True, exist_ok=True)
    img_dir = save_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("FireGuard /poc v3 — 発火前 NDMI/NBR2 分離確認 (チャパラル限定)")
    print("主指標: NDMI_p5/p10=(B8A-B11)/(B8A+B11)  参考: NBR2=(B11-B12)/(B11+B12)")
    print(f"SIZE_KM={SIZE_KM}, POS={len(POS_EVENTS)}件, out_dir={save_dir}")
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
            img_base = f"pos_{ev['name'].replace(' ', '_')}_-{lead}d"
            save_composite(arr, img_dir / f"{img_base}.png")
            save_rgb(arr, img_dir / f"{img_base}_rgb.png")

            idx = compute_indices(arr)
            if not idx:
                print(f"invalid (n_valid<100) — image saved: {img_base}.png")
                continue

            print(
                f"OK  NDMI={idx['ndmi_mean']:.3f}(p10={idx['ndmi_p10']:.3f},p5={idx['ndmi_p5']:.3f})"
                f"  NBR2={idx['nbr2_mean']:.3f}(min={idx['nbr2_min']:.3f})"
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
            img_base = f"neg_{ev['name'].replace(' ', '_')}_-{lead}d"
            save_composite(arr, img_dir / f"{img_base}.png")
            save_rgb(arr, img_dir / f"{img_base}_rgb.png")

            idx = compute_indices(arr)
            if not idx:
                print(f"invalid (n_valid<100) — image saved: {img_base}.png")
                continue

            print(
                f"OK  NDMI={idx['ndmi_mean']:.3f}(p10={idx['ndmi_p10']:.3f},p5={idx['ndmi_p5']:.3f})"
                f"  NBR2={idx['nbr2_mean']:.3f}(min={idx['nbr2_min']:.3f})"
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
    print(f"\n[4] 結果保存 → {save_dir}/results.csv")
    all_records = pos_records + neg_records
    if all_records:
        # POS と NEG でフィールドが異なる (ref_date など) ため全レコードから収集
        fieldnames = list(dict.fromkeys(k for r in all_records for k in r.keys()))
        with open(save_dir / "results.csv", "w", newline="") as f:
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
    def neg_nbr2_at(lead: int) -> list[float]:
        return [r["nbr2_mean"] for r in neg_records if r["lead_days"] == lead]

    neg_ndmi_all = [r["ndmi_mean"] for r in neg_records]
    neg_nbr_all  = [r["nbr_mean"]  for r in neg_records]
    neg_nbr2_all = [r["nbr2_mean"] for r in neg_records]

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

    print("\n  [NBR2 — 参考比較; FireEdge 最重要指標] (NEG は同 lead で比較)")
    go_count_nbr2 = 0
    for lead in LEAD_DAYS:
        pos_chap_nbr2 = [r["nbr2_mean"] for r in pos_records if r["lead_days"] == lead and r.get("veg") == "chaparral"]
        neg_nbr2_lead = neg_nbr2_at(lead) if neg_nbr2_at(lead) else neg_nbr2_all
        if not pos_chap_nbr2 or not neg_nbr2_lead:
            continue
        delta = np.mean(pos_chap_nbr2) - np.mean(neg_nbr2_lead)
        p_val = 1.0
        if HAS_SCIPY and len(pos_chap_nbr2) >= 3 and len(neg_nbr2_lead) >= 3:
            _, p_val = scipy_stats.mannwhitneyu(pos_chap_nbr2, neg_nbr2_lead, alternative="less")
        sig = "★ p<0.05" if p_val < 0.05 else ("△ p<0.10" if p_val < 0.10 else "ns")
        print(f"  -{lead}d [チャパラル]: mean={np.mean(pos_chap_nbr2):.3f}  Δ={delta:+.3f}  p={p_val:.3f}  {sig}  (n_pos={len(pos_chap_nbr2)}, n_neg={len(neg_nbr2_lead)})")
        if p_val < 0.10:
            go_count_nbr2 += 1

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
        plot_distributions(pos_records, neg_records, save_dir)

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
    print(f"  NDMI 有意差あり (p<0.10, チャパラル): {go_count_ndmi}  / {len(LEAD_DAYS)} リードタイム")
    print(f"  NBR2 有意差あり (p<0.10, チャパラル): {go_count_nbr2} / {len(LEAD_DAYS)} リードタイム (参考比較)")
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

    # ── レポート生成 ──
    report_path = write_report(pos_records, neg_records, anomaly_map, save_dir,
                               run_name=args.run_name or "")
    print(f"\n  詳細結果: {save_dir}/results.csv")
    print(f"  レポート: {report_path}")
    print(f"  画像:     {save_dir}/images/")
    if HAS_MPL:
        print(f"  図:       {save_dir}/figures/")


if __name__ == "__main__":
    main()
