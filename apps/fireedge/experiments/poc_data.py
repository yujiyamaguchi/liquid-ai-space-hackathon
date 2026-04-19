"""
/poc — データ PoC スクリプト
==============================
CLAUDE.md /poc 完了条件:
  ① GT データソース (FIRMS) から対象イベントの座標・日時を取得し、
     SimSat がその座標・日時の Sentinel-2 データを返せること
  ② スペクトル的なシグナルが positive / negative クラスで分離できること
     (NBR2 / SWIR22 / fire_pixel_ratio の閾値が成立するか数値で確認)

Usage:
  cd apps/fireedge
  uv run python poc_data.py [--area AREA] [--days DAYS]

  AREA: "africa" (default) | "australia" | "amazon" | "global"
  DAYS: FIRMS 取得日数 (1–10, default 7)
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import requests

# -------------------------------------------------------------------
# パス解決 (apps/fireedge/ 直下で実行する前提)
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))  # apps/fireedge/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.data_fetcher import SimSatClient
from src.spectral import SpectralProcessor

# -------------------------------------------------------------------
# FIRMS NRT API 設定
# -------------------------------------------------------------------
FIRMS_BASE  = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_KEY   = os.environ.get("FIRMS_MAP_KEY", "")

AREAS = {
    # 常時火災が多いサブサハラ帯に絞る (全 Africa だと days>1 で 400)
    "africa":    "5,5,40,15",
    # オーストラリア北部 (乾季の火災シーズン)
    "australia": "125,-20,150,-10",
    # アマゾン
    "amazon":    "-70,-15,-45,5",
    # 東南アジア (焼畑)
    "seasia":    "95,5,140,25",
}

SIMSAT_BASE = "http://localhost:9005"

# 火災シグナル閾値 (spectral.py と一致)
NBR2_FIRE_THRESHOLD  = -0.05   # 以下 → 火災疑い
SWIR22_FIRE_MIN      = 0.15    # 以上 → 火災疑い
FIRE_PIXEL_RATIO_MIN = 0.01    # シーン内1%以上が火災画素なら positive

# negative サンプル用オフセット [度]
NEGATIVE_LAT_OFFSET = 2.0


# -------------------------------------------------------------------
# データクラス
# -------------------------------------------------------------------
@dataclass
class FIRMSEvent:
    lat: float
    lon: float
    frp: float
    brightness: float
    confidence: str
    acq_date: str
    acq_time: str

    @property
    def iso_datetime(self) -> str:
        hh = self.acq_time[:2]
        mm = self.acq_time[2:] if len(self.acq_time) >= 4 else "00"
        return f"{self.acq_date}T{hh}:{mm}:00Z"


@dataclass
class SpectralResult:
    label: str        # "positive" or "negative"
    lat: float
    lon: float
    timestamp: str
    image_available: bool
    cloud_cover: float
    source: str
    nbr2: float | None
    swir22: float | None
    fire_pixel_ratio: float | None
    nbr2_verdict: str   # "fire" / "no fire" / "n/a"
    swir22_verdict: str
    image_datetime: str


# -------------------------------------------------------------------
# FIRMS 取得
# -------------------------------------------------------------------
def fetch_firms_events(area: str, days: int) -> list[FIRMSEvent]:
    """FIRMS NRT VIIRS_SNPP から指定エリア・日数分の火災ホットスポットを取得する。"""
    area_str = AREAS.get(area, AREAS["africa"])
    url = f"{FIRMS_BASE}/{FIRMS_KEY}/VIIRS_SNPP_NRT/{area_str}/{days}"
    print(f"\n[FIRMS] GET {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    events: list[FIRMSEvent] = []
    reader = csv.DictReader(io.StringIO(r.text))
    for row in reader:
        try:
            events.append(FIRMSEvent(
                lat=float(row.get("latitude", 0)),
                lon=float(row.get("longitude", 0)),
                frp=float(row.get("frp", 0) or 0),
                brightness=float(row.get("bright_ti4") or row.get("brightness") or 0),
                confidence=str(row.get("confidence", "nominal")).strip().lower(),
                acq_date=row.get("acq_date", ""),
                acq_time=str(row.get("acq_time", "0000")).zfill(4),
            ))
        except (ValueError, KeyError):
            continue

    print(f"[FIRMS] {len(events)} ホットスポット取得")
    return events


def select_top_events(events: list[FIRMSEvent], n: int = 5) -> list[FIRMSEvent]:
    """FRP 降順・confidence='high' 優先で上位 n 件を選ぶ。"""
    high  = [e for e in events if e.confidence == "h"]
    other = [e for e in events if e.confidence != "h"]
    ranked = sorted(high, key=lambda e: e.frp, reverse=True) + \
             sorted(other, key=lambda e: e.frp, reverse=True)
    return ranked[:n]


# -------------------------------------------------------------------
# SimSat 取得
# -------------------------------------------------------------------
def query_simsat(
    client: SimSatClient,
    lat: float,
    lon: float,
    timestamp: str,
    label: str,
) -> SpectralResult:
    """SimSat から指定座標・日時の Sentinel-2 データを取得し、スペクトル指標を計算する。"""
    print(f"  → SimSat [{label}] lat={lat:.3f} lon={lon:.3f} ts={timestamp}")
    try:
        response = client.fetch_fire_scene(lon=lon, lat=lat, timestamp=timestamp)
    except Exception as e:
        print(f"     ERROR: {e}")
        return SpectralResult(
            label=label, lat=lat, lon=lon, timestamp=timestamp,
            image_available=False, cloud_cover=0.0, source="error",
            nbr2=None, swir22=None, fire_pixel_ratio=None,
            nbr2_verdict="n/a", swir22_verdict="n/a", image_datetime="",
        )

    if not response.image_available or response.image_array is None:
        print(f"     image_available=False  source={response.source}  cc={response.cloud_cover:.1f}%")
        return SpectralResult(
            label=label, lat=lat, lon=lon, timestamp=timestamp,
            image_available=False, cloud_cover=response.cloud_cover, source=response.source,
            nbr2=None, swir22=None, fire_pixel_ratio=None,
            nbr2_verdict="n/a", swir22_verdict="n/a", image_datetime=response.datetime,
        )

    processor = SpectralProcessor()
    scene = processor.process(response)
    idx = scene.indices

    nbr2_verdict  = "🔥 fire"  if idx.nbr2 < NBR2_FIRE_THRESHOLD  else "🌿 no fire"
    swir22_verdict = "🔥 fire" if idx.mean_swir22 > SWIR22_FIRE_MIN else "🌿 no fire"

    print(f"     ✅ image_available=True  cc={response.cloud_cover:.1f}%  "
          f"NBR2={idx.nbr2:+.3f}  SWIR22={idx.mean_swir22:.3f}  "
          f"fire_px={idx.fire_pixel_ratio:.3f}")

    return SpectralResult(
        label=label, lat=lat, lon=lon, timestamp=timestamp,
        image_available=True, cloud_cover=response.cloud_cover, source=response.source,
        nbr2=idx.nbr2, swir22=idx.mean_swir22,
        fire_pixel_ratio=idx.fire_pixel_ratio,
        nbr2_verdict=nbr2_verdict, swir22_verdict=swir22_verdict,
        image_datetime=response.datetime,
    )


# -------------------------------------------------------------------
# レポート
# -------------------------------------------------------------------
def print_report(pairs: list[tuple[SpectralResult, SpectralResult]]) -> None:
    sep = "─" * 90
    print(f"\n\n{'='*90}")
    print("  /poc 結果レポート — スペクトルシグナル分離確認")
    print(f"{'='*90}")

    go_conditions = []

    for i, (pos, neg) in enumerate(pairs, 1):
        print(f"\n【Event {i}】")
        print(f"  Positive (fire)   : lat={pos.lat:.3f} lon={pos.lon:.3f}  "
              f"SimSat={pos.image_datetime or 'N/A'}  cc={pos.cloud_cover:.1f}%")
        print(f"  Negative (no fire): lat={neg.lat:.3f} lon={neg.lon:.3f}  "
              f"SimSat={neg.image_datetime or 'N/A'}  cc={neg.cloud_cover:.1f}%")
        print(f"  {sep[:60]}")

        if not pos.image_available:
            print("  ⚠️  Positive: SimSat から画像取得不可 → このイベントはスキップ")
            go_conditions.append(False)
            continue

        print(f"  {'指標':<20} {'Positive':>12} {'Negative':>12}  {'判定':>12}")
        print(f"  {sep[:60]}")

        # NBR2
        neg_nbr2_str  = f"{neg.nbr2:+.3f}" if neg.nbr2 is not None else "n/a"
        neg_sw_str    = f"{neg.swir22:.3f}" if neg.swir22 is not None else "n/a"
        neg_fp_str    = f"{neg.fire_pixel_ratio:.3f}" if neg.fire_pixel_ratio is not None else "n/a"

        nbr2_sep  = pos.nbr2 is not None and (neg.nbr2 is None or pos.nbr2 < neg.nbr2 - 0.02)
        swir_sep  = pos.swir22 is not None and (neg.swir22 is None or pos.swir22 > neg.swir22 + 0.02)
        fp_sep    = pos.fire_pixel_ratio is not None and (neg.fire_pixel_ratio is None or pos.fire_pixel_ratio > neg.fire_pixel_ratio)

        print(f"  {'NBR2':<20} {pos.nbr2:+12.3f} {neg_nbr2_str:>12}  "
              f"{'✅ 分離' if nbr2_sep else '❌ 未分離':>12}")
        print(f"  {'mean SWIR22':<20} {pos.swir22:12.3f} {neg_sw_str:>12}  "
              f"{'✅ 分離' if swir_sep else '❌ 未分離':>12}")
        print(f"  {'fire_pixel_ratio':<20} {pos.fire_pixel_ratio:12.3f} {neg_fp_str:>12}  "
              f"{'✅ 分離' if fp_sep else '❌ 未分離':>12}")
        print(f"  {'NBR2 verdict':<20} {pos.nbr2_verdict:>12} {neg.nbr2_verdict if neg.nbr2_verdict else 'n/a':>12}")

        signal_ok = nbr2_sep or swir_sep
        go_conditions.append(signal_ok)

    # --- Go/No-Go 判定 ---
    n_go = sum(go_conditions)
    n_total = len(go_conditions)
    print(f"\n{'='*90}")
    print(f"  Go/No-Go 判定: {n_go}/{n_total} イベントでシグナル分離を確認")
    if n_go >= 1:
        print("  ✅ GO — /poc2 (Model PoC) に進む")
    else:
        print("  ❌ NO-GO — シグナル分離が不十分。バンド構成・閾値の見直しが必要")
    print(f"{'='*90}\n")


# -------------------------------------------------------------------
# エントリポイント
# -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="/poc データ検証スクリプト")
    parser.add_argument("--area", default="africa",
                        choices=list(AREAS.keys()), help="FIRMS 対象エリア")
    parser.add_argument("--days", type=int, default=3, help="FIRMS 取得日数 (1-5推奨; エリアが大きいと上限あり)")
    parser.add_argument("--top", type=int, default=5, help="検証するイベント数")
    args = parser.parse_args()

    if not FIRMS_KEY:
        print("ERROR: 環境変数 FIRMS_MAP_KEY が未設定です。.env を確認してください。")
        sys.exit(1)

    print("=" * 70)
    print("  FireEdge /poc — データ PoC")
    print(f"  FIRMS エリア: {args.area}  過去 {args.days} 日  上位 {args.top} イベント")
    print("=" * 70)

    # ① FIRMS から火災イベント取得
    events = fetch_firms_events(args.area, args.days)
    if not events:
        print("ERROR: FIRMS から火災イベントを取得できませんでした。")
        sys.exit(1)

    top_events = select_top_events(events, n=args.top)
    print(f"\n[選択] FRP 上位 {len(top_events)} イベント:")
    for i, e in enumerate(top_events, 1):
        print(f"  {i}. lat={e.lat:.3f} lon={e.lon:.3f}  FRP={e.frp:.1f} MW  "
              f"confidence={e.confidence}  datetime={e.acq_date}T{e.acq_time[:2]}:{e.acq_time[2:]}Z")

    # ② SimSat に問い合わせ
    client = SimSatClient(base_url=SIMSAT_BASE)
    pairs: list[tuple[SpectralResult, SpectralResult]] = []

    for i, event in enumerate(top_events, 1):
        print(f"\n[Event {i}/{len(top_events)}] 検証中 ...")
        # SimSat は timestamp として「いつ頃の画像を探すか」を受け取る。
        # FIRMS の取得日時を渡し、window_seconds=864000 (10日) で最近の画像を探す。
        pos_ts  = event.iso_datetime
        pos = query_simsat(client, event.lat, event.lon, pos_ts, label="positive")

        # Negative: 同じ longitude、lat を +2度ずらす
        neg_lat = event.lat + NEGATIVE_LAT_OFFSET
        neg = query_simsat(client, neg_lat, event.lon, pos_ts, label="negative")

        pairs.append((pos, neg))
        time.sleep(1.0)  # SimSat API の過負荷回避

    # ③ レポート
    print_report(pairs)


if __name__ == "__main__":
    # .env から環境変数を読み込む
    env_path = os.path.join(os.path.dirname(SCRIPT_DIR), "..", ".env")
    env_path = os.path.normpath(os.path.join(ROOT_DIR, "../../.env"))
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    main()
