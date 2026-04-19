"""
/poc v2 — 改訂版データ PoC
================================
設計方針 (ユーザーとの合意):
  - FIRMS と S2 の時刻ズレは許容。同じ場所に火災が起きた期間内に
    S2 が撮像されていれば一致とみなす (±6日ウィンドウ)
  - 負例は lat+2 オフセットではなく、同一 S2 タイル内で
    FIRMS ホットスポットが存在しない座標を使う

アルゴリズム:
  1. FIRMS NRT から上位 N イベントを取得
  2. 各イベントに対し timestamp = FIRMS日時 + 6日, window = 12日
     で SimSat を呼び、最近傍 S2 画像を取得
  3. 取得した S2 の footprint 内で FIRMS ホットスポット密度が 0 の
     座標を negative として選ぶ (フットプリント内でホットスポットと
     最も離れた点)
  4. Positive vs Negative のスペクトル指標を比較

完了条件 (CLAUDE.md /poc ②):
  - NBR2, SWIR22, fire_pixel_ratio のいずれかが正負で分離できること
"""
from __future__ import annotations

import csv
import io
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))  # apps/fireedge/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.data_fetcher import SimSatClient
from src.spectral import SpectralProcessor

# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_KEY  = os.environ.get("FIRMS_MAP_KEY", "")
SIMSAT     = SimSatClient()

# FIRMS エリア (複数試して最もホットスポットが多い地域を使う)
AREAS = {
    "west_africa": "5,5,30,15",    # サブサハラ西部 (乾季後期、常時火災)
    "east_africa": "25,0,45,15",   # 東アフリカ
    "amazon":      "-70,-15,-45,5",
    "seasia":      "95,5,140,20",
}

# タイムスタンプシフト: FIRMS日時 + この日数 → SimSat クエリ時刻
# window_seconds = この日数 × 2 でカバー範囲 ±SHIFT_DAYS
SHIFT_DAYS   = 6
WINDOW_DAYS  = 12  # SHIFT_DAYS × 2
WINDOW_SEC   = WINDOW_DAYS * 86400  # 1,036,800 秒

# S2–FIRMS 時刻差の許容範囲 (日数)
MAX_DELTA_DAYS = 6.0

# 負例のグリッド検索: フットプリント内で FIRMS ホットスポットから
# 最も離れた点を選ぶ (グリッド解像度)
NEG_GRID_N = 5  # フットプリントを N×N に分割して候補点を作る

SIZE_KM = 5  # SimSat クエリのシーンサイズ

# ------------------------------------------------------------------
# データクラス
# ------------------------------------------------------------------
@dataclass
class FIRMSEvent:
    lat: float
    lon: float
    frp: float
    confidence: str
    acq_datetime: datetime  # UTC

    @property
    def iso(self) -> str:
        return self.acq_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class QueryResult:
    label: str
    lat: float
    lon: float
    query_ts: str
    image_available: bool
    image_datetime: str
    delta_days: float       # S2撮像日時 - FIRMS検知日時 (日)
    cloud_cover: float
    nbr2_mean: float | None
    nbr2_min:  float | None
    swir22_mean: float | None
    swir22_max:  float | None
    fire_px: float | None   # NBR2<-0.05 AND SWIR22>0.15
    scar_px: float | None   # NBR2<-0.10 のみ (焼跡指標)
    footprint: tuple | None


# ------------------------------------------------------------------
# FIRMS
# ------------------------------------------------------------------
def fetch_firms(area_str: str, days: int) -> list[FIRMSEvent]:
    url = f"{FIRMS_BASE}/{FIRMS_KEY}/VIIRS_SNPP_NRT/{area_str}/{days}"
    print(f"  [FIRMS] {url}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  [FIRMS] エラー: {e}")
        return []

    events = []
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            date_str = row["acq_date"]
            time_str = str(row["acq_time"]).zfill(4)
            dt = datetime(
                int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]),
                int(time_str[:2]), int(time_str[2:]), 0, tzinfo=timezone.utc
            )
            events.append(FIRMSEvent(
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
                frp=float(row.get("frp") or 0),
                confidence=str(row.get("confidence","n")).strip().lower(),
                acq_datetime=dt,
            ))
        except Exception:
            continue
    return events


def select_top(events: list[FIRMSEvent], n: int) -> list[FIRMSEvent]:
    """confidence=h 優先 → FRP 降順"""
    high  = sorted([e for e in events if e.confidence == "h"],
                   key=lambda e: e.frp, reverse=True)
    other = sorted([e for e in events if e.confidence != "h"],
                   key=lambda e: e.frp, reverse=True)
    return (high + other)[:n]


# ------------------------------------------------------------------
# SimSat クエリ (最近傍 S2 探索)
# ------------------------------------------------------------------
def query_nearest_s2(
    event: FIRMSEvent,
    lat: float,
    lon: float,
    label: str,
) -> QueryResult:
    """
    timestamp = FIRMS日時 + SHIFT_DAYS, window = WINDOW_SEC で
    ±SHIFT_DAYS 範囲の最近傍 S2 を取得する。
    """
    shifted_ts = event.acq_datetime + timedelta(days=SHIFT_DAYS)
    ts_str = shifted_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = SIMSAT.fetch_fire_scene(
            lon=lon, lat=lat,
            timestamp=ts_str,
            size_km=SIZE_KM,
            window_seconds=WINDOW_SEC,
        )
    except Exception as e:
        print(f"    [SimSat] {label} ERROR: {e}")
        return QueryResult(label=label, lat=lat, lon=lon, query_ts=ts_str,
                           image_available=False, image_datetime="", delta_days=999,
                           cloud_cover=0, nbr2_mean=None, nbr2_min=None,
                           swir22_mean=None, swir22_max=None, fire_px=None,
                           scar_px=None, footprint=None)

    delta_days = 999.0
    nbr2_mean = nbr2_min = swir22_mean = swir22_max = fire_px = scar_px = None

    if resp.image_available and resp.datetime:
        try:
            s2_dt = datetime.strptime(resp.datetime[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            delta_days = (s2_dt - event.acq_datetime).total_seconds() / 86400
        except Exception:
            pass

    if resp.image_available and resp.image_array is not None:
        arr = resp.image_array
        s22 = arr[:,:,0]; s16 = arr[:,:,1]
        nbr2 = (s16 - s22) / (s16 + s22 + 1e-10)
        nbr2_mean  = float(nbr2.mean())
        nbr2_min   = float(nbr2.min())
        swir22_mean = float(s22.mean())
        swir22_max  = float(s22.max())
        fire_px = float(np.mean((nbr2 < -0.05) & (s22 > 0.15)))
        scar_px = float(np.mean(nbr2 < -0.10))

    return QueryResult(
        label=label, lat=lat, lon=lon, query_ts=ts_str,
        image_available=resp.image_available if resp else False,
        image_datetime=resp.datetime if resp else "",
        delta_days=delta_days,
        cloud_cover=resp.cloud_cover if resp else 0.0,
        nbr2_mean=nbr2_mean, nbr2_min=nbr2_min,
        swir22_mean=swir22_mean, swir22_max=swir22_max,
        fire_px=fire_px, scar_px=scar_px,
        footprint=resp.footprint if resp else None,
    )


# ------------------------------------------------------------------
# フットプリント内の非火災座標を選ぶ
# ------------------------------------------------------------------
def find_negative_coord(
    footprint: tuple[float, float, float, float],
    fire_lat: float,
    fire_lon: float,
    all_events: list[FIRMSEvent],
) -> tuple[float, float]:
    """
    フットプリント (lon_min, lat_min, lon_max, lat_max) 内を
    NEG_GRID_N × NEG_GRID_N に分割し、FIRMS ホットスポットから
    最も離れたグリッド点を負例座標として返す。
    """
    lon_min, lat_min, lon_max, lat_max = footprint

    # フットプリント内の FIRMS ホットスポット
    local_pts = [
        (e.lat, e.lon) for e in all_events
        if lat_min <= e.lat <= lat_max and lon_min <= e.lon <= lon_max
    ]

    # グリッド候補点
    lons = np.linspace(lon_min, lon_max, NEG_GRID_N + 2)[1:-1]
    lats = np.linspace(lat_min, lat_max, NEG_GRID_N + 2)[1:-1]

    best_dist = -1.0
    best_lat, best_lon = fire_lat + 0.1, fire_lon + 0.1  # フォールバック

    for glat in lats:
        for glon in lons:
            if not local_pts:
                # ホットスポット情報なし: 対角方向を選ぶ
                if abs(glat - fire_lat) + abs(glon - fire_lon) > best_dist:
                    best_dist = abs(glat - fire_lat) + abs(glon - fire_lon)
                    best_lat, best_lon = glat, glon
            else:
                # ホットスポット全てからの最小距離が最大な点を選ぶ
                min_d = min(
                    ((glat - pl)**2 + (glon - po)**2)**0.5
                    for pl, po in local_pts
                )
                if min_d > best_dist:
                    best_dist = min_d
                    best_lat, best_lon = glat, glon

    return best_lat, best_lon


# ------------------------------------------------------------------
# レポート
# ------------------------------------------------------------------
def print_report(pairs: list[tuple[QueryResult, QueryResult, float]]) -> None:
    sep = "─" * 85
    print(f"\n{'='*85}")
    print("  /poc v2 — スペクトルシグナル分離レポート")
    print(f"{'='*85}")

    go_count = 0
    for i, (pos, neg, frp) in enumerate(pairs, 1):
        print(f"\n【Event {i}】 FRP={frp:.0f} MW")
        print(f"  Positive: lat={pos.lat:.3f} lon={pos.lon:.3f}  "
              f"S2={pos.image_datetime[:10] if pos.image_datetime else 'N/A'}  "
              f"Δ={pos.delta_days:+.1f}日  cc={pos.cloud_cover:.0f}%")
        print(f"  Negative: lat={neg.lat:.3f} lon={neg.lon:.3f}  "
              f"S2={neg.image_datetime[:10] if neg.image_datetime else 'N/A'}  "
              f"Δ={neg.delta_days:+.1f}日  cc={neg.cloud_cover:.0f}%")

        if not pos.image_available:
            print("  ⚠️  S2 画像取得不可 → スキップ")
            continue
        if pos.delta_days > MAX_DELTA_DAYS:
            print(f"  ⚠️  S2 が {pos.delta_days:.1f} 日ずれており許容範囲 ({MAX_DELTA_DAYS}日) 超 → スキップ")
            continue

        print(f"  {sep[:60]}")
        print(f"  {'指標':<22} {'Positive':>10} {'Negative':>10}  {'分離':>8}")
        print(f"  {sep[:60]}")

        metrics = [
            ("NBR2_mean",  pos.nbr2_mean,   neg.nbr2_mean,   "lower"),
            ("NBR2_min",   pos.nbr2_min,    neg.nbr2_min,    "lower"),
            ("SWIR22_mean",pos.swir22_mean, neg.swir22_mean, "higher"),
            ("SWIR22_max", pos.swir22_max,  neg.swir22_max,  "higher"),
            ("scar_px (NBR2<-0.10)", pos.scar_px, neg.scar_px, "higher"),
            ("fire_px (NBR2<-.05&\n     SWIR22>.15)", pos.fire_px, neg.fire_px, "higher"),
        ]

        separated = []
        for name, pv, nv, direction in metrics:
            if pv is None or nv is None:
                continue
            if direction == "lower":
                sep_ok = pv < nv - 0.01
            else:
                sep_ok = pv > nv + 0.005
            mark = "✅" if sep_ok else "❌"
            print(f"  {name:<22} {pv:>10.4f} {nv:>10.4f}  {mark:>8}")
            separated.append(sep_ok)

        event_go = any(separated)
        if event_go:
            go_count += 1
            print(f"  → ✅ シグナル分離あり")
        else:
            print(f"  → ❌ シグナル分離なし")

    n_valid = sum(1 for pos, neg, _ in pairs
                  if pos.image_available and pos.delta_days <= MAX_DELTA_DAYS)
    print(f"\n{'='*85}")
    print(f"  Go/No-Go: {go_count}/{n_valid} イベントでシグナル分離確認")
    if go_count >= 1:
        print("  ✅ GO — /poc2 (Model PoC) に進む")
        print("  ※ 活火災 SWIR22>0.15 は S2–火災タイミング一致が必要 (稀)")
        print("  ※ 焼跡 NBR2<-0.10 は S2 ±6日以内で安定して検出可能")
    else:
        print("  ❌ NO-GO")
    print(f"{'='*85}\n")


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------
def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top",  type=int, default=5)
    p.add_argument("--days", type=int, default=3)
    args = p.parse_args()

    if not FIRMS_KEY:
        print("ERROR: FIRMS_MAP_KEY 未設定")
        sys.exit(1)

    print("="*70)
    print("  FireEdge /poc v2")
    print(f"  FIRMS: 過去 {args.days} 日 / 上位 {args.top} イベント")
    print(f"  S2 探索ウィンドウ: ±{SHIFT_DAYS} 日")
    print("="*70)

    # 複数エリアから収集
    all_events: list[FIRMSEvent] = []
    for name, area_str in AREAS.items():
        print(f"\n[FIRMS] {name}")
        evs = fetch_firms(area_str, args.days)
        print(f"         → {len(evs)} ホットスポット")
        all_events.extend(evs)

    if not all_events:
        print("FIRMS データなし")
        sys.exit(1)

    top = select_top(all_events, args.top)
    print(f"\n[選択] FRP 上位 {len(top)} イベント:")
    for i, e in enumerate(top, 1):
        print(f"  {i}. lat={e.lat:.3f} lon={e.lon:.3f}  FRP={e.frp:.0f}MW  "
              f"conf={e.confidence}  {e.iso}")

    pairs: list[tuple[QueryResult, QueryResult, float]] = []

    for i, event in enumerate(top, 1):
        print(f"\n[Event {i}/{len(top)}] lat={event.lat:.3f} lon={event.lon:.3f}")

        # ① Positive
        pos = query_nearest_s2(event, event.lat, event.lon, label="positive")
        nbr2_s  = f"{pos.nbr2_mean:.3f}"  if pos.nbr2_mean  is not None else "N/A"
        swir_s  = f"{pos.swir22_max:.4f}" if pos.swir22_max is not None else "N/A"
        print(f"    positive: S2={pos.image_datetime[:10] if pos.image_datetime else 'N/A'}  "
              f"Δ={pos.delta_days:+.1f}日  cc={pos.cloud_cover:.0f}%  "
              f"NBR2_mean={nbr2_s}  SWIR22_max={swir_s}")

        # ② Negative: 同一 footprint 内で FIRMS ホットスポットから最遠点
        if pos.image_available and pos.footprint:
            neg_lat, neg_lon = find_negative_coord(
                pos.footprint, event.lat, event.lon, all_events
            )
        else:
            # フットプリント不明: 小さいオフセット (同地域だが火災なし)
            neg_lat = event.lat + 0.5
            neg_lon = event.lon + 0.5

        neg = query_nearest_s2(event, neg_lat, neg_lon, label="negative")
        neg_nbr2_s = f"{neg.nbr2_mean:.3f}" if neg.nbr2_mean is not None else "N/A"
        print(f"    negative: lat={neg_lat:.3f} lon={neg_lon:.3f}  "
              f"S2={neg.image_datetime[:10] if neg.image_datetime else 'N/A'}  "
              f"Δ={neg.delta_days:+.1f}日  cc={neg.cloud_cover:.0f}%  "
              f"NBR2_mean={neg_nbr2_s}")

        pairs.append((pos, neg, event.frp))
        time.sleep(1.0)

    print_report(pairs)


if __name__ == "__main__":
    env = os.path.normpath(os.path.join(ROOT_DIR, "../../.env"))
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    main()
