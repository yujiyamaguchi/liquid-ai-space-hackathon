"""
FireEdge /build Dataset Builder
=================================
FIRMS API → SimSat → SWIR composite → Train/Val/Test JSONL

データ設計 (/poc2 汎化確認の結果に基づく):
  POS          : FIRMS VIIRS_SNPP_NRT 検知座標 → SimSat (検知日+SHIFT_DAYS, Δ≥0)
  NEG temporal : 同座標 → SimSat (検知日-NEG_OFFSET_DAYS)  ← poc2 と同方式
  NEG diverse  : FP-prone バイオーム 45件 (砂漠10, 都市8, 湿地8, サバンナ8, 温帯森林6, 農地5)

分割:
  Train 70% / Val 15% / Test 15%  (stratified by label)
  Generalization test: DIVERSE_NEG_LOCATIONS 16地点 (データセットとは独立)

使い方:
    cd apps/fireedge
    uv run python -m finetune.dataset_builder
    uv run python -m finetune.dataset_builder --n-pos 100 --save-dir data/build/dataset
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import csv
import io

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_fetcher import SimSatClient
from src.interfaces import (
    FIRE_DETECTION_SYSTEM_PROMPT,
    FIRE_DETECTION_USER_PROMPT,
)
from src.spectral import SpectralProcessor

# ===========================================================================
# 定数
# ===========================================================================

SHIFT_DAYS      = 2    # POS: FIRMS検知日 + N日後 (burn scar が安定)
NEG_OFFSET_DAYS = 180  # NEG temporal: 検知日 - N日前 (fire season 外)
WINDOW_SEC      = 12 * 86400  # SimSat 検索ウィンドウ ±12日
SIZE_KM         = 5.0         # シーンサイズ

# FIRMS Area CSV API の制約:
#   - 最大取得日数: 5日 (NRT) ← "Invalid day range. Expects [1..5]." で確認済み
#   - バウンディングボックス: 過大だと400エラーになる場合あり
#     → poc2_icl.py で実績のある小さなボックスを流用し、エリア数を増やして補う
FIRMS_AREAS = {
    # poc2_icl.py で実績済み
    "west_africa":  "5,5,30,15",
    "east_africa":  "25,0,45,15",
    "seasia":       "95,5,140,20",
    "amazon":       "-70,-15,-45,5",
    # 追加エリア (poc2 と同程度のボックスサイズ)
    "australia":    "130,-35,155,-15",
    "cent_africa":  "10,-10,35,5",
    "us_west":      "-125,35,-105,50",
    "siberia":      "80,50,130,65",
}
FIRMS_PRODUCT = "VIIRS_SNPP_NRT"

# poc2 汎化確認でFP-proneだったカテゴリを優先的に追加
# 各エントリ: desc, lat, lon, ts
DIVERSE_NEG_LOCATIONS: list[dict] = [
    # 砂漠・乾燥地 (10件) — poc2 FP率 67%
    {"desc": "Sahara Algeria",            "lat": 23.0,  "lon":   5.0,  "ts": "2026-03-15T10:00:00Z"},
    {"desc": "Sahara Libya",              "lat": 26.0,  "lon":  14.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Arabian Peninsula KSA",     "lat": 24.0,  "lon":  45.0,  "ts": "2026-03-15T08:00:00Z"},
    {"desc": "Arabian Peninsula UAE",     "lat": 24.5,  "lon":  54.5,  "ts": "2026-03-15T07:00:00Z"},
    {"desc": "Australian outback SA",     "lat":-25.0,  "lon": 135.0,  "ts": "2026-03-15T02:00:00Z"},
    {"desc": "Australian outback NT",     "lat":-22.0,  "lon": 133.0,  "ts": "2026-03-15T01:00:00Z"},
    {"desc": "Atacama Desert Chile",      "lat":-24.0,  "lon": -69.5,  "ts": "2026-03-15T14:00:00Z"},
    {"desc": "Gobi Desert Mongolia",      "lat": 44.0,  "lon": 106.0,  "ts": "2026-03-15T04:00:00Z"},
    {"desc": "Namib Desert Namibia",      "lat":-22.5,  "lon":  15.0,  "ts": "2026-03-15T10:00:00Z"},
    {"desc": "Iranian plateau",           "lat": 32.0,  "lon":  55.0,  "ts": "2026-03-15T07:00:00Z"},

    # 都市郊外 (8件) — poc2 FP率 50%
    {"desc": "London suburbs",            "lat": 51.5,  "lon":  -0.1,  "ts": "2026-03-15T11:00:00Z"},
    {"desc": "Paris suburbs",             "lat": 48.8,  "lon":   2.2,  "ts": "2026-03-15T11:00:00Z"},
    {"desc": "Chicago suburbs",           "lat": 41.9,  "lon": -87.7,  "ts": "2026-03-15T17:00:00Z"},
    {"desc": "Seoul suburbs",             "lat": 37.5,  "lon": 127.0,  "ts": "2026-03-15T02:00:00Z"},
    {"desc": "Sydney suburbs",            "lat":-33.8,  "lon": 151.0,  "ts": "2026-03-15T00:00:00Z"},
    {"desc": "Buenos Aires suburbs",      "lat":-34.6,  "lon": -58.5,  "ts": "2026-03-15T14:00:00Z"},
    {"desc": "Istanbul suburbs",          "lat": 41.0,  "lon":  29.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Mumbai suburbs",            "lat": 19.1,  "lon":  73.0,  "ts": "2026-03-15T07:00:00Z"},

    # 湿地・デルタ (8件) — poc2 FP率 100% (1/1)
    {"desc": "Bangladesh Ganges delta",   "lat": 22.5,  "lon":  90.5,  "ts": "2026-03-15T05:00:00Z"},
    {"desc": "Mekong delta Vietnam",      "lat":  9.5,  "lon": 105.5,  "ts": "2026-03-15T04:00:00Z"},
    {"desc": "Nile delta Egypt",          "lat": 31.0,  "lon":  31.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Okavango delta Botswana",   "lat":-19.5,  "lon":  23.0,  "ts": "2026-03-15T10:00:00Z"},
    {"desc": "Danube delta Romania",      "lat": 45.0,  "lon":  29.5,  "ts": "2026-03-15T10:00:00Z"},
    {"desc": "Pantanal Brazil (wet)",     "lat":-17.0,  "lon": -57.0,  "ts": "2026-03-15T13:00:00Z"},
    {"desc": "Sudd wetland South Sudan",  "lat":  7.5,  "lon":  30.5,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Sundarbans mangrove",       "lat": 21.9,  "lon":  89.2,  "ts": "2026-03-15T05:00:00Z"},

    # サバンナ非火災期 (8件) — poc2 FP率 100% (1/1)
    {"desc": "Kenya savanna rainy",       "lat":  1.0,  "lon":  37.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Tanzania Serengeti rainy",  "lat": -2.5,  "lon":  35.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Zambia plateau rainy",      "lat":-14.0,  "lon":  28.0,  "ts": "2026-03-15T10:00:00Z"},
    {"desc": "Colombian Llanos rainy",    "lat":  5.0,  "lon": -70.0,  "ts": "2026-03-15T14:00:00Z"},
    {"desc": "Venezuela Llanos rainy",    "lat":  7.0,  "lon": -66.0,  "ts": "2026-03-15T14:00:00Z"},
    {"desc": "Myanmar dry zone",          "lat": 21.0,  "lon":  95.5,  "ts": "2026-03-15T04:00:00Z"},
    {"desc": "Indian Deccan plateau",     "lat": 17.5,  "lon":  78.0,  "ts": "2026-03-15T07:00:00Z"},
    {"desc": "Brazilian Cerrado rainy",   "lat":-12.0,  "lon": -46.0,  "ts": "2026-03-15T13:00:00Z"},

    # 温帯森林 (6件) — poc2 FP率 33% (Germany)
    {"desc": "Germany Black Forest",      "lat": 48.0,  "lon":   8.2,  "ts": "2026-03-15T11:00:00Z"},
    {"desc": "Canada boreal Manitoba",    "lat": 55.0,  "lon":-100.0,  "ts": "2026-03-15T18:00:00Z"},
    {"desc": "Brazil deep Amazon",        "lat": -2.0,  "lon": -62.0,  "ts": "2026-03-15T15:00:00Z"},
    {"desc": "Scandinavia boreal Norway", "lat": 63.0,  "lon":  13.0,  "ts": "2026-03-15T11:00:00Z"},
    {"desc": "Carpathian forest Romania", "lat": 46.0,  "lon":  25.0,  "ts": "2026-03-15T10:00:00Z"},
    {"desc": "New Zealand temperate",     "lat":-43.5,  "lon": 171.5,  "ts": "2026-03-15T02:00:00Z"},

    # 農地・草地 (5件) — poc2 FP率 50% (Japan)
    {"desc": "France agricultural",       "lat": 45.0,  "lon":   2.0,  "ts": "2026-03-15T11:00:00Z"},
    {"desc": "Japan rice paddies Aichi",  "lat": 35.0,  "lon": 137.0,  "ts": "2026-03-15T02:00:00Z"},
    {"desc": "Ireland grassland",         "lat": 53.0,  "lon":  -8.0,  "ts": "2026-03-15T12:00:00Z"},
    {"desc": "Ukraine wheat fields",      "lat": 49.0,  "lon":  32.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Argentine Pampas",          "lat":-34.0,  "lon": -62.0,  "ts": "2026-03-15T14:00:00Z"},
]

# ===========================================================================
# FIRMS API
# ===========================================================================

def _firms_key() -> str:
    key = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("FIRMS_API_KEY", "")
    if not key:
        raise RuntimeError("FIRMS_MAP_KEY / FIRMS_API_KEY が設定されていません")
    return key


def fetch_firms(area: str, days: int = 5) -> list[dict]:
    """FIRMS VIIRS_SNPP_NRT から検知イベントを取得する。
    days は 1〜5 のみ有効 (NRT 上限)。"""
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{_firms_key()}/{FIRMS_PRODUCT}/{area}/{days}"
    )
    try:
        r = requests.get(url, timeout=60)
        if not r.ok:
            print(f"    [FIRMS] HTTP {r.status_code}: {r.text[:300]}")
            return []
    except Exception as e:
        print(f"    [FIRMS] 取得失敗: {e}")
        return []

    # csv.DictReader でヘッダー名ベースにパース (poc2_icl.py と同一方式)
    events = []
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            date_str = row["acq_date"]
            time_str = str(row["acq_time"]).zfill(4)
            events.append({
                "lat":  float(row["latitude"]),
                "lon":  float(row["longitude"]),
                "frp":  float(row.get("frp") or 0),
                "date": date_str,            # "YYYY-MM-DD"
                "time": time_str,            # "HHMM"
                "conf": str(row.get("confidence", "n")).strip().lower(),
            })
        except Exception:
            continue
    return events


def _event_to_pos_ts(event: dict) -> str:
    """FIRMS イベント → POS タイムスタンプ (検知日 + SHIFT_DAYS)"""
    dt = datetime.strptime(event["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt += timedelta(days=SHIFT_DAYS)
    return dt.strftime("%Y-%m-%dT12:00:00Z")


def _event_to_neg_ts(event: dict) -> str:
    """FIRMS イベント → NEG タイムスタンプ (検知日 - NEG_OFFSET_DAYS)"""
    dt = datetime.strptime(event["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt -= timedelta(days=NEG_OFFSET_DAYS)
    return dt.strftime("%Y-%m-%dT12:00:00Z")


# ===========================================================================
# GT 構築
# ===========================================================================

def build_ground_truth(indices, is_fire: bool) -> dict:
    """スペクトル指標 + FIRMS ラベル → GT JSON"""
    if is_fire:
        nbr2_score = min(1.0, abs(min(indices.nbr2 + 0.05, 0)) / 0.45)
        swir_score = min(1.0, max(indices.mean_swir22 - 0.15, 0) / 0.35)
        fire_confidence = round(0.60 + 0.35 * (nbr2_score + swir_score) / 2, 3)
        ratio = indices.fire_pixel_ratio
        if   ratio < 0.03: severity = "LOW"
        elif ratio < 0.10: severity = "MEDIUM"
        elif ratio < 0.25: severity = "HIGH"
        else:              severity = "CRITICAL"
    else:
        fire_confidence = round(max(0.05, 0.25 - abs(min(indices.nbr2, 0)) * 0.5), 3)
        severity = "NONE"
        ratio = 0.0

    smoke_detected = is_fire and indices.ndvi < 0.15
    fire_area_ha   = round(ratio * 25_000, 1)  # 5km × 5km = 25km² = 2500ha

    return {
        "smoke_detected":     smoke_detected,
        "smoke_confidence":   round(0.60 if smoke_detected else 0.08, 2),
        "smoke_area_fraction": round(0.15 if smoke_detected else 0.0, 2),
        "fire_detected":      is_fire,
        "fire_confidence":    fire_confidence,
        "fire_area_ha":       fire_area_ha if is_fire else 0.0,
        "fire_front_bbox":    None,
        "spread_direction":   None,
        "severity":           severity,
        "alert_recommended":  is_fire and fire_confidence >= 0.6,
        "description": (
            f"{'Active fire/burn scar detected' if is_fire else 'No fire detected'} "
            f"(NBR2={indices.nbr2:.3f}, SWIR22={indices.mean_swir22:.3f})"
        ),
    }


def build_conversation(gt: dict, indices) -> dict:
    """GT + spectral indices → messages リスト"""
    user_text = FIRE_DETECTION_USER_PROMPT.format(
        nbr2=indices.nbr2,
        ndvi=indices.ndvi,
        bai=indices.bai,
        mean_swir22=indices.mean_swir22,
        fire_pixel_ratio=indices.fire_pixel_ratio,
    )
    return [
        {"role": "system",
         "content": [{"type": "text", "text": FIRE_DETECTION_SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant",
         "content": [{"type": "text", "text": json.dumps(gt, ensure_ascii=False)}]},
    ]


# ===========================================================================
# Dataset Builder
# ===========================================================================

class DatasetBuilder:
    def __init__(self, save_dir: str = "data/build/dataset", max_cloud: float = 50.0):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_cloud = max_cloud
        self.client   = SimSatClient()
        self.proc     = SpectralProcessor()
        self.records: list[dict] = []
        self._load_checkpoint()

    # ------------------------------------------------------------------

    def _ckpt(self) -> Path:
        return self.save_dir / "records.jsonl"

    def _load_checkpoint(self):
        p = self._ckpt()
        if p.exists():
            self.records = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            fire_n    = sum(1 for r in self.records if r["label"])
            nofire_n  = sum(1 for r in self.records if not r["label"])
            print(f"[Builder] チェックポイントから {len(self.records)} 件をロード"
                  f"  (fire={fire_n}, no-fire={nofire_n})")

    def _save_checkpoint(self):
        with open(self._ckpt(), "w") as f:
            for r in self.records:
                rec = {k: v for k, v in r.items() if k != "image"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _save_image(self, img: Image.Image, idx: int) -> str:
        path = self.save_dir / f"img_{idx:04d}.png"
        img.save(path)
        return str(path)

    # ------------------------------------------------------------------

    def _fetch_scene(self, lon: float, lat: float, timestamp: str) -> Optional[dict]:
        """1シーンを取得してサンプル dict を返す。失敗時は None。"""
        try:
            resp = self.client.fetch_fire_scene(
                lon=lon, lat=lat, timestamp=timestamp,
                size_km=SIZE_KM, window_seconds=WINDOW_SEC,
            )
        except Exception as e:
            print(f" → skip (fetch error: {e})")
            return None

        if not resp.image_available or resp.image_array is None:
            cc = f"{resp.cloud_cover:.0f}%" if resp.cloud_cover is not None else "?"
            print(f" → skip (no image, cc={cc})")
            return None
        if resp.cloud_cover > self.max_cloud:
            print(f" → skip (cloud={resp.cloud_cover:.0f}%)")
            return None

        try:
            scene = self.proc.process(resp)
        except Exception as e:
            print(f" → skip (spectral error: {e})")
            return None

        return {
            "scene":   scene,
            "capture": resp.datetime,
        }

    def _already_fetched(self, lon: float, lat: float, ts_date: str) -> bool:
        return any(
            abs(r["lon"] - lon) < 0.01
            and abs(r["lat"] - lat) < 0.01
            and r.get("ts_date") == ts_date
            for r in self.records
        )

    def _add_record(self, lon: float, lat: float, ts_date: str,
                    scene, is_fire: bool, source: str, desc: str):
        gt   = build_ground_truth(scene.indices, is_fire)
        conv = build_conversation(gt, scene.indices)
        idx  = len(self.records)
        img_path = self._save_image(scene.fire_composite, idx)
        self.records.append({
            "label":           is_fire,
            "source":          source,   # "firms_pos" | "firms_neg" | "diverse_neg"
            "desc":            desc,
            "lon":             lon,
            "lat":             lat,
            "ts_date":         ts_date,
            "capture":         scene.capture_datetime,
            "cloud_cover":     scene.cloud_cover,
            "nbr2":            float(scene.indices.nbr2),
            "mean_swir22":     float(scene.indices.mean_swir22),
            "fire_pixel_ratio": float(scene.indices.fire_pixel_ratio),
            "image_path":      img_path,
            "messages_json":   json.dumps(conv, ensure_ascii=False),
        })
        self._save_checkpoint()

    # ------------------------------------------------------------------

    def collect_firms(self, n_pos: int = 100, firms_days: int = 30) -> None:
        """
        FIRMS イベント → POS + NEG temporal ペアを収集。

        Δ≥0 フィルター: POS タイムスタンプが FIRMS 検知日以降のものだけ採用。
        """
        existing_pos = sum(1 for r in self.records if r["source"] == "firms_pos")
        existing_neg = sum(1 for r in self.records if r["source"] == "firms_neg")
        need_pos = n_pos - existing_pos
        if need_pos <= 0:
            print(f"[Builder] FIRMS POS 既に {existing_pos} 件 → スキップ")
            return

        print(f"\n[Builder] FIRMS イベント取得 (過去{firms_days}日, 目標POS={n_pos}件) ...")
        events: list[dict] = []
        for name, area in FIRMS_AREAS.items():
            evs = fetch_firms(area, firms_days)
            print(f"    {name}: {len(evs)} イベント")
            events.extend(evs)

        if not events:
            print("[Builder] FIRMS データなし")
            return

        # FRP 降順にソート (高FRPほど burn scar が明瞭)
        events.sort(key=lambda e: e["frp"], reverse=True)

        # バッファ: 3倍を目安に候補を確保
        buffer = min(len(events), need_pos * 4)
        events = events[:buffer]

        pos_collected = 0
        neg_collected = 0

        for ev in events:
            if pos_collected >= need_pos:
                break

            lon, lat = ev["lon"], ev["lat"]
            pos_ts = _event_to_pos_ts(ev)
            neg_ts = _event_to_neg_ts(ev)
            ts_date_pos = pos_ts[:10]
            ts_date_neg = neg_ts[:10]

            # --- POS ---
            if not self._already_fetched(lon, lat, ts_date_pos):
                desc_pos = f"FIRMS FRP={ev['frp']:.0f}MW ({lon:.2f},{lat:.2f}) {ev['date']}"
                print(f"  POS {desc_pos} ...", end=" ", flush=True)
                t0 = time.perf_counter()
                result = self._fetch_scene(lon, lat, pos_ts)
                if result is None:
                    continue

                # Δ≥0 フィルター: capture が FIRMS 検知日以降か確認
                cap_date = result["capture"][:10]
                if cap_date < ev["date"]:
                    print(f" → skip (Δ<0: capture={cap_date} < firms={ev['date']})")
                    continue

                self._add_record(lon, lat, ts_date_pos,
                                 result["scene"], is_fire=True,
                                 source="firms_pos", desc=desc_pos)
                print(f"✅ NBR2={result['scene'].indices.nbr2:.3f} ({time.perf_counter()-t0:.1f}s)")
                pos_collected += 1

            # --- NEG temporal (同座標・-180日) ---
            if not self._already_fetched(lon, lat, ts_date_neg):
                desc_neg = f"NEG temporal ({lon:.2f},{lat:.2f}) {neg_ts[:10]}"
                print(f"  NEG {desc_neg} ...", end=" ", flush=True)
                t0 = time.perf_counter()
                result = self._fetch_scene(lon, lat, neg_ts)
                if result is None:
                    continue

                self._add_record(lon, lat, ts_date_neg,
                                 result["scene"], is_fire=False,
                                 source="firms_neg", desc=desc_neg)
                print(f"✅ NBR2={result['scene'].indices.nbr2:.3f} ({time.perf_counter()-t0:.1f}s)")
                neg_collected += 1

        print(f"\n[Builder] FIRMS 収集完了: POS+={pos_collected}, NEG+={neg_collected}")

    # ------------------------------------------------------------------

    def collect_diverse_neg(self, locations: list[dict] | None = None) -> None:
        """FP-prone バイオームの非火災地点を収集。"""
        if locations is None:
            locations = DIVERSE_NEG_LOCATIONS

        existing = sum(1 for r in self.records if r["source"] == "diverse_neg")
        print(f"\n[Builder] diverse NEG 収集 (既存={existing}, 目標={len(locations)}) ...")

        collected = 0
        for loc in locations:
            ts_date = loc["ts"][:10]
            if self._already_fetched(loc["lon"], loc["lat"], ts_date):
                continue

            print(f"  [{loc['desc']}] ...", end=" ", flush=True)
            t0 = time.perf_counter()
            result = self._fetch_scene(loc["lon"], loc["lat"], loc["ts"])
            if result is None:
                continue

            self._add_record(loc["lon"], loc["lat"], ts_date,
                             result["scene"], is_fire=False,
                             source="diverse_neg", desc=loc["desc"])
            print(f"✅ NBR2={result['scene'].indices.nbr2:.3f} ({time.perf_counter()-t0:.1f}s)")
            collected += 1

        print(f"[Builder] diverse NEG 収集完了: +{collected} 件")

    # ------------------------------------------------------------------

    def finalize(self, val_ratio: float = 0.15, test_ratio: float = 0.15) -> None:
        """
        Train / Val / Test 分割して HuggingFace Dataset 形式で保存。

        分割戦略:
          - stratified by label (fire/no-fire)
          - diverse_neg は val/test に均等配分されるよう試みる
          - test は学習中に一切触れない held-out set
        """
        from datasets import Dataset

        if not self.records:
            print("[Builder] レコードがありません")
            return

        n = len(self.records)
        n_val  = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test

        print(f"\n[Builder] 分割: train={n_train}, val={n_val}, test={n_test} (total={n})")
        print(f"  fire={sum(r['label'] for r in self.records)}, "
              f"no-fire={sum(not r['label'] for r in self.records)}")

        # --- Stratified split ---
        import random
        random.seed(42)
        fire_recs   = [r for r in self.records if r["label"]]
        nofire_recs = [r for r in self.records if not r["label"]]

        def split_class(recs, val_r, test_r):
            recs = recs[:]
            random.shuffle(recs)
            n_v = max(1, round(len(recs) * val_r))
            n_t = max(1, round(len(recs) * test_r))
            return recs[n_v + n_t:], recs[:n_v], recs[n_v:n_v + n_t]

        fire_train,   fire_val,   fire_test   = split_class(fire_recs,   val_ratio, test_ratio)
        nofire_train, nofire_val, nofire_test = split_class(nofire_recs, val_ratio, test_ratio)

        splits = {
            "train": fire_train + nofire_train,
            "val":   fire_val   + nofire_val,
            "test":  fire_test  + nofire_test,
        }

        hf_dir = self.save_dir.parent / "hf_dataset"
        hf_dir.mkdir(parents=True, exist_ok=True)

        for split_name, recs in splits.items():
            random.shuffle(recs)
            imgs, texts, labels, sources = [], [], [], []
            for r in recs:
                imgs.append(Image.open(r["image_path"]).convert("RGB"))
                texts.append(r["messages_json"])
                labels.append(int(r["label"]))
                sources.append(r.get("source", "unknown"))

            ds = Dataset.from_dict({
                "image":        imgs,
                "messages_json": texts,
                "label":        labels,
                "source":       sources,
                "nbr2":         [r["nbr2"] for r in recs],
                "mean_swir22":  [r["mean_swir22"] for r in recs],
            })
            split_dir = hf_dir / split_name
            ds.save_to_disk(str(split_dir))

            fire_n   = sum(labels)
            nofire_n = len(labels) - fire_n
            diverse_n = sum(1 for s in sources if s == "diverse_neg")
            print(f"  [{split_name}] {len(recs)} 件"
                  f" (fire={fire_n}, no-fire={nofire_n}, diverse_neg={diverse_n})")

        print(f"\n[Builder] HF Dataset 保存完了: {hf_dir}")

    # ------------------------------------------------------------------

    def report(self) -> None:
        """データセット統計を表示。"""
        total = len(self.records)
        if total == 0:
            print("レコードなし")
            return
        fire_n   = sum(r["label"] for r in self.records)
        nofire_n = total - fire_n
        sources  = {}
        for r in self.records:
            s = r.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1

        print(f"\n{'='*50}")
        print(f"Dataset Report  (total={total})")
        print(f"  fire={fire_n}  no-fire={nofire_n}")
        for s, n in sorted(sources.items()):
            print(f"  {s}: {n}")
        if fire_n > 0 and nofire_n > 0:
            pos_nbr2 = [r["nbr2"] for r in self.records if r["label"]]
            neg_nbr2 = [r["nbr2"] for r in self.records if not r["label"]]
            print(f"  POS NBR2: mean={np.mean(pos_nbr2):.3f}  "
                  f"min={min(pos_nbr2):.3f}  max={max(pos_nbr2):.3f}")
            print(f"  NEG NBR2: mean={np.mean(neg_nbr2):.3f}  "
                  f"min={min(neg_nbr2):.3f}  max={max(neg_nbr2):.3f}")
        print(f"{'='*50}\n")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    # .env ロード
    _env = ROOT / ".." / ".." / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    p = argparse.ArgumentParser()
    p.add_argument("--n-pos",      type=int, default=100,
                   help="FIRMS POS 件数 (NEG temporal は同数収集)")
    p.add_argument("--firms-days", type=int, default=5,
                   help="FIRMS API 取得日数 (NRT 上限=5日)")
    p.add_argument("--save-dir",   default="data/build/dataset")
    p.add_argument("--no-diverse", action="store_true",
                   help="diverse NEG をスキップ (デバッグ用)")
    args = p.parse_args()

    builder = DatasetBuilder(save_dir=args.save_dir)
    builder.collect_firms(n_pos=args.n_pos, firms_days=args.firms_days)
    if not args.no_diverse:
        builder.collect_diverse_neg()
    builder.report()
    builder.finalize()
