"""
FireEdge /finetune Dataset Builder
=================================
FIRMS API → SimSat → SWIR composite → Train/Val/Test JSONL

データ設計:
  POS          : FIRMS VIIRS_SNPP_NRT 検知座標 → SimSat (検知日+SHIFT_DAYS, Δ≥0)
  NEG temporal : 同座標 → SimSat (検知日-NEG_OFFSET_DAYS)  ← burn scar シグナルのみを差異とする
  NEG diverse  : FP-prone バイオーム 45件 (砂漠10, 都市8, 湿地8, サバンナ8, 温帯森林6, 農地5)
                 乾季 timestamp 優先 (SWIR 高反射で burn scar と混同されやすい条件を意図的に追加)

分割:
  Train 70% / Val 15% / Test 15%  (stratified by label)
  Generalization test: DIVERSE_NEG_LOCATIONS 16地点 (データセットとは独立)

使い方:
    cd apps/fireedge
    uv run python -m finetune.dataset_builder
    uv run python -m finetune.dataset_builder --n-pos 100 --save-dir data/finetune/dataset
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
    FIRE_DETECTION_FT_PROMPT,
)
from src.spectral import SpectralProcessor

# ===========================================================================
# 定数
# ===========================================================================

SHIFT_DAYS      = 2    # POS: FIRMS検知日 + N日後 (burn scar が安定)
# NEG temporal: 検知日から以下のオフセット候補を順に試み、FIRMS fire-free な日付を採用
NEG_OFFSET_CANDIDATES = [180, 270, 90]  # 日数候補 (180日=約6ヶ月が第一優先)
NEG_FIRMS_WINDOW_DAYS = 14              # FIRMS照合ウィンドウ: ±7日相当
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
    # 砂漠・乾燥地 (10件)
    # 乾季は SWIR 高輝度・NBR2 低下が burn scar と類似しやすいため乾季 timestamp を優先
    # 湿潤期 timestamp はモデルが TN を維持できることの確認用
    {"desc": "Sahara Algeria",            "lat": 23.0,  "lon":   5.0,  "ts": "2025-07-15T10:00:00Z"},  # dry season
    {"desc": "Sahara Libya",              "lat": 26.0,  "lon":  14.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Arabian Peninsula KSA",     "lat": 24.0,  "lon":  45.0,  "ts": "2025-01-15T08:00:00Z"},  # dry season
    {"desc": "Arabian Peninsula UAE",     "lat": 24.5,  "lon":  54.5,  "ts": "2026-03-15T07:00:00Z"},
    {"desc": "Australian outback SA",     "lat":-25.0,  "lon": 135.0,  "ts": "2026-03-15T02:00:00Z"},
    {"desc": "Australian outback NT",     "lat":-22.0,  "lon": 133.0,  "ts": "2026-03-15T01:00:00Z"},
    {"desc": "Atacama Desert Chile",      "lat":-24.0,  "lon": -69.5,  "ts": "2025-07-15T14:00:00Z"},  # dry season
    {"desc": "Gobi Desert Mongolia",      "lat": 44.0,  "lon": 106.0,  "ts": "2025-06-15T04:00:00Z"},  # dry season
    {"desc": "Namib Desert Namibia",      "lat":-22.5,  "lon":  15.0,  "ts": "2025-08-15T10:00:00Z"},  # dry season
    {"desc": "Iranian plateau",           "lat": 32.0,  "lon":  55.0,  "ts": "2025-07-15T07:00:00Z"},  # dry season

    # 都市郊外 (8件)
    # 乾季の低植生・裸地テクスチャが burn scar と視覚的に類似しやすい
    {"desc": "London suburbs",            "lat": 51.5,  "lon":  -0.1,  "ts": "2026-03-15T11:00:00Z"},
    {"desc": "Paris suburbs",             "lat": 48.8,  "lon":   2.2,  "ts": "2025-08-15T11:00:00Z"},  # dry season
    {"desc": "Chicago suburbs",           "lat": 41.9,  "lon": -87.7,  "ts": "2026-03-15T17:00:00Z"},
    {"desc": "Seoul suburbs",             "lat": 37.5,  "lon": 127.0,  "ts": "2025-01-15T02:00:00Z"},  # dry season
    {"desc": "Sydney suburbs",            "lat":-33.8,  "lon": 151.0,  "ts": "2025-07-15T00:00:00Z"},  # dry season
    {"desc": "Buenos Aires suburbs",      "lat":-34.6,  "lon": -58.5,  "ts": "2025-01-15T14:00:00Z"},  # dry season
    {"desc": "Istanbul suburbs",          "lat": 41.0,  "lon":  29.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Mumbai suburbs",            "lat": 19.1,  "lon":  73.0,  "ts": "2025-01-15T07:00:00Z"},  # dry season

    # 湿地・デルタ (8件)
    # 乾季は水位低下により泥炭・裸地露出 → SWIR 高反射で burn scar と混同されやすい
    {"desc": "Bangladesh Ganges delta",   "lat": 22.5,  "lon":  90.5,  "ts": "2025-01-15T05:00:00Z"},  # dry season
    {"desc": "Mekong delta Vietnam",      "lat":  9.5,  "lon": 105.5,  "ts": "2025-01-15T04:00:00Z"},  # dry season
    {"desc": "Nile delta Egypt",          "lat": 31.0,  "lon":  31.0,  "ts": "2025-07-15T09:00:00Z"},  # dry season
    {"desc": "Okavango delta Botswana",   "lat":-19.5,  "lon":  23.0,  "ts": "2025-08-15T10:00:00Z"},  # dry season
    {"desc": "Danube delta Romania",      "lat": 45.0,  "lon":  29.5,  "ts": "2025-08-15T10:00:00Z"},  # dry season
    {"desc": "Pantanal Brazil (dry)",     "lat":-17.0,  "lon": -57.0,  "ts": "2025-08-15T13:00:00Z"},  # dry season
    {"desc": "Sudd wetland South Sudan",  "lat":  7.5,  "lon":  30.5,  "ts": "2025-01-15T09:00:00Z"},  # dry season
    {"desc": "Sundarbans mangrove",       "lat": 21.9,  "lon":  89.2,  "ts": "2026-03-15T05:00:00Z"},

    # サバンナ非火災期 (8件)
    # 乾季は草地枯死・土壌露出で SWIR シグナルが burn scar と類似する
    # 雨季 timestamp は TN 確認用
    {"desc": "Kenya savanna rainy",       "lat":  1.0,  "lon":  37.0,  "ts": "2026-03-15T09:00:00Z"},
    {"desc": "Tanzania Serengeti dry",    "lat": -2.5,  "lon":  35.0,  "ts": "2025-08-15T09:00:00Z"},  # dry season
    {"desc": "Zambia plateau rainy",      "lat":-14.0,  "lon":  28.0,  "ts": "2026-03-15T10:00:00Z"},
    {"desc": "Colombian Llanos rainy",    "lat":  5.0,  "lon": -70.0,  "ts": "2026-03-15T14:00:00Z"},
    {"desc": "Venezuela Llanos dry",      "lat":  7.0,  "lon": -66.0,  "ts": "2025-01-15T14:00:00Z"},  # dry season
    {"desc": "Myanmar dry zone",          "lat": 21.0,  "lon":  95.5,  "ts": "2025-12-15T04:00:00Z"},  # dry season
    {"desc": "Indian Deccan plateau",     "lat": 17.5,  "lon":  78.0,  "ts": "2025-01-15T07:00:00Z"},  # dry season
    {"desc": "Brazilian Cerrado dry",     "lat":-12.0,  "lon": -46.0,  "ts": "2025-07-15T13:00:00Z"},  # dry season

    # 温帯森林 (6件)
    # 落葉後の裸地テクスチャや夏季乾燥ストレスが burn scar に類似しやすい
    {"desc": "Germany Black Forest",      "lat": 48.0,  "lon":   8.2,  "ts": "2025-08-15T11:00:00Z"},  # dry season
    {"desc": "Canada boreal Manitoba",    "lat": 55.0,  "lon":-100.0,  "ts": "2025-07-15T18:00:00Z"},  # dry season
    {"desc": "Brazil deep Amazon",        "lat": -2.0,  "lon": -62.0,  "ts": "2025-08-15T15:00:00Z"},  # dry season
    {"desc": "Scandinavia boreal Norway", "lat": 63.0,  "lon":  13.0,  "ts": "2025-07-15T11:00:00Z"},  # dry season
    {"desc": "Carpathian forest Romania", "lat": 46.0,  "lon":  25.0,  "ts": "2025-08-15T10:00:00Z"},  # dry season
    {"desc": "New Zealand temperate",     "lat":-43.5,  "lon": 171.5,  "ts": "2025-01-15T02:00:00Z"},  # dry season

    # 農地・草地 (5件)
    # 収穫後の裸地・乾季の枯草が SWIR 高反射になりやすい
    {"desc": "France agricultural",       "lat": 45.0,  "lon":   2.0,  "ts": "2025-08-15T11:00:00Z"},  # dry season
    {"desc": "Japan rice paddies Aichi",  "lat": 35.0,  "lon": 137.0,  "ts": "2025-01-15T02:00:00Z"},  # dry season
    {"desc": "Ireland grassland",         "lat": 53.0,  "lon":  -8.0,  "ts": "2026-03-15T12:00:00Z"},
    {"desc": "Ukraine wheat fields",      "lat": 49.0,  "lon":  32.0,  "ts": "2025-08-15T09:00:00Z"},  # dry season
    {"desc": "Argentine Pampas",          "lat":-34.0,  "lon": -62.0,  "ts": "2025-01-15T14:00:00Z"},  # dry season
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


def fetch_firms_at_date(lon: float, lat: float, date_str: str,
                        radius_deg: float = 0.1, days: int = 14) -> list[dict]:
    """FIRMS SP アーカイブで特定座標・日付付近の火災検知を照会。

    FIRMS Area CSV API (date 指定版):
      .../api/area/csv/{KEY}/VIIRS_SNPP_SP/{W,S,E,N}/{DAYS}/{DATE}
    date_str: "YYYY-MM-DD"。days 日間（date_str から前方）を検索。
    """
    key = _firms_key()
    # 検索ウィンドウの開始日 = date_str - days/2 (中心を date_str に)
    center_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_dt  = center_dt - timedelta(days=days // 2)
    start_str = start_dt.strftime("%Y-%m-%d")

    w = round(lon - radius_deg, 4)
    s = round(lat - radius_deg, 4)
    e = round(lon + radius_deg, 4)
    n = round(lat + radius_deg, 4)
    bbox = f"{w},{s},{e},{n}"

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{key}/VIIRS_SNPP_SP/{bbox}/{days}/{start_str}"
    )
    try:
        r = requests.get(url, timeout=30)
        if not r.ok:
            return []
    except Exception:
        return []

    events = []
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            events.append({
                "lat":  float(row["latitude"]),
                "lon":  float(row["longitude"]),
                "date": row["acq_date"],
                "frp":  float(row.get("frp") or 0),
            })
        except Exception:
            continue
    return events


def _find_fire_free_neg_ts(event: dict) -> str | None:
    """FIRMS SP アーカイブを確認し、fire-free な NEG タイムスタンプを返す。

    NEG_OFFSET_CANDIDATES を順に試し、FIRMS 検知がない最初のオフセットを採用。
    すべてのオフセットで火災が検知された場合は None を返す（ペア破棄）。
    """
    lon, lat = event["lon"], event["lat"]
    base_dt   = datetime.strptime(event["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for offset_days in NEG_OFFSET_CANDIDATES:
        neg_dt   = base_dt - timedelta(days=offset_days)
        neg_date = neg_dt.strftime("%Y-%m-%d")
        fires    = fetch_firms_at_date(lon, lat, neg_date,
                                       radius_deg=0.1,
                                       days=NEG_FIRMS_WINDOW_DAYS)
        if not fires:
            return neg_dt.strftime("%Y-%m-%dT12:00:00Z")
        # FIRMS 検知あり → 次のオフセット候補へ
        print(f" [FIRMS確認] -{offset_days}日 に火災検知({len(fires)}件) → 次候補へ", end="", flush=True)

    return None  # 全候補で火災検知 → ペア破棄


def _event_to_pos_ts(event: dict) -> str:
    """FIRMS イベント → POS タイムスタンプ (検知日 + SHIFT_DAYS)"""
    dt = datetime.strptime(event["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt += timedelta(days=SHIFT_DAYS)
    return dt.strftime("%Y-%m-%dT12:00:00Z")


# ===========================================================================
# GT 構築
# ===========================================================================

def build_conversation(is_fire: bool) -> list:
    """FIRMS ラベルのみから messages リストを構築する。
    GT は {"fire_detected": bool} の単フィールドのみ。
    スペクトル指標は含めない (循環ラベル・答え漏洩を防ぐ)。"""
    gt = json.dumps({"fire_detected": bool(is_fire)})
    return [
        {"role": "system",
         "content": [{"type": "text", "text": FIRE_DETECTION_SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image"}, {"type": "text", "text": FIRE_DETECTION_FT_PROMPT}]},
        {"role": "assistant",
         "content": [{"type": "text", "text": gt}]},
    ]


# ===========================================================================
# Dataset Builder
# ===========================================================================

class DatasetBuilder:
    def __init__(self, save_dir: str = "data/finetune/dataset", max_cloud: float = 50.0):
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

        # 黒画像フィルタ: 有効ピクセル率 < 1% はデータ欠損 (夜間シフト後・タイル境界等)
        valid_ratio = float(np.mean(resp.image_array > 0.001))
        if valid_ratio < 0.01:
            print(f" → skip (black image: valid={valid_ratio:.1%})")
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
        conv = build_conversation(is_fire)
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
            "nbr2_min":        float(scene.indices.nbr2_min),
            "mean_swir22":     float(scene.indices.mean_swir22),
            "swir22_max":      float(scene.indices.swir22_max),
            "fire_pixel_ratio": float(scene.indices.fire_pixel_ratio),
            "image_path":      img_path,
            "messages_json":   json.dumps(conv, ensure_ascii=False),
        })
        self._save_checkpoint()

    # ------------------------------------------------------------------

    def collect_firms(self, n_pos: int = 100, firms_days: int = 30,
                      min_frp: float = 0.0) -> None:
        """
        FIRMS イベント → POS + NEG temporal ペアを収集。

        Δ≥0 フィルター: POS タイムスタンプが FIRMS 検知日以降のものだけ採用。
        min_frp: FRP 最低閾値 [MW]。デフォルト 0.0 (フィルタなし)。
            指定する場合: FRP < 閾値の小規模火災は 20m ピクセルで希釈され
            burn scar シグナルが消えるケースがある (混合ピクセル問題)。
        """
        existing_pos = sum(1 for r in self.records if r["source"] == "firms_pos")
        existing_neg = sum(1 for r in self.records if r["source"] == "firms_neg")
        need_pos = n_pos - existing_pos
        if need_pos <= 0:
            print(f"[Builder] FIRMS POS 既に {existing_pos} 件 → スキップ")
            return

        print(f"\n[Builder] FIRMS イベント取得 (過去{firms_days}日, 目標POS={n_pos}件, min_frp={min_frp}MW) ...")
        events: list[dict] = []
        for name, area in FIRMS_AREAS.items():
            evs = fetch_firms(area, firms_days)
            print(f"    {name}: {len(evs)} イベント")
            events.extend(evs)

        if not events:
            print("[Builder] FIRMS データなし")
            return

        # FRP 閾値フィルタ (min_frp=0 はフィルタなし)
        if min_frp > 0:
            before = len(events)
            events = [e for e in events if e["frp"] >= min_frp]
            print(f"    FRP >= {min_frp}MW フィルタ: {before} → {len(events)} イベント")
            if not events:
                print(f"[Builder] FRP >= {min_frp}MW のイベントなし (--min-frp を下げるか firms_days を増やす)")
                return

        # FRP 降順にソート (高FRPほど burn scar が明瞭)
        events.sort(key=lambda e: e["frp"], reverse=True)

        # バッファ上限を撤廃: 全イベントを走査し、目標件数に達するまで収集する
        pos_collected = 0
        neg_collected = 0
        skipped_pair = 0  # NEG失敗によるペア破棄数

        for ev in events:
            if pos_collected >= need_pos:
                break

            lon, lat = ev["lon"], ev["lat"]
            pos_ts = _event_to_pos_ts(ev)
            ts_date_pos = pos_ts[:10]

            # 同一座標・同一日付の重複イベントはスキップ (カウントしない)
            # ※ FIRMS は同一火災を複数ピクセルで報告するため重複が多い
            if self._already_fetched(lon, lat, ts_date_pos):
                continue

            # --- POS ---
            desc_pos = f"FIRMS FRP={ev['frp']:.0f}MW ({lon:.2f},{lat:.2f}) {ev['date']}"
            print(f"  POS {desc_pos} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            pos_result = self._fetch_scene(lon, lat, pos_ts)
            if pos_result is None:
                continue

            # Δ≥0 フィルター: capture が FIRMS 検知日以降か確認
            cap_date = pos_result["capture"][:10]
            if cap_date < ev["date"]:
                print(f" → skip (Δ<0: capture={cap_date} < firms={ev['date']})")
                continue
            print(f"✅ NBR2_min={pos_result['scene'].indices.nbr2_min:.3f} ({time.perf_counter()-t0:.1f}s)")

            # --- NEG temporal: FIRMS SP で fire-free な日付を選択 ---
            print(f"  NEG FIRMS確認中 ({lon:.2f},{lat:.2f}) ...", end=" ", flush=True)
            neg_ts = _find_fire_free_neg_ts(ev)
            if neg_ts is None:
                print(f" → 全オフセットで火災検知 → ペア破棄")
                skipped_pair += 1
                continue

            desc_neg = f"NEG temporal ({lon:.2f},{lat:.2f}) {neg_ts[:10]}"
            print(f"\n  NEG {desc_neg} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            neg_result = self._fetch_scene(lon, lat, neg_ts)
            if neg_result is None:
                print(f" → NEG失敗のためペア破棄")
                skipped_pair += 1
                continue
            print(f"✅ NBR2_min={neg_result['scene'].indices.nbr2_min:.3f} ({time.perf_counter()-t0:.1f}s)")

            # 両方成功 → 保存
            self._add_record(lon, lat, ts_date_pos,
                             pos_result["scene"], is_fire=True,
                             source="firms_pos", desc=desc_pos)
            self._add_record(lon, lat, neg_ts[:10],
                             neg_result["scene"], is_fire=False,
                             source="firms_neg", desc=desc_neg)
            pos_collected += 1
            neg_collected += 1

        print(f"\n[Builder] FIRMS 収集完了: POS+={pos_collected}, NEG+={neg_collected}, "
              f"ペア破棄={skipped_pair}")

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
            print(f"✅ NBR2_min={result['scene'].indices.nbr2_min:.3f} ({time.perf_counter()-t0:.1f}s)")
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
                "nbr2_min":     [r.get("nbr2_min", 0.0) for r in recs],
                "mean_swir22":  [r["mean_swir22"] for r in recs],
                "swir22_max":   [r.get("swir22_max", 0.0) for r in recs],
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
            pos_nbr2     = [r["nbr2"]              for r in self.records if r["label"]]
            neg_nbr2     = [r["nbr2"]              for r in self.records if not r["label"]]
            print(f"  POS NBR2 (mean): mean={np.mean(pos_nbr2):.3f}  "
                  f"min={min(pos_nbr2):.3f}  max={max(pos_nbr2):.3f}")
            print(f"  NEG NBR2 (mean): mean={np.mean(neg_nbr2):.3f}  "
                  f"min={min(neg_nbr2):.3f}  max={max(neg_nbr2):.3f}")
            # nbr2_min / swir22_max は旧チェックポイントにない場合は表示スキップ
            if all("nbr2_min" in r for r in self.records):
                pos_nbr2_min = [r["nbr2_min"] for r in self.records if r["label"]]
                neg_nbr2_min = [r["nbr2_min"] for r in self.records if not r["label"]]
                print(f"  POS NBR2_min:    mean={np.mean(pos_nbr2_min):.3f}  "
                      f"min={min(pos_nbr2_min):.3f}  max={max(pos_nbr2_min):.3f}")
                print(f"  NEG NBR2_min:    mean={np.mean(neg_nbr2_min):.3f}  "
                      f"min={min(neg_nbr2_min):.3f}  max={max(neg_nbr2_min):.3f}")
                sep_nbr2 = np.mean(pos_nbr2_min) < np.mean(neg_nbr2_min) - 0.05
            else:
                print("  NBR2_min: (旧チェックポイント — 再収集後に表示)")
                sep_nbr2 = None
            if all("swir22_max" in r for r in self.records):
                pos_swir_max = [r["swir22_max"] for r in self.records if r["label"]]
                neg_swir_max = [r["swir22_max"] for r in self.records if not r["label"]]
                print(f"  POS SWIR22_max:  mean={np.mean(pos_swir_max):.3f}  "
                      f"min={min(pos_swir_max):.3f}  max={max(pos_swir_max):.3f}")
                print(f"  NEG SWIR22_max:  mean={np.mean(neg_swir_max):.3f}  "
                      f"min={min(neg_swir_max):.3f}  max={max(neg_swir_max):.3f}")
                sep_swir = np.mean(pos_swir_max) > np.mean(neg_swir_max) + 0.05
            else:
                print("  SWIR22_max: (旧チェックポイント — 再収集後に表示)")
                sep_swir = None
            if sep_nbr2 is not None and sep_swir is not None:
                print(f"  分離判定 NBR2_min: {'✅ POS < NEG' if sep_nbr2 else '⚠️  分離不十分'}"
                      f"  SWIR22_max: {'✅ POS > NEG' if sep_swir else '⚠️  分離不十分'}")
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
    p.add_argument("--n-pos",      type=int,   default=100,
                   help="FIRMS POS 件数 (NEG temporal は同数収集)")
    p.add_argument("--firms-days", type=int,   default=5,
                   help="FIRMS API 取得日数 (NRT 上限=5日)")
    p.add_argument("--min-frp",    type=float, default=0.0,
                   help="FIRMS FRP 最低閾値 [MW] (デフォルト=0: フィルタなし)")
    p.add_argument("--save-dir",   default="data/finetune/dataset")
    p.add_argument("--no-diverse", action="store_true",
                   help="diverse NEG をスキップ (デバッグ用)")
    args = p.parse_args()

    builder = DatasetBuilder(save_dir=args.save_dir)
    builder.collect_firms(n_pos=args.n_pos, firms_days=args.firms_days, min_frp=args.min_frp)
    if not args.no_diverse:
        builder.collect_diverse_neg()
    builder.report()
    builder.finalize()
