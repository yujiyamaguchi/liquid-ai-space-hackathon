"""
FireEdge Dataset Builder
=========================
SimSat API から Sentinel-2 SWIR シーンを取得し、
スペクトル指標をグラウンドトゥルースとして使って
VLM 学習用 JSONL データセットを構築する。

使い方:
    uv run python -m finetune.dataset_builder
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from interfaces import (
    FIRE_DETECTION_SYSTEM_PROMPT,
    FIRE_DETECTION_USER_PROMPT,
)
from src.data_fetcher import SimSatClient
from src.spectral import SpectralProcessor

# ------------------------------------------------------------------ constants

# 1月に火災が多い座標 (南半球夏季・アフリカ・南米)
FIRE_PRONE_LOCATIONS: list[tuple[float, float]] = [
    # Australia QLD / NSW (夏季ブッシュファイア)
    (142.0, -30.0), (143.5, -29.5), (141.0, -31.0),
    (149.0, -33.0), (148.0, -32.5), (150.3, -33.7),
    # Australia VIC / SA
    (147.2, -36.8), (146.0, -37.0), (138.6, -31.5),
    # West Australia (北部 savanna fire season)
    (117.8, -22.5), (121.0, -21.0),
    # South Africa / Botswana savanna
    (19.0, -24.0), (20.0, -23.0), (22.0, -22.0),
    # Mozambique
    (35.0, -16.0), (34.5, -15.5),
    # Brazil Cerrado / Amazon
    (-47.0, -13.0), (-48.0, -14.0), (-52.0, -5.5),
    # Bolivia (Chiquitanía)
    (-62.0, -16.5), (-61.0, -15.0),
    # Chile / Argentina
    (-71.6, -33.0), (-71.0, -39.5),
    # California (LA fire reference)
    (-118.1, 34.4), (-119.0, 37.0),
    # Indonesia Kalimantan (peat fire)
    (113.5, -1.5), (115.0, -2.0),
]

# 火災が起きにくい座標 (海洋・極地・冬季・都市)
NO_FIRE_LOCATIONS: list[tuple[float, float]] = [
    # Pacific Ocean
    (-150.0, 5.0), (160.0, 5.0), (-170.0, 20.0),
    # Atlantic Ocean
    (-30.0, 20.0), (-40.0, 10.0),
    # Indian Ocean
    (75.0, -10.0), (80.0, -15.0),
    # Greenland ice sheet
    (-42.0, 72.0), (-45.0, 70.0),
    # Sahara desert (hot but no fire)
    (10.0, 23.0), (25.0, 25.0), (30.0, 27.0),
    # Siberia / Arctic (Jan = frozen)
    (100.0, 62.0), (70.0, 68.0), (-95.0, 70.0),
    # Alps / Europe (winter)
    (8.0, 46.5), (2.3, 48.9), (13.0, 52.0),
    # Urban Asia (very low fire risk)
    (139.7, 35.7), (103.8, 1.3),
    # Amazon intact forest (no dry season in Jan)
    (-60.0, -3.0),
    # Congo Basin
    (24.5, -1.5),
    # Antarctic Peninsula
    (-62.0, -64.0),
]

TIMESTAMPS = [
    "2026-01-05T04:00:00Z",
    "2026-01-10T04:00:00Z",
    "2026-01-15T05:00:00Z",
    "2026-01-20T04:00:00Z",
]


# ------------------------------------------------------------------ GT logic

def build_ground_truth(indices) -> dict:
    """スペクトル指標 → グラウンドトゥルース JSON を生成。"""
    fire_detected = (indices.nbr2 < -0.05) and (indices.mean_swir22 > 0.15)

    if fire_detected:
        nbr2_score = min(1.0, abs(indices.nbr2 + 0.05) / 0.45)
        swir_score = min(1.0, (indices.mean_swir22 - 0.15) / 0.35)
        fire_confidence = round(0.55 + 0.40 * (nbr2_score + swir_score) / 2, 3)
    else:
        fire_confidence = round(max(0.05, 0.35 - abs(min(indices.nbr2, 0)) * 0.8), 3)

    ratio = indices.fire_pixel_ratio
    if not fire_detected:
        severity = "NONE"
    elif ratio < 0.03:
        severity = "LOW"
    elif ratio < 0.10:
        severity = "MEDIUM"
    elif ratio < 0.25:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    smoke_detected = fire_detected and indices.ndvi < 0.15
    fire_area_ha = round(ratio * 400_000, 1)   # 20km × 20km = 400 km² = 40,000 ha

    return {
        "smoke_detected": smoke_detected,
        "smoke_confidence": round(0.55 if smoke_detected else 0.10, 2),
        "fire_detected": fire_detected,
        "fire_confidence": fire_confidence,
        "fire_area_ha": fire_area_ha,
        "fire_front_bbox": None,
        "spread_direction": None,
        "severity": severity,
        "alert_recommended": fire_detected and fire_confidence >= 0.6,
        "description": (
            f"{'Active fire detected' if fire_detected else 'No fire detected'} "
            f"(NBR2={indices.nbr2:.3f}, SWIR2.2={indices.mean_swir22:.3f}, "
            f"fire_px={ratio*100:.2f}%, BAI={indices.bai:.1f})"
        ),
    }


def build_conversation(scene, gt: dict) -> dict:
    """ProcessedScene → messages 形式の会話を返す。"""
    user_prompt = FIRE_DETECTION_USER_PROMPT
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": FIRE_DETECTION_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(gt, ensure_ascii=False)}],
            },
        ],
        "label": gt["fire_detected"],
        "nbr2": float(scene.indices.nbr2),
        "mean_swir22": float(scene.indices.mean_swir22),
        "fire_pixel_ratio": float(scene.indices.fire_pixel_ratio),
    }


# ------------------------------------------------------------------ builder

class DatasetBuilder:
    def __init__(self, save_dir: str = "data/finetune/dataset",
                 max_cloud: float = 50.0):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_cloud = max_cloud
        self.client = SimSatClient()
        self.proc = SpectralProcessor()
        self.records: list[dict] = []
        self._load_checkpoint()

    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        return self.save_dir / "records.jsonl"

    def _load_checkpoint(self):
        p = self._checkpoint_path()
        if p.exists():
            self.records = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            print(f"[Builder] チェックポイントから {len(self.records)} 件をロード")

    def _save_checkpoint(self):
        with open(self._checkpoint_path(), "w") as f:
            for r in self.records:
                rec = {k: v for k, v in r.items() if k != "image"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _save_image(self, img: Image.Image, idx: int) -> str:
        path = self.save_dir / f"img_{idx:04d}.png"
        img.save(path)
        return str(path)

    # ------------------------------------------------------------------

    def fetch_one(self, lon: float, lat: float,
                  timestamp: str) -> Optional[dict]:
        """1シーンを取得して学習サンプルを返す。失敗時は None。"""
        try:
            response = self.client.fetch_fire_scene(
                lon=lon, lat=lat, timestamp=timestamp, size_km=20.0
            )
        except Exception as e:
            print(f"  [skip] fetch error ({lon:.1f},{lat:.1f}): {e}")
            return None

        if not response.image_available:
            print(f"  [skip] image_not_available ({lon:.1f},{lat:.1f})")
            return None
        if response.cloud_cover > self.max_cloud:
            print(f"  [skip] cloud={response.cloud_cover:.0f}% ({lon:.1f},{lat:.1f})")
            return None

        try:
            scene = self.proc.process(response)
        except Exception as e:
            print(f"  [skip] spectral error ({lon:.1f},{lat:.1f}): {e}")
            return None

        gt = build_ground_truth(scene.indices)
        conv = build_conversation(scene, gt)
        conv["image"] = scene.fire_composite   # PIL Image (448×448)
        conv["lon"] = lon
        conv["lat"] = lat
        conv["timestamp"] = timestamp
        conv["cloud_cover"] = response.cloud_cover
        return conv

    # ------------------------------------------------------------------

    def build(self, n_fire: int = 40, n_nofire: int = 40):
        """
        fire / no-fire シーンをそれぞれ n_fire / n_nofire 件収集する。
        チェックポイントから再開可能。
        """
        existing_fire = sum(1 for r in self.records if r["label"])
        existing_nofire = sum(1 for r in self.records if not r["label"])
        print(f"[Builder] 既存: fire={existing_fire}, no_fire={existing_nofire}")

        # ---- fire-prone locations ----
        need_fire = n_fire - existing_fire
        if need_fire > 0:
            print(f"\n[Builder] fire シーン収集 (残{need_fire}件) ...")
            self._collect(FIRE_PRONE_LOCATIONS, TIMESTAMPS,
                          target_label=True, n_needed=need_fire)

        # ---- no-fire locations ----
        need_nofire = n_nofire - existing_nofire
        if need_nofire > 0:
            print(f"\n[Builder] no-fire シーン収集 (残{need_nofire}件) ...")
            self._collect(NO_FIRE_LOCATIONS, TIMESTAMPS,
                          target_label=False, n_needed=need_nofire)

        print(f"\n[Builder] 完了: 合計 {len(self.records)} 件")
        self._finalize()

    def _collect(self, locations, timestamps, target_label: bool, n_needed: int):
        collected = 0
        for lon, lat in locations:
            if collected >= n_needed:
                break
            for ts in timestamps:
                if collected >= n_needed:
                    break
                # 重複チェック
                if any(abs(r["lon"] - lon) < 0.01 and abs(r["lat"] - lat) < 0.01
                       and r["timestamp"] == ts for r in self.records):
                    continue

                print(f"  Fetching ({lon:.2f}, {lat:.2f}) @ {ts[:10]} ...", end=" ", flush=True)
                t0 = time.perf_counter()
                sample = self.fetch_one(lon, lat, ts)
                elapsed = time.perf_counter() - t0

                if sample is None:
                    continue

                actual_label = sample["label"]
                fire_mark = "🔥" if actual_label else "🌿"
                print(f"{fire_mark} NBR2={sample['nbr2']:.3f} SWIR={sample['mean_swir22']:.3f} ({elapsed:.1f}s)")

                # fire-prone ロケーションでも no-fire になりうる → どちらも収集
                idx = len(self.records)
                img_path = self._save_image(sample.pop("image"), idx)
                sample["image_path"] = img_path
                self.records.append(sample)
                self._save_checkpoint()

                if actual_label == target_label:
                    collected += 1

    def _finalize(self):
        """datasets.Dataset として保存。"""
        from datasets import Dataset, Features, Value, Image as DsImage

        imgs, texts, labels = [], [], []
        for r in self.records:
            img = Image.open(r["image_path"]).convert("RGB")
            imgs.append(img)
            texts.append(json.dumps(r["messages"], ensure_ascii=False))
            labels.append(int(r["label"]))

        ds = Dataset.from_dict({
            "image": imgs,
            "messages_json": texts,
            "label": labels,
            "nbr2": [r["nbr2"] for r in self.records],
            "mean_swir22": [r["mean_swir22"] for r in self.records],
            "fire_pixel_ratio": [r["fire_pixel_ratio"] for r in self.records],
        })

        # train / validation 分割
        ds = ds.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
        ds.save_to_disk(str(Path(self.save_dir).parent / "hf_dataset"))
        print(f"[Builder] HF Dataset 保存完了: train={len(ds['train'])}, val={len(ds['test'])}")
        print(f"  fire ratio train: {sum(ds['train']['label'])}/{len(ds['train'])}")
        print(f"  fire ratio val:   {sum(ds['test']['label'])}/{len(ds['test'])}")


# ------------------------------------------------------------------ CLI

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-fire",   type=int, default=40)
    p.add_argument("--n-nofire", type=int, default=40)
    p.add_argument("--save-dir", default="data/finetune/dataset")
    args = p.parse_args()

    builder = DatasetBuilder(save_dir=args.save_dir)
    builder.build(n_fire=args.n_fire, n_nofire=args.n_nofire)
