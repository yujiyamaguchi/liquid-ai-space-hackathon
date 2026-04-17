"""
[4] Pipeline — FireEdge エンドツーエンドオーケストレーター
==========================================================
DataFetcher → SpectralProcessor → FireDetector を順に呼び出し、
FireEdgeAlert (< 2KB JSON) を生成する。

使い方:
    pipeline = FireEdgePipeline()
    alert = pipeline.run()        # 現在の衛星位置で実行
    print(alert.to_json())

    # または緯度経度・時刻を指定
    alert = pipeline.run(lon=138.5, lat=35.3, timestamp="2026-01-15T05:00:00Z")
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime, timezone
from typing import Optional

import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import (
    FireEdgeAlert,
    LFMInferenceConfig,
    SpectralPreprocessConfig,
)
from src.data_fetcher import SimSatClient
from src.detector import FireDetector
from src.spectral import SpectralProcessor


class FireEdgePipeline:
    """
    衛星エッジ上で動作する煙透過型火災検知パイプライン。

    リソース目標 (RTX 5090 / エッジ環境):
      - VRAM  : < 2 GB (bfloat16 での LFM 2.5-VL-450M)
      - レイテンシ: < 3 秒 / シーン (データ取得込み)
      - 出力   : < 2 KB JSON アラート
    """

    def __init__(
        self,
        simsat_url: str = "http://localhost:9005",
        spectral_config: Optional[SpectralPreprocessConfig] = None,
        lfm_config: Optional[LFMInferenceConfig] = None,
    ) -> None:
        self.client    = SimSatClient(base_url=simsat_url)
        self.spectral  = SpectralProcessor(config=spectral_config)
        self.detector  = FireDetector(config=lfm_config)

    # ------------------------------------------------------------------
    # メインエントリポイント
    # ------------------------------------------------------------------

    def run(
        self,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
        timestamp: Optional[str] = None,
        size_km: float = 20.0,
    ) -> FireEdgeAlert:
        """
        E2E 推論を実行して FireEdgeAlert を返す。

        Args:
            lon, lat, timestamp : 指定なしの場合は現在の衛星位置を自動取得
            size_km             : シーンサイズ [km]

        Returns:
            FireEdgeAlert (dataclasses.asdict() で JSON 変換可能)
        """
        t_start = time.perf_counter()

        # [1] データ取得
        print("[Pipeline] Fetching satellite data ...")
        response = self.client.fetch_fire_scene(
            lon=lon, lat=lat, timestamp=timestamp, size_km=size_km
        )

        if not response.image_available:
            raise RuntimeError(
                f"No imagery available for the requested location/time. "
                f"cloud_cover={response.cloud_cover:.1f}%"
            )

        # [2] スペクトル処理
        print("[Pipeline] Processing spectral bands ...")
        scene = self.spectral.process(response)

        # [3] LFM 推論
        print("[Pipeline] Running LFM 2.5-VL inference ...")
        detection = self.detector.detect(scene)

        # [4] アラート組み立て
        total_ms = (time.perf_counter() - t_start) * 1000
        peak_vram = _get_peak_vram_mb()

        # satellite_position が (0,0,0) の場合は footprint の中心座標を使用
        fp = response.footprint  # (lon_min, lat_min, lon_max, lat_max)
        pos_lon = response.satellite_position.lon if response.satellite_position.lon != 0.0 else (fp[0] + fp[2]) / 2
        pos_lat = response.satellite_position.lat if response.satellite_position.lat != 0.0 else (fp[1] + fp[3]) / 2
        scene_id = _make_scene_id(
            response.source,
            response.datetime,
            pos_lon,
            pos_lat,
        )

        data_quality = _assess_quality(response.cloud_cover)

        alert = FireEdgeAlert(
            scene_id=scene_id,
            satellite_source=response.source,
            capture_datetime=response.datetime,
            processed_datetime=datetime.now(timezone.utc).isoformat(),
            footprint=response.footprint,
            satellite_position=response.satellite_position,
            detection=detection,
            cloud_cover=response.cloud_cover,
            data_quality=data_quality,
            indices=scene.indices,
            total_pipeline_time_ms=total_ms,
            peak_vram_mb=peak_vram,
        )

        print(
            f"[Pipeline] Done. "
            f"fire={detection.fire_detected}, "
            f"severity={detection.severity.value}, "
            f"time={total_ms:.0f}ms, "
            f"VRAM={peak_vram:.0f}MB"
        )
        return alert

    def to_json(self, alert: FireEdgeAlert, indent: int = 2, include_raw: bool = False) -> str:
        """FireEdgeAlert を JSON 文字列にシリアライズする。
        include_raw=False の場合 raw_llm_output を除外してサイズを削減する。
        """
        d = _alert_to_dict(alert)
        if not include_raw and "detection" in d:
            d["detection"].pop("raw_llm_output", None)
        return json.dumps(d, indent=indent, ensure_ascii=False)


# ------------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------------

def _make_scene_id(source: str, dt: str, lon: float, lat: float) -> str:
    safe_dt = dt.replace(":", "").replace("-", "")[:15]
    return f"{source}_{safe_dt}_{lon:.2f}_{lat:.2f}"


def _assess_quality(cloud_cover: float) -> str:
    if cloud_cover < 30:
        return "GOOD"
    if cloud_cover < 60:
        return "DEGRADED"
    return "POOR"


def _get_peak_vram_mb() -> float:
    try:
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        return 0.0


def _alert_to_dict(alert: FireEdgeAlert) -> dict:
    """dataclass を JSON シリアライズ可能な dict に変換する。Enum は .value に変換。"""
    raw = dataclasses.asdict(alert)

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if hasattr(obj, "value"):  # Enum
            return obj.value
        return obj

    return _convert(raw)
