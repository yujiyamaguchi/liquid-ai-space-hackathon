"""
[1] DataFetcher — SimSat API クライアント
==========================================
SimSat API (localhost:9005) から衛星データを取得し、
SentinelImageResponse として返す。

入力 : なし (現在の衛星位置を自動取得) or (lon, lat, timestamp) 指定
出力 : SentinelImageResponse  ← interfaces.py 参照
"""

from __future__ import annotations

import base64
import io
import time
from typing import Optional

import numpy as np
import requests
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import (
    FIRE_DETECTION_BANDS,
    RECOMMENDED_SIZE_KM,
    SMOKE_DETECTION_BANDS,
    SatellitePosition,
    SentinelImageResponse,
    SpectralBand,
)

SIMSAT_BASE = "http://localhost:9005"
DEFAULT_WINDOW_SECONDS = 864000  # 10 days


class SimSatClient:
    """SimSat API の薄いラッパー。全メソッドは同期。"""

    def __init__(self, base_url: str = SIMSAT_BASE, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # 位置情報
    # ------------------------------------------------------------------

    def get_current_position(self) -> SatellitePosition:
        """衛星の現在位置を取得する。"""
        resp = self._get("/data/current/position")
        lon, lat, alt = resp["lon-lat-alt"]
        return SatellitePosition(
            lon=lon,
            lat=lat,
            alt_km=alt,
            timestamp=resp["timestamp"],
        )

    # ------------------------------------------------------------------
    # Sentinel-2 画像取得
    # ------------------------------------------------------------------

    def fetch_fire_scene(
        self,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
        timestamp: Optional[str] = None,
        size_km: float = RECOMMENDED_SIZE_KM,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> SentinelImageResponse:
        """
        火災検知に必要な全バンドを一括取得する。

        バンド順序 (image_array の ch インデックス):
          ch0: swir22  (B12: 2190nm) — 活火
          ch1: swir16  (B11: 1610nm) — 熱異常
          ch2: nir     (B08: 842nm)  — 植生
          ch3: red     (B04: 665nm)  — 可視赤
          ch4: green   (B03: 560nm)  — 可視緑
          ch5: blue    (B02: 490nm)  — 可視青

        Returns:
            SentinelImageResponse。image_array は (H, W, 6) float32 [0,1]。
            image_available=False の場合は image_array=None。
        """
        all_bands = FIRE_DETECTION_BANDS + SMOKE_DETECTION_BANDS
        band_names = [b.value for b in all_bands]

        params = {
            "spectral_bands": band_names,
            "size_km": size_km,
            "return_type": "array",
            "window_seconds": window_seconds,
        }

        if lon is not None and lat is not None and timestamp is not None:
            params["lon"] = lon
            params["lat"] = lat
            params["timestamp"] = timestamp
            endpoint = "/data/image/sentinel"
        else:
            endpoint = "/data/current/image/sentinel"

        resp = self._get(endpoint, params=params)
        return self._parse_sentinel_response(resp, band_names)

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        r = self._session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _parse_sentinel_response(
        data: dict, requested_bands: list[str]
    ) -> SentinelImageResponse:
        """API レスポンス dict → SentinelImageResponse に変換する。

        実際のAPIレスポンス構造:
          {
            "image": {"metadata": {"shape": [C,H,W], "dtype": "uint16", "bands": [...]},
                      "image": "<base64 raw bytes>"},
            "sentinel_metadata": {"image_available": bool, "source": str,
                                  "spectral_bands": [...], "footprint": [...],
                                  "size_km": float, "cloud_cover": float,
                                  "datetime": str,
                                  "satellite_position": [lon, lat, alt],  # currentのみ
                                  "timestamp": str}                        # currentのみ
          }
        """
        # メタデータは sentinel_metadata キーまたはトップレベル (フォールバック)
        meta = data.get("sentinel_metadata") or data

        pos_raw = meta.get("satellite_position", [0.0, 0.0, 0.0])
        satellite_position = SatellitePosition(
            lon=pos_raw[0],
            lat=pos_raw[1],
            alt_km=pos_raw[2],
            timestamp=meta.get("timestamp", ""),
        )

        image_array = None
        img_entry = data.get("image")
        if meta.get("image_available") and img_entry is not None:
            if isinstance(img_entry, dict) and "image" in img_entry:
                # serialize_xarray_dataset 形式: {metadata: {shape, dtype, bands}, image: b64}
                img_meta = img_entry.get("metadata", {})
                shape = img_meta.get("shape")   # e.g. [3, H, W]  (C, H, W)
                dtype = img_meta.get("dtype", "uint16")
                raw_b64 = img_entry["image"]
                img_bytes = base64.b64decode(raw_b64)
                arr = np.frombuffer(img_bytes, dtype=np.dtype(dtype))
                if shape:
                    arr = arr.reshape(shape)          # (C, H, W)
                    arr = np.transpose(arr, (1, 2, 0))  # → (H, W, C)
                # uint16 → float32 [0, 1]
                if np.issubdtype(np.dtype(dtype), np.integer):
                    arr = arr.astype(np.float32) / np.iinfo(np.dtype(dtype)).max
                else:
                    arr = arr.astype(np.float32)
                image_array = arr
            elif isinstance(img_entry, str):
                # PNG base64
                img_bytes = base64.b64decode(img_entry)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image_array = np.array(pil_img, dtype=np.float32) / 255.0
            elif isinstance(img_entry, list):
                image_array = np.array(img_entry, dtype=np.float32)
                if image_array.ndim == 3 and image_array.shape[0] == len(requested_bands):
                    image_array = np.transpose(image_array, (1, 2, 0))

        fp = meta.get("footprint", [0.0, 0.0, 0.0, 0.0])

        return SentinelImageResponse(
            image_available=meta.get("image_available", False),
            source=meta.get("source", "unknown"),
            spectral_bands=requested_bands,
            footprint=tuple(fp),
            size_km=meta.get("size_km", RECOMMENDED_SIZE_KM),
            cloud_cover=meta.get("cloud_cover", 0.0),
            datetime=meta.get("datetime", ""),
            satellite_position=satellite_position,
            timestamp=meta.get("timestamp", ""),
            image_array=image_array,
        )
