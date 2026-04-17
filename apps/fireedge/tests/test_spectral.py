"""
SpectralProcessor の単体テスト。GPU・SimSat API 不要で完結。
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import SatellitePosition, SentinelImageResponse, SpectralPreprocessConfig
from src.spectral import SpectralProcessor, NBR2_FIRE_THRESHOLD


def _make_response(arr: np.ndarray) -> SentinelImageResponse:
    return SentinelImageResponse(
        image_available=True,
        source="sentinel-2a",
        spectral_bands=["swir22", "swir16", "nir", "red", "green", "blue"],
        footprint=(0.0, 0.0, 0.1, 0.1),
        size_km=20.0,
        cloud_cover=5.0,
        datetime="2026-01-15T05:00:00Z",
        satellite_position=SatellitePosition(lon=138.5, lat=35.3, alt_km=550.0, timestamp="2026-01-15T05:00:00Z"),
        timestamp="2026-01-15T05:00:00Z",
        image_array=arr,
    )


def make_scene(swir22_val, swir16_val, nir_val=0.5, red_val=0.1, g=0.1, b=0.1):
    """全画素が同一値のシンプルなテスト画像 (64x64x6)"""
    arr = np.zeros((64, 64, 6), dtype=np.float32)
    arr[:, :, 0] = swir22_val   # swir22
    arr[:, :, 1] = swir16_val   # swir16
    arr[:, :, 2] = nir_val      # nir
    arr[:, :, 3] = red_val      # red
    arr[:, :, 4] = g            # green
    arr[:, :, 5] = b            # blue
    return arr


class TestNBR2:
    def test_active_fire_gives_negative_nbr2(self):
        """活火: swir22 >> swir16 → NBR2 < 0"""
        arr = make_scene(swir22_val=0.8, swir16_val=0.3)
        proc = SpectralProcessor()
        scene = proc.process(_make_response(arr))
        assert scene.indices.nbr2 < 0, f"expected NBR2 < 0, got {scene.indices.nbr2}"

    def test_healthy_vegetation_gives_positive_nbr2(self):
        """健全植生: swir16 > swir22 → NBR2 > 0"""
        arr = make_scene(swir22_val=0.1, swir16_val=0.4)
        proc = SpectralProcessor()
        scene = proc.process(_make_response(arr))
        assert scene.indices.nbr2 > 0, f"expected NBR2 > 0, got {scene.indices.nbr2}"

    def test_fire_pixel_ratio_high_for_active_fire(self):
        """活火シーンでは火災画素比率が高くなること"""
        arr = make_scene(swir22_val=0.8, swir16_val=0.2)  # NBR2 ≈ -0.6, swir22 > 0.15
        proc = SpectralProcessor()
        scene = proc.process(_make_response(arr))
        assert scene.indices.fire_pixel_ratio > 0.9


class TestOutputShape:
    def test_fire_composite_size(self):
        """fire_composite は設定サイズの正方形 PIL 画像であること"""
        cfg = SpectralPreprocessConfig(target_size_px=224)
        arr = make_scene(0.5, 0.3)
        scene = SpectralProcessor(cfg).process(_make_response(arr))
        assert scene.fire_composite.size == (224, 224)
        assert scene.fire_composite.mode == "RGB"

    def test_rgb_image_size(self):
        arr = make_scene(0.5, 0.3)
        scene = SpectralProcessor().process(_make_response(arr))
        assert scene.rgb_image.size == (448, 448)

    def test_raises_on_missing_image(self):
        """image_array=None の場合は ValueError"""
        resp = _make_response(make_scene(0.5, 0.3))
        resp.image_array = None
        with pytest.raises(ValueError):
            SpectralProcessor().process(resp)
