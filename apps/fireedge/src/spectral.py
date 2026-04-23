"""
[2] SpectralProcessor — バンド合成・スペクトル指標計算
======================================================
SentinelImageResponse を受け取り、LFM 2.5-VL への入力となる
ProcessedScene を生成する。

バンドチャネルインデックス (data_fetcher.py と一致させること):
  ch0: swir22, ch1: swir16, ch2: nir
  ch3: red,    ch4: green,  ch5: blue
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .interfaces import (
    ProcessedScene,
    SentinelImageResponse,
    SpectralIndices,
    SpectralPreprocessConfig,
)

# チャネルインデックス (fetch_fire_scene のバンド順序に対応)
CH_SWIR22 = 0
CH_SWIR16 = 1
CH_NIR    = 2
CH_RED    = 3
CH_GREEN  = 4
CH_BLUE   = 5

# 火災画素判定閾値 (interfaces.py SPECTRAL_FORMULAS に準拠)
NBR2_FIRE_THRESHOLD  = -0.05
SWIR22_FIRE_MIN      = 0.15

# 煙画素判定閾値
SMOKE_BRIGHTNESS_MIN = 0.6
SMOKE_NDVI_MAX       = 0.1


class SpectralProcessor:
    """numpy ベースのスペクトル処理エンジン。GPU 不要、エッジ動作前提。"""

    def __init__(self, config: SpectralPreprocessConfig | None = None) -> None:
        self.cfg = config or SpectralPreprocessConfig()

    def process(self, response: SentinelImageResponse) -> ProcessedScene:
        """
        SentinelImageResponse → ProcessedScene

        Raises:
            ValueError: image_array が None (画像未取得) の場合
        """
        if response.image_array is None:
            raise ValueError(
                f"image_available={response.image_available}. "
                "画像データがありません。cloud_cover や window_seconds を確認してください。"
            )

        arr = response.image_array  # (H, W, 6) float32 [0, 1]

        # --- バンド分離 ---
        swir22 = arr[:, :, CH_SWIR22]
        swir16 = arr[:, :, CH_SWIR16]
        nir    = arr[:, :, CH_NIR]
        red    = arr[:, :, CH_RED]
        green  = arr[:, :, CH_GREEN]
        blue   = arr[:, :, CH_BLUE]

        # --- スペクトル指標計算 ---
        indices = self._compute_indices(swir22, swir16, nir, red)

        # --- 疑似カラー合成 (SWIR22, SWIR16, NIR → R, G, B) ---
        fire_composite_arr = np.stack([swir22, swir16, nir], axis=-1)
        fire_img = self._to_pil(fire_composite_arr, per_band=self.cfg.normalize_per_band)

        # --- 通常 RGB 画像 ---
        rgb_arr = np.stack([red, green, blue], axis=-1)
        rgb_img = self._to_pil(rgb_arr, per_band=False)

        return ProcessedScene(
            fire_composite=fire_img,
            rgb_image=rgb_img,
            indices=indices,
            footprint=response.footprint,
            capture_datetime=response.datetime,
            cloud_cover=response.cloud_cover,
        )

    # ------------------------------------------------------------------
    # スペクトル指標
    # ------------------------------------------------------------------

    def _compute_indices(
        self,
        swir22: np.ndarray,
        swir16: np.ndarray,
        nir: np.ndarray,
        red: np.ndarray,
    ) -> SpectralIndices:
        eps = 1e-10

        # NBR2 = (swir16 - swir22) / (swir16 + swir22)
        nbr2_arr = (swir16 - swir22) / (swir16 + swir22 + eps)
        nbr2 = float(np.nanmean(nbr2_arr))
        nbr2_min = float(np.nanmin(nbr2_arr))   # 火炎ピクセルに感応 (poc2 と同じ)

        # NDVI = (nir - red) / (nir + red)
        ndvi_arr = (nir - red) / (nir + red + eps)
        ndvi = float(np.nanmean(ndvi_arr))

        # BAI = 1 / ((0.1 - red)^2 + (0.06 - nir)^2)
        bai_arr = 1.0 / ((0.1 - red) ** 2 + (0.06 - nir) ** 2 + eps)
        bai = float(np.nanmean(bai_arr))

        mean_swir22 = float(np.nanmean(swir22))
        swir22_max  = float(np.nanmax(swir22))   # 熱異常ピクセルに感応 (poc2 と同じ)

        # 火災画素比率
        fire_mask = (nbr2_arr < NBR2_FIRE_THRESHOLD) & (swir22 > SWIR22_FIRE_MIN)
        fire_pixel_ratio = float(np.mean(fire_mask))

        return SpectralIndices(
            nbr2=nbr2,
            nbr2_min=nbr2_min,
            ndvi=ndvi,
            bai=bai,
            mean_swir22=mean_swir22,
            swir22_max=swir22_max,
            fire_pixel_ratio=fire_pixel_ratio,
        )

    # ------------------------------------------------------------------
    # 画像変換ユーティリティ
    # ------------------------------------------------------------------

    def _to_pil(self, arr: np.ndarray, per_band: bool = True) -> Image.Image:
        """
        (H, W, 3) float32 → PIL RGB Image (target_size_px × target_size_px)

        Steps:
          1. パーセンタイルクリップ (外れ値除去)
          2. [0, 1] 正規化
          3. uint8 変換
          4. リサイズ (Lanczos)
        """
        clipped = self._percentile_clip(arr, per_band=per_band)
        uint8 = (clipped * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(uint8, mode="RGB")
        size = self.cfg.target_size_px
        return pil.resize((size, size), Image.LANCZOS)

    def _percentile_clip(self, arr: np.ndarray, per_band: bool) -> np.ndarray:
        lo, hi = self.cfg.percentile_clip
        out = np.empty_like(arr, dtype=np.float32)

        if per_band:
            # SWIR22(ch0) と SWIR16(ch1) は共有スケールで正規化し、バンド間比を保持する。
            # 独立正規化では弱火炎の SWIR22>SWIR16 差分が圧縮され、burn scar の赤色が消える。
            swir = arr[:, :, :2]
            vmin = float(np.nanpercentile(swir, lo))
            vmax = float(np.nanpercentile(swir, hi))
            span = vmax - vmin if vmax > vmin else 1.0
            out[:, :, 0] = (arr[:, :, 0] - vmin) / span
            out[:, :, 1] = (arr[:, :, 1] - vmin) / span
            # NIR 以降は独立
            for c in range(2, arr.shape[2]):
                ch = arr[:, :, c]
                vmin_c = float(np.nanpercentile(ch, lo))
                vmax_c = float(np.nanpercentile(ch, hi))
                span_c = vmax_c - vmin_c if vmax_c > vmin_c else 1.0
                out[:, :, c] = (ch - vmin_c) / span_c
        else:
            vmin = float(np.nanpercentile(arr, lo))
            vmax = float(np.nanpercentile(arr, hi))
            span = vmax - vmin if vmax > vmin else 1.0
            out = (arr - vmin) / span

        return np.clip(out, 0.0, 1.0)
