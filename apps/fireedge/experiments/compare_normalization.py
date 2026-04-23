"""compare_normalization.py — SWIR 正規化比較
=============================================
FN サンプルを SimSat から再取得し、旧正規化 (per-band 独立) vs
新正規化 (SWIR22+SWIR16 共有) を同一の生データで並べて比較する。

使い方:
    cd apps/fireedge
    uv run python experiments/compare_normalization.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# .env ロード
_env = ROOT / ".." / ".." / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from src.data_fetcher import SimSatClient
from src.spectral import SpectralProcessor

# FN サンプル（records.jsonl から特定済み）
# idx=96: NBR2_min=-0.117, idx=90: NBR2_min=-0.095, idx=132: NBR2_min=-0.083
FN_SAMPLES = [
    {"lon": 81.44682,  "lat": 51.07901,  "ts": "2026-04-23", "nbr2_min": -0.117},
    {"lon": 147.49368, "lat": -34.32925, "ts": "2026-04-23", "nbr2_min": -0.095},
    {"lon": 102.34390, "lat": 18.77505,  "ts": "2026-04-21", "nbr2_min": -0.083},
]

WINDOW_SEC = 12 * 86400
SIZE_KM    = 5.0
TARGET_PX  = 448
OUT_DIR    = ROOT / "data" / "finetune" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def old_percentile_clip(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """旧正規化: 全バンド独立でパーセンタイルクリップ。"""
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        vmin = float(np.nanpercentile(ch, lo))
        vmax = float(np.nanpercentile(ch, hi))
        span = vmax - vmin if vmax > vmin else 1.0
        out[:, :, c] = (ch - vmin) / span
    return np.clip(out, 0.0, 1.0)


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB").resize((TARGET_PX, TARGET_PX), Image.LANCZOS)


def main():
    client  = SimSatClient()
    new_proc = SpectralProcessor()  # shared SWIR 正規化

    n = len(FN_SAMPLES)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "SWIR 正規化比較 (FN サンプル)\n"
        "旧: 全バンド独立  /  新: SWIR22+SWIR16 共有・NIR 独立",
        fontsize=13, fontweight="bold",
    )

    col_titles = [
        "旧正規化 (per-band 独立)",
        "新正規化 (SWIR22/SWIR16 共有)",
        "差分 |新 − 旧|",
    ]
    for col_i, t in enumerate(col_titles):
        axes[0, col_i].set_title(t, fontsize=10, fontweight="bold")

    for row, s in enumerate(FN_SAMPLES):
        lon, lat, ts, nbr2 = s["lon"], s["lat"], s["ts"], s["nbr2_min"]
        print(f"\n[{row+1}/{n}] lon={lon}, lat={lat}, ts={ts}")

        try:
            resp = client.fetch_fire_scene(
                lon=lon, lat=lat, timestamp=ts,
                size_km=SIZE_KM, window_seconds=WINDOW_SEC,
            )
        except Exception as e:
            print(f"  ⚠️  取得失敗: {e}")
            for ax in axes[row]:
                ax.axis("off")
            continue

        if not resp.image_available or resp.image_array is None:
            print(f"  ⚠️  画像なし (cc={resp.cloud_cover}%)")
            for ax in axes[row]:
                ax.axis("off")
            continue

        raw = resp.image_array  # (H, W, 6) float32
        composite_raw = raw[:, :, :3]  # SWIR22, SWIR16, NIR

        # 旧正規化: 全バンド独立
        old_norm = old_percentile_clip(composite_raw)
        old_img  = arr_to_pil(old_norm)

        # 新正規化: SpectralProcessor (SWIR 共有)
        new_scene = new_proc.process(resp)
        new_img   = new_scene.fire_composite

        # 差分
        old_arr  = np.array(old_img).astype(np.float32) / 255.0
        new_arr  = np.array(new_img).astype(np.float32) / 255.0
        diff_arr = np.abs(new_arr - old_arr)
        diff_img = Image.fromarray((diff_arr * 255).clip(0, 255).astype(np.uint8), mode="RGB")

        # 旧のスペクトル値（参考）
        swir22 = raw[:, :, 0]
        swir16 = raw[:, :, 1]
        swir22_max = float(np.nanmax(swir22))
        nbr2_calc  = float(np.nanmin((swir16 - swir22) / (swir16 + swir22 + 1e-10)))

        for col_i, img in enumerate([old_img, new_img, diff_img]):
            ax = axes[row, col_i]
            ax.imshow(img)
            ax.axis("off")

        row_label = (
            f"FN sample\n"
            f"NBR2_min={nbr2:.3f}\n"
            f"SWIR22_max={swir22_max:.3f}\n"
            f"({lat:.2f}°, {lon:.2f}°)"
        )
        axes[row, 0].set_ylabel(row_label, fontsize=9, rotation=0,
                                labelpad=140, va="center")

        diff_mean = float(diff_arr.mean())
        print(f"  ✅ 成功  cc={resp.cloud_cover:.0f}%  "
              f"SWIR22_max={swir22_max:.3f}  NBR2_min(再計算)={nbr2_calc:.3f}  "
              f"diff_mean={diff_mean:.3f}")

    plt.tight_layout()
    out_path = OUT_DIR / "compare_normalization.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n[compare] 保存: {out_path}")


if __name__ == "__main__":
    main()
