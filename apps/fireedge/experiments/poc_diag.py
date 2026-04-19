"""
/poc 診断スクリプト — 個別バンド値・pixel 分布を検査する
"""

import os, sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))  # apps/fireedge/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.data_fetcher import SimSatClient

client = SimSatClient()

# Event 1: Sudan, strong fire FRP=196MW, 2026-04-16T10:56
cases = [
    dict(label="Event1_fire_5km",  lat=13.639, lon=34.461, ts="2026-04-16T10:56:00Z", size_km=5),
    dict(label="Event1_fire_20km", lat=13.639, lon=34.461, ts="2026-04-16T10:56:00Z", size_km=20),
    dict(label="Event3_fire_5km",  lat=10.045, lon=30.288, ts="2026-04-17T10:38:00Z", size_km=5),
    dict(label="Event3_fire_20km", lat=10.045, lon=30.288, ts="2026-04-17T10:38:00Z", size_km=20),
]

CH = ["swir22", "swir16", "nir", "red", "green", "blue"]

for c in cases:
    resp = client.fetch_fire_scene(lon=c["lon"], lat=c["lat"],
                                   timestamp=c["ts"], size_km=c["size_km"])
    print(f"\n{'='*60}")
    print(f"  {c['label']}")
    print(f"  image_available={resp.image_available}  cc={resp.cloud_cover:.1f}%")
    print(f"  image_datetime={resp.datetime}  source={resp.source}")
    print(f"  footprint={[f'{v:.3f}' for v in resp.footprint]}")
    if resp.image_array is None:
        print("  image_array=None → スキップ")
        continue

    arr = resp.image_array
    print(f"  array shape={arr.shape}  dtype={arr.dtype}")
    print(f"\n  {'Band':<10} {'min':>8} {'p2':>8} {'mean':>8} {'p98':>8} {'max':>8}  {'nonzero%':>9}")
    for i, name in enumerate(CH):
        ch = arr[:, :, i]
        p2, p98 = float(np.nanpercentile(ch, 2)), float(np.nanpercentile(ch, 98))
        nz_pct = float(np.mean(ch > 0)) * 100
        print(f"  {name:<10} {ch.min():8.4f} {p2:8.4f} {ch.mean():8.4f} {p98:8.4f} {ch.max():8.4f}  {nz_pct:8.1f}%")

    # NBR2 distribution
    swir22 = arr[:, :, 0]
    swir16 = arr[:, :, 1]
    nbr2   = (swir16 - swir22) / (swir16 + swir22 + 1e-10)
    fire_mask = (nbr2 < -0.05) & (swir22 > 0.15)
    print(f"\n  fire_pixel_ratio (NBR2<-0.05 & SWIR22>0.15): {float(np.mean(fire_mask)):.4f}")
    print(f"  NBR2 range: [{nbr2.min():.3f}, {nbr2.max():.3f}]  mean={nbr2.mean():.3f}")
