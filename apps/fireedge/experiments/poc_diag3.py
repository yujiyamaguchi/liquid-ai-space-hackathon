"""
/poc 診断3 — 歴史的大規模火災でのシグナル分離確認

テスト対象 (SWIR22 > 0.15 が文献で確認されている事例):
  1. California Camp Fire (2018-11-10): 大規模森林火災
  2. Australia Black Summer (2020-01-04): 数百万 ha 焼失
  3. Brazil Pantanal (2020-10-01): 南米最大湿地帯の大火災
  4. Amazon (2019-08-27): 大規模森林火災

各イベントで:
  - Positive: 火災現場
  - Negative: 同じ画像内の非燃焼エリア (lon +0.5 オフセット)
"""
import os, sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))  # apps/fireedge/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.data_fetcher import SimSatClient

client = SimSatClient()
WINDOW_3D = 259200  # 3 days

cases = [
    # California Camp Fire — 大規模山火事 (Paradise, CA)
    dict(label="CampFire_POS", lat=39.77, lon=-121.62, ts="2018-11-10T20:00:00Z", size_km=5),
    dict(label="CampFire_NEG", lat=39.77, lon=-120.90, ts="2018-11-10T20:00:00Z", size_km=5),

    # Australia Black Summer — NSW (Cobargo area)
    dict(label="AusBlack_POS", lat=-36.4, lon=149.9,   ts="2020-01-04T00:00:00Z", size_km=5),
    dict(label="AusBlack_NEG", lat=-36.4, lon=151.0,   ts="2020-01-04T00:00:00Z", size_km=5),

    # Brazil Pantanal
    dict(label="Pantanal_POS", lat=-17.2, lon=-57.8,   ts="2020-10-01T14:00:00Z", size_km=5),
    dict(label="Pantanal_NEG", lat=-17.2, lon=-56.5,   ts="2020-10-01T14:00:00Z", size_km=5),

    # Amazon (Acre state)
    dict(label="Amazon_POS",   lat=-9.5,  lon=-68.5,   ts="2019-08-27T15:00:00Z", size_km=5),
    dict(label="Amazon_NEG",   lat=-9.5,  lon=-67.5,   ts="2019-08-27T15:00:00Z", size_km=5),
]

print(f"{'='*70}")
print("  歴史的大規模火災 SWIR22 診断")
print(f"{'='*70}")

results = {}
for c in cases:
    resp = client.fetch_fire_scene(
        lon=c["lon"], lat=c["lat"], timestamp=c["ts"],
        size_km=c["size_km"], window_seconds=WINDOW_3D
    )

    if not resp.image_available or resp.image_array is None:
        cc_str = f"{resp.cloud_cover:.0f}%" if resp.cloud_cover is not None else "?"
        print(f"\n[{c['label']}] ❌ image N/A (cc={cc_str} or no tile)")
        results[c["label"]] = None
        continue

    arr = resp.image_array
    swir22 = arr[:, :, 0]
    swir16 = arr[:, :, 1]
    nir    = arr[:, :, 2]
    nbr2   = (swir16 - swir22) / (swir16 + swir22 + 1e-10)
    fire_px = np.mean((nbr2 < -0.05) & (swir22 > 0.15))
    scar_px = np.mean(nbr2 < -0.05)

    results[c["label"]] = {
        "img_dt": resp.datetime, "cc": resp.cloud_cover,
        "swir22_mean": float(swir22.mean()), "swir22_max": float(swir22.max()),
        "nbr2_mean": float(nbr2.mean()), "nbr2_min": float(nbr2.min()),
        "fire_px": float(fire_px), "scar_px": float(scar_px),
    }

    mark = "🔥" if swir22.max() > 0.15 or fire_px > 0.001 else "  "
    print(f"\n{mark} [{c['label']}]")
    print(f"   img_dt={resp.datetime}  cc={resp.cloud_cover:.1f}%  shape={arr.shape}")
    print(f"   SWIR22: mean={swir22.mean():.4f}  max={swir22.max():.4f}")
    print(f"   NBR2:   mean={nbr2.mean():+.3f}  min={nbr2.min():+.3f}")
    print(f"   fire_px_ratio={fire_px:.4f}  scar_px_ratio={scar_px:.4f}")

# --- ペア比較 ---
print(f"\n\n{'='*70}")
print("  ペア比較サマリ")
print(f"{'='*70}")
pairs = [
    ("CampFire_POS", "CampFire_NEG", "California Camp Fire"),
    ("AusBlack_POS", "AusBlack_NEG", "Australia Black Summer"),
    ("Pantanal_POS", "Pantanal_NEG", "Brazil Pantanal"),
    ("Amazon_POS",   "Amazon_NEG",   "Amazon"),
]

for pn, nn, label in pairs:
    pos = results.get(pn)
    neg = results.get(nn)
    if pos is None or neg is None:
        status = "⚠️  データなし"
    else:
        swir_sep = pos["swir22_mean"] > neg["swir22_mean"] + 0.01
        nbr2_sep = pos["nbr2_mean"] < neg["nbr2_mean"] - 0.05
        fire_sig = pos["swir22_max"] > 0.15
        status = "✅ GO" if (swir_sep or nbr2_sep) and fire_sig else \
                 "⚠️  弱いシグナル" if swir_sep or nbr2_sep else "❌ 分離なし"
    print(f"\n  {label}: {status}")
    if pos and neg:
        print(f"    POS SWIR22_max={pos['swir22_max']:.4f}  NEG SWIR22_max={neg['swir22_max']:.4f}")
        print(f"    POS NBR2_mean={pos['nbr2_mean']:+.3f}   NEG NBR2_mean={neg['nbr2_mean']:+.3f}")
        print(f"    fire_pixel_ratio: POS={pos['fire_px']:.4f}  NEG={neg['fire_px']:.4f}")
