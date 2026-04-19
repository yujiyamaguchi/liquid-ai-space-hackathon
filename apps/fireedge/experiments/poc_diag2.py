"""
/poc 診断2 — タイムスタンプを「今日」に設定して burn scar を探す

戦略:
  - FIRMSで確認された火災座標に対し、timestamp=今日 (2026-04-18) で SimSat クエリ
  - window_seconds=432000 (5日) で April 13-18 の最新 S2 画像を取得
  - 焼け跡 (burn scar) と近傍の非燃焼地を比較
"""
import os, sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))  # apps/fireedge/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.data_fetcher import SimSatClient

client = SimSatClient()

# 今日の日時 (UTC)
TODAY_TS   = "2026-04-18T23:59:59Z"
WINDOW_5D  = 432000   # 5日

# --- テストケース ---
# Sudan fire cluster (April 16, FRP=196MW): burn scar を today で見る
# + nearby non-fire (lat+2)
cases = [
    # Sudan fire site — 今日のタイムスタンプで過去5日の最新 S2 を取得
    dict(label="Sudan_fire_BURNSITE",    lat=13.639, lon=34.461, ts=TODAY_TS, size_km=5),
    dict(label="Sudan_fire_NOFIRE",      lat=15.639, lon=34.461, ts=TODAY_TS, size_km=5),
    # South Sudan fire (April 17, FRP=82MW)
    dict(label="SouthSudan_fire_SITE",   lat=10.045, lon=30.288, ts=TODAY_TS, size_km=5),
    dict(label="SouthSudan_fire_NOFIRE", lat=12.045, lon=30.288, ts=TODAY_TS, size_km=5),
    # West Africa (Ghana/Ivory Coast border) — dry season burn scar hotspot
    dict(label="WestAfrica_BURNSITE",    lat=8.0,    lon=-1.0,   ts=TODAY_TS, size_km=5),
    dict(label="WestAfrica_NOFIRE",      lat=10.0,   lon=-1.0,   ts=TODAY_TS, size_km=5),
]

CH = ["swir22", "swir16", "nir", "red", "green", "blue"]

print(f"{'='*70}")
print(f"  Burn scar vs Non-fire 診断 (timestamp={TODAY_TS}, window=5日)")
print(f"{'='*70}")

results = []
for c in cases:
    resp = client.fetch_fire_scene(
        lon=c["lon"], lat=c["lat"], timestamp=c["ts"],
        size_km=c["size_km"], window_seconds=WINDOW_5D
    )
    if not resp.image_available or resp.image_array is None:
        print(f"\n[{c['label']}] ❌ image_available=False / None")
        results.append(None)
        continue

    arr = resp.image_array
    swir22 = arr[:, :, 0]
    swir16 = arr[:, :, 1]
    nir    = arr[:, :, 2]
    red    = arr[:, :, 3]

    nbr2 = (swir16 - swir22) / (swir16 + swir22 + 1e-10)
    ndvi = (nir - red) / (nir + red + 1e-10)

    # 焼け跡スレッショルド: NBR2 < -0.05 (active) or NBR2 < -0.02 (scar)
    fire_px    = np.mean((nbr2 < -0.05) & (swir22 > 0.15))
    scar_px    = np.mean(nbr2 < -0.05)

    info = {
        "label": c["label"],
        "img_dt": resp.datetime,
        "cc": resp.cloud_cover,
        "nbr2_mean": float(nbr2.mean()),
        "nbr2_min":  float(nbr2.min()),
        "nbr2_p5":   float(np.percentile(nbr2, 5)),
        "swir22_mean": float(swir22.mean()),
        "swir22_max":  float(swir22.max()),
        "ndvi_mean":   float(ndvi.mean()),
        "fire_px_ratio": float(fire_px),
        "scar_px_ratio": float(scar_px),
    }
    results.append(info)

    print(f"\n[{c['label']}]")
    print(f"  S2 datetime: {info['img_dt']}  cloud={info['cc']:.1f}%  shape={arr.shape}")
    print(f"  NBR2  mean={info['nbr2_mean']:+.3f}  min={info['nbr2_min']:+.3f}  p5={info['nbr2_p5']:+.3f}")
    print(f"  SWIR22 mean={info['swir22_mean']:.4f}  max={info['swir22_max']:.4f}")
    print(f"  NDVI  mean={info['ndvi_mean']:+.3f}")
    print(f"  fire_px_ratio (NBR2<-0.05 & SWIR22>0.15): {info['fire_px_ratio']:.4f}")
    print(f"  scar_px_ratio (NBR2<-0.05 only):           {info['scar_px_ratio']:.4f}")

# --- ペア比較サマリ ---
print(f"\n\n{'='*70}")
print("  ペア比較 (Positive vs Negative)")
print(f"{'='*70}")
pairs = [
    ("Sudan_fire_BURNSITE", "Sudan_fire_NOFIRE"),
    ("SouthSudan_fire_SITE", "SouthSudan_fire_NOFIRE"),
    ("WestAfrica_BURNSITE", "WestAfrica_NOFIRE"),
]
by_label = {r["label"]: r for r in results if r is not None}

for pos_name, neg_name in pairs:
    pos = by_label.get(pos_name)
    neg = by_label.get(neg_name)
    if pos is None or neg is None:
        print(f"\n  {pos_name} vs {neg_name}: データ不足")
        continue
    print(f"\n  {pos_name} (FIRE) vs {neg_name} (NO-FIRE)")
    print(f"  {'指標':<25} {'FIRE':>10} {'NO-FIRE':>10}  {'分離':>8}")
    for key in ["nbr2_mean", "nbr2_min", "swir22_mean", "swir22_max", "scar_px_ratio"]:
        pv, nv = pos[key], neg[key]
        # 分離判定: fire site が「より fire-like」か
        if "swir" in key:
            sep = "✅" if pv > nv + 0.005 else "❌"
        else:
            sep = "✅" if pv < nv - 0.02 else "❌"
        print(f"  {key:<25} {pv:>10.4f} {nv:>10.4f}  {sep:>8}")
