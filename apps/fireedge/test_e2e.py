"""
E2E 動作確認スクリプト
SimSat → DataFetcher → SpectralProcessor → FireDetector の順に検証
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Step 1: SimSat 接続確認 ──────────────────────────────────────────
print("=" * 60)
print("Step 1: SimSat API 接続確認")
print("=" * 60)
from src.data_fetcher import SimSatClient

client = SimSatClient()
pos = client.get_current_position()
print(f"  衛星位置: lon={pos.lon:.2f}, lat={pos.lat:.2f}, alt={pos.alt_km:.1f}km")
print(f"  タイムスタンプ: {pos.timestamp}")

# ── Step 2: Sentinel-2 データ取得 ────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Sentinel-2 全6バンド取得")
print("=" * 60)

# 日本中部 (野火リスクエリア) を指定して確実に画像取得
response = client.fetch_fire_scene(
    lon=138.5, lat=35.3, timestamp="2026-01-15T05:00:00Z", size_km=20.0
)
print(f"  image_available: {response.image_available}")
print(f"  source: {response.source}")
print(f"  cloud_cover: {response.cloud_cover:.1f}%")
print(f"  datetime: {response.datetime}")
if response.image_array is not None:
    print(f"  image_array shape: {response.image_array.shape}")
    print(f"  値範囲: [{response.image_array.min():.4f}, {response.image_array.max():.4f}]")
else:
    print("  image_array: None")

if not response.image_available or response.image_array is None:
    print("ERROR: 画像が取得できませんでした。別の座標を試してください。")
    sys.exit(1)

# ── Step 3: スペクトル処理 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: スペクトル処理 (SpectralProcessor)")
print("=" * 60)
from src.spectral import SpectralProcessor

proc = SpectralProcessor()
scene = proc.process(response)
print(f"  fire_composite size: {scene.fire_composite.size}")
print(f"  rgb_image size:      {scene.rgb_image.size}")
print(f"  NBR2:       {scene.indices.nbr2:.4f}")
print(f"  NDVI:       {scene.indices.ndvi:.4f}")
print(f"  BAI:        {scene.indices.bai:.2f}")
print(f"  mean_swir22:{scene.indices.mean_swir22:.4f}")
print(f"  fire_pixel_ratio: {scene.indices.fire_pixel_ratio:.4f}")

# 画像保存 (デモ用)
os.makedirs("data/samples", exist_ok=True)
scene.fire_composite.save("data/samples/fire_composite_test.png")
scene.rgb_image.save("data/samples/rgb_test.png")
print("  → data/samples/fire_composite_test.png 保存完了")
print("  → data/samples/rgb_test.png 保存完了")

# ── Step 4: LFM 推論 ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: LFM 2.5-VL 推論 (FireDetector)")
print("=" * 60)
from src.detector import FireDetector

detector = FireDetector()
detection = detector.detect(scene)
print(f"  smoke_detected:   {detection.smoke_detected} ({detection.smoke_confidence:.2f})")
print(f"  fire_detected:    {detection.fire_detected}  ({detection.fire_confidence:.2f})")
print(f"  severity:         {detection.severity.value}")
print(f"  alert_recommended:{detection.alert_recommended}")
print(f"  fire_area_ha:     {detection.fire_area_ha:.1f} ha")
print(f"  bbox:             {detection.fire_front_bbox}")
print(f"  inference_time:   {detection.inference_time_ms:.0f} ms")
print(f"\n  description: {detection.description}")

# ── Step 5: フルパイプライン ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: FireEdgePipeline.run() E2E")
print("=" * 60)
from src.pipeline import FireEdgePipeline

pipeline = FireEdgePipeline()
alert = pipeline.run(lon=138.5, lat=35.3, timestamp="2026-01-15T05:00:00Z")
alert_json = pipeline.to_json(alert)
print(alert_json)

with open("data/samples/alert_test.json", "w") as f:
    f.write(alert_json)
print("\n  → data/samples/alert_test.json 保存完了")

json_bytes = len(alert_json.encode())
print(f"  アラートサイズ: {json_bytes} bytes (<2KB目標: {'✓' if json_bytes < 2048 else '✗'})")
print(f"  推論時間: {alert.total_pipeline_time_ms:.0f} ms (<3000ms目標: {'✓' if alert.total_pipeline_time_ms < 3000 else '✗'})")
print(f"  VRAM使用: {alert.peak_vram_mb:.0f} MB (<2048MB目標: {'✓' if alert.peak_vram_mb < 2048 else '✗'})")
print("\nE2E 完了!")
