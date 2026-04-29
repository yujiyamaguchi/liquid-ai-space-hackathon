"""
FireEdge E2E Demo
=================
SimSat (Sentinel-2) → SWIR composite → LoRA fine-tuned LFM → fire alert JSON

Usage:
    cd apps/fireedge
    uv run python demo.py                           # proven fire scene (default)
    uv run python demo.py --lon 96.80 --lat 19.76 --timestamp 2025-03-22T04:13:27Z
    uv run python demo.py --no-fire                 # no-fire scene comparison
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_fetcher import SimSatClient
from src.spectral import SpectralProcessor
from src.interfaces import FIRE_DETECTION_SYSTEM_PROMPT, FIRE_DETECTION_FT_PROMPT

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

MODEL_ID    = "LiquidAI/LFM2.5-VL-450M"
ADAPTER_DIR = ROOT / "output" / "fireedge-lora" / "adapter"

# データセットで実証済みの火災座標 (NBR2_min < -0.18、クラウド 0%)
DEFAULT_FIRE_LON = 106.55
DEFAULT_FIRE_LAT = 15.997
DEFAULT_FIRE_TS  = "2025-03-25T03:33:47Z"

# 比較用: 非火災座標
DEFAULT_NOFIRE_LON = 103.30
DEFAULT_NOFIRE_LAT = 15.00
DEFAULT_NOFIRE_TS  = "2025-06-27T04:00:00Z"  # 乾季後の fire-free 地点


# ---------------------------------------------------------------------------
# 推論
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    for pattern in [r'\{[^{}]*\}', r'\{.*\}']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None


def load_model(device: str = "cuda"):
    print(f"[Demo] Loading base model: {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    print(f"[Demo] Applying LoRA adapter: {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model = model.merge_and_unload()
    model.eval()
    return model, processor


def run_inference(model, processor, image: Image.Image, device: str = "cuda") -> dict:
    messages = [
        {"role": "system", "content": [{"type": "text",  "text": FIRE_DETECTION_SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "image", "image": image},
                                       {"type": "text",  "text": FIRE_DETECTION_FT_PROMPT}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    return _extract_json(raw) or {"fire_detected": False, "raw": raw}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="FireEdge E2E Demo")
    p.add_argument("--lon",       type=float, default=DEFAULT_FIRE_LON)
    p.add_argument("--lat",       type=float, default=DEFAULT_FIRE_LAT)
    p.add_argument("--timestamp", type=str,   default=DEFAULT_FIRE_TS)
    p.add_argument("--no-fire",   action="store_true",
                   help="Use a no-fire scene for comparison")
    p.add_argument("--device",    type=str,   default="cuda")
    p.add_argument("--save-image", type=str,  default=None,
                   help="Save SWIR composite to this path (e.g. scene.png)")
    args = p.parse_args()

    if args.no_fire:
        args.lon, args.lat, args.timestamp = DEFAULT_NOFIRE_LON, DEFAULT_NOFIRE_LAT, DEFAULT_NOFIRE_TS

    print("=" * 60)
    print("FireEdge — Smoke-Penetrating Wildfire Detection")
    print("  LFM 2.5-VL-450M (LoRA fine-tuned) × Sentinel-2 SWIR")
    print("=" * 60)
    print(f"[Demo] Scene: lon={args.lon}  lat={args.lat}")
    print(f"[Demo] Timestamp: {args.timestamp}")

    t_total = time.perf_counter()

    # ------------------------------------------------------------------ SimSat
    print("\n[1/3] Fetching Sentinel-2 scene from SimSat ...")
    client = SimSatClient()
    t0 = time.perf_counter()
    try:
        response = client.fetch_fire_scene(
            lon=args.lon, lat=args.lat, timestamp=args.timestamp
        )
    except Exception as e:
        print(f"[ERROR] SimSat 接続失敗: {e}")
        print("  SimSat が起動しているか確認: curl http://localhost:9005/")
        sys.exit(1)

    fetch_ms = (time.perf_counter() - t0) * 1000
    print(f"  Source:     {response.source}")
    print(f"  Captured:   {response.datetime}")
    print(f"  Cloud:      {response.cloud_cover:.1f}%")
    print(f"  Fetch time: {fetch_ms:.0f} ms")

    if not response.image_available or response.image_array is None:
        print("[WARN] SimSat returned no image for this scene.")
        sys.exit(1)

    # -------------------------------------------------------- Spectral process
    print("\n[2/3] Processing SWIR bands ...")
    proc  = SpectralProcessor()
    scene = proc.process(response)

    print(f"  NBR2_min:    {scene.indices.nbr2_min:.3f}  (< -0.05 → fire candidate)")
    print(f"  SWIR22_max:  {scene.indices.swir22_max:.3f} (> 0.15  → thermal anomaly)")
    spectral_flag = scene.indices.nbr2_min < -0.05 and scene.indices.swir22_max > 0.15
    print(f"  Spectral GT: {'🔥 FIRE (NBR2+SWIR22 threshold)' if spectral_flag else '✅ NO FIRE'}")

    if args.save_image:
        scene.fire_composite.save(args.save_image)
        print(f"  SWIR composite saved: {args.save_image}")

    # -------------------------------------------------------------------- LFM
    print("\n[3/3] Running LFM 2.5-VL-450M inference (LoRA adapter) ...")
    model, processor = load_model(args.device)

    t0 = time.perf_counter()
    result = run_inference(model, processor, scene.fire_composite, args.device)
    infer_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000

    # ------------------------------------------------------------------ Output
    fire_detected = bool(result.get("fire_detected", False))

    alert = {
        "fire_detected":      fire_detected,
        "scene": {
            "lon":            args.lon,
            "lat":            args.lat,
            "timestamp":      args.timestamp,
            "capture":        response.datetime,
            "cloud_pct":      response.cloud_cover,
            "source":         response.source,
        },
        "spectral_indices": {
            "nbr2_min":       round(scene.indices.nbr2_min, 4),
            "swir22_max":     round(scene.indices.swir22_max, 4),
        },
        "performance": {
            "fetch_ms":       round(fetch_ms),
            "infer_ms":       round(infer_ms),
            "total_ms":       round(total_ms),
        },
    }

    print("\n" + "=" * 60)
    status = "🔥 FIRE DETECTED" if fire_detected else "✅ NO FIRE"
    print(f"RESULT: {status}")
    print("=" * 60)
    print(json.dumps(alert, indent=2))
    print(f"\n  Inference: {infer_ms:.0f} ms  |  Total: {total_ms:.0f} ms")

    # Alert size (simulates downlink payload)
    payload = json.dumps(alert, separators=(",", ":")).encode()
    print(f"  Alert size: {len(payload)} bytes  (target < 2048 bytes)")


if __name__ == "__main__":
    main()
