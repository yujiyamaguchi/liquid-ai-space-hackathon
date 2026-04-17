# FireEdge 🔥🛰️
### Smoke-Penetrating Wildfire Detection on Satellite Edge

**LFM 2.5-VL-450M × Sentinel-2 SWIR | AI in Space Hackathon 2026**

---

## The Problem

Wildfires kill thousands and burn millions of hectares annually. Existing satellite detection relies on RGB imagery — but **smoke hides fire**. By the time a fire is visible from the ground or to RGB cameras, it may have already spread uncontrollably.

## The Solution

> **What RGB can't see, SWIR can.**

FireEdge runs **entirely on-orbit**, analyzing Sentinel-2 SWIR bands (1.6μm & 2.2μm) that **penetrate smoke** to reveal hidden fire fronts. The 450M-parameter LFM model fits comfortably in a satellite's edge compute budget.

```
Cloud/Smoke → RGB sees nothing   SWIR 2.2μm → Fire glows bright red/orange
```

![RGB vs SWIR Comparison](data/samples/rgb_vs_swir_comparison.png)

---

## Why Edge AI?

| | Cloud AI | **FireEdge (Edge AI)** |
|---|---|---|
| Bandwidth | ~4 MB raw image downlink | **< 2 KB alert JSON** (3000× smaller) |
| Latency | Hours (wait for ground station) | **Seconds (on-orbit real-time)** |
| Smoke | RGB only → blind under smoke | **SWIR penetrates smoke** |
| Privacy | Raw imagery leaves the satellite | **Inference on-board, no raw data** |
| Power | Large GPU required | **450M model, ~900 MB VRAM** |

---

## Architecture

```
SimSat API (Sentinel-2)
        │
        ▼
  DataFetcher          ← 6 spectral bands (SWIR22, SWIR16, NIR, R, G, B)
        │
        ▼
  SpectralProcessor    ← SWIR false-color composite + NBR2/NDVI/BAI indices
        │
        ▼
  LFM 2.5-VL-450M      ← Vision-language reasoning over false-color image
        │
        ▼
  FireEdgeAlert JSON   ← < 2 KB, downlinked to ground
        │
        ▼
  FIRMSValidator       ← Ground truth cross-check with NASA FIRMS VIIRS
```

---

## Spectral Science

The core insight is the **NBR2 (Normalized Burn Ratio 2)** combined with SWIR2.2 brightness:

```
NBR2 = (SWIR1.6 - SWIR2.2) / (SWIR1.6 + SWIR2.2)

Active fire:  NBR2 < -0.05  AND  SWIR2.2 > 0.15
Burn scar:    NBR2 < -0.05  (lower SWIR2.2)
Healthy veg:  NBR2 > 0.3
```

The false-color composite assigns:
- **Red channel** = SWIR 2.2μm → active fire appears **bright red/orange**
- **Green channel** = SWIR 1.6μm → thermal anomalies appear **green**
- **Blue channel** = NIR 865nm → healthy vegetation appears **blue**

---

## Quick Start

### Prerequisites
- Docker (for SimSat)
- Python 3.10+ with `uv`
- NVIDIA GPU recommended (RTX 5090 tested, ~900 MB VRAM)
- Hugging Face account (for LFM model)

### 1. Start SimSat
```bash
cd /path/to/SimSat
docker compose up -d
```

### 2. Start simulation
```bash
curl -X POST http://localhost:8000/api/commands/ \
  -H "Content-Type: application/json" \
  -d '{"command": "start", "start_time": "2026-01-15T05:00:00Z", "replay_speed": 10.0}'
```

### 3. Install dependencies
```bash
uv sync
uv run hf auth login  # Hugging Face login for LFM model
```

### 4. Run E2E pipeline
```python
from src.pipeline import FireEdgePipeline

pipeline = FireEdgePipeline()

# Option A: Current satellite position
alert = pipeline.run()

# Option B: Specific location (Australia summer fire zone)
alert = pipeline.run(lon=142.0, lat=-30.0, timestamp="2026-01-15T05:00:00Z")

print(pipeline.to_json(alert))
```

### 5. Open demo notebook
```bash
uv run jupyter notebook docs/demo.ipynb
```

---

## Performance Results

| Metric | Result | Target |
|---|---|---|
| Alert JSON size | **1,273 bytes** | < 2,048 bytes ✓ |
| Peak VRAM | **~900 MB** | < 2,048 MB ✓ |
| Inference time | **~5 sec** (model cached) | < 10 sec ✓ |
| Compression vs raw | **~3,000×** | — |
| Fire Confidence | **0.6–0.8** | > 0.5 ✓ |
| FIRMS True Positive | **17 hotspots confirmed** | Ground truth ✓ |

---

## Project Structure

```
fireedge/
├── interfaces.py          # SDD: all type definitions & prompts
├── src/
│   ├── data_fetcher.py    # SimSat API client (Sentinel-2 bands)
│   ├── spectral.py        # SWIR false-color + NBR2/NDVI/BAI
│   ├── detector.py        # LFM 2.5-VL-450M inference engine
│   ├── pipeline.py        # End-to-end orchestrator
│   └── validator.py       # NASA FIRMS ground truth cross-check
├── tests/
│   └── test_spectral.py   # 6/6 unit tests passing
├── docs/
│   └── demo.ipynb         # Interactive demo notebook
└── data/
    ├── samples/           # Generated demo images & alerts
    └── firms_cache/       # Cached NASA FIRMS VIIRS CSV responses
```

---

## Ground Truth Validation (NASA FIRMS)

FireEdge cross-checks its detections against **NASA FIRMS VIIRS SNPP NRT** hotspot data.

### Validation Result — Australia QLD (2026-01-15)

![FIRMS Ground Truth Validation](data/samples/firms_validation.png)

| Metric | Result |
|---|---|
| FIRMS VIIRS hotspots in footprint | **17** |
| True Positive (FireEdge ✓ & FIRMS ✓) | **True** |
| Total Fire Radiative Power | **539 MW** |
| Nearest FIRMS hotspot to predicted bbox | **6.34 km** |
| Spatial agreement (< 10 km) | **✓** |

```
=== FIRMS Ground Truth 照合結果 ===
  フットプリント内 FIRMS ホットスポット数: 17
  True Positive  (FireEdge ✓, FIRMS ✓): True
  False Positive (FireEdge ✓, FIRMS ✗): False
  False Negative (FireEdge ✗, FIRMS ✓): False
  最近傍 FIRMS 点との距離: 6.34 km
```

```python
from src.validator import FIRMSValidator

validator = FIRMSValidator(map_key="YOUR_FIRMS_MAP_KEY")
result = validator.validate(alert)
print(validator.summarize(result))
```

Get a free FIRMS API key at: https://firms.modaps.eosdis.nasa.gov/api/

---

## Hackathon Judging Alignment

| Criterion | FireEdge's Approach |
|---|---|
| **Innovation** | SWIR smoke-penetration + LFM VL reasoning — first on-orbit wildfire pipeline with this combination |
| **Impact** | Real-time fire alerts in < 2 KB, enabling rapid response before ground observers detect the fire |
| **Complexity** | Multispectral band fusion (6 bands), spectral index computation, VLM grounding, FIRMS cross-validation |
| **Efficiency** | 450M params, bfloat16, < 1 GB VRAM — purpose-built for satellite edge constraints |

---

*Built with [LFM 2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) by Liquid AI × [SimSat](https://github.com/DPhi-Space/SimSat) by DPhi Space*
