# FireEdge 🔥🛰️
### Smoke-Penetrating Wildfire Detection on Satellite Edge

**LFM 2.5-VL-450M (LoRA fine-tuned) × Sentinel-2 SWIR | AI in Space Hackathon 2026**

---

## The Problem

Wildfires kill thousands and burn millions of hectares annually. Existing satellite detection relies on RGB imagery — but **smoke hides fire**. By the time a fire is visible from the ground or to RGB cameras, it may have already spread uncontrollably.

- **FIRMS/VIIRS**: 375 m resolution, 9–15 hour processing delay after detection
- **RGB cameras**: Completely blind under smoke cover
- **Ground observers**: Cannot reach vast, remote forest areas

## The Solution

> **What RGB can't see, SWIR can.**

FireEdge runs **entirely on-orbit**, analyzing Sentinel-2 SWIR bands (1.6 μm & 2.2 μm) that **penetrate smoke** to reveal hidden fire fronts. A LoRA fine-tuned 450 M-parameter LFM model fits comfortably in a satellite's edge compute budget.

```
Smoke/Cloud cover → RGB camera sees nothing
SWIR 2.2 μm      → Active fire glows bright red/orange through smoke
```

---

## Why Edge AI?

| | Ground-based AI | **FireEdge (On-orbit AI)** |
|---|---|---|
| Bandwidth | ~4 MB raw image downlink | **< 2 KB alert JSON** (~3000× smaller) |
| Latency | Hours (ground station contact) | **Seconds (on-orbit real-time)** |
| Smoke | RGB blind under smoke | **SWIR penetrates smoke** |
| Raw data | Full image leaves satellite | **Inference on-board, no raw downlink** |
| Model size | Unconstrained | **450 M params, ~900 MB VRAM** |

---

## Architecture

```
SimSat API (Sentinel-2)
        │
        ▼
  DataFetcher          ←  6 bands: SWIR22, SWIR16, NIR, R, G, B
        │
        ▼
  SpectralProcessor    ←  SWIR false-color composite + NBR2 / SWIR22_max indices
        │
        ▼
  LFM 2.5-VL-450M      ←  LoRA fine-tuned on 300 real Sentinel-2 fire scenes
        │
        ▼
  Alert JSON           ←  { "fire_detected": true/false } + scene metadata
```

---

## Spectral Science

The core insight combines **NBR2** (Normalized Burn Ratio 2) with SWIR 2.2 μm brightness:

```
NBR2 = (SWIR1.6 − SWIR2.2) / (SWIR1.6 + SWIR2.2)

Active fire:   NBR2 < −0.05  AND  SWIR2.2 > 0.15
Burn scar:     NBR2 < −0.05  (lower SWIR2.2 than active fire)
Healthy veg:   NBR2 > 0.30
```

The false-color composite:
- **Red channel** = SWIR 2.2 μm → active fire appears **bright red/orange**
- **Green channel** = SWIR 1.6 μm → thermal anomalies appear **green**
- **Blue channel** = NIR 865 nm → healthy vegetation appears **blue**

---

## Quick Start

### Prerequisites
- Docker (for SimSat)
- Python 3.10+ with `uv`
- NVIDIA GPU (RTX 5090 tested; ~900 MB VRAM required)
- Hugging Face account (to download LFM base model)

### 1. Start SimSat
```bash
cd simsat && docker compose up -d
# Health check:
curl http://localhost:9005/   # → {"message":"Simulation API is online"}
```

### 2. Install dependencies & authenticate
```bash
cd apps/fireedge
uv sync
huggingface-cli login         # needed to download LiquidAI/LFM2.5-VL-450M
```

### 3. Run the E2E demo
```bash
cd apps/fireedge

# Default: proven fire scene (Southeast Asia, 2025-03-25)
uv run python demo.py

# Custom scene
uv run python demo.py --lon 96.80 --lat 19.76 --timestamp 2025-03-22T04:13:27Z

# No-fire comparison
uv run python demo.py --no-fire

# Save SWIR composite image
uv run python demo.py --save-image scene.png
```

**Example output:**
```json
{
  "fire_detected": true,
  "scene": {
    "lon": 106.55, "lat": 15.997,
    "capture": "2025-03-25T03:33:47Z",
    "cloud_pct": 0.0,
    "source": "sentinel-2b"
  },
  "spectral_indices": { "nbr2_min": -0.183, "swir22_max": 0.094 },
  "performance": { "fetch_ms": 820, "infer_ms": 148, "total_ms": 14200 }
}
```

---

## Fine-Tuning Results

FireEdge LoRA-fine-tunes LFM 2.5-VL-450M on **300 real Sentinel-2 fire scenes** collected via SimSat, with NASA FIRMS VIIRS SP as ground truth.

### Dataset
| Split | Fire | No-fire | Total |
|---|---|---|---|
| Train | 70 | 140 | 210 |
| Val | 15 | 30 | 45 |
| Test | 15 | 30 | 45 |

Positive: FIRMS hotspot coordinates → SimSat SWIR scene (burn scar stable, +2 days offset)
Negative: same coordinates before fire (−180 days) + diverse global no-fire scenes

### Evaluation — Test Set (n=45, held-out)

| Metric | Base LFM2.5-VL | **FireEdge LoRA** | Δ |
|---|---|---|---|
| Precision | 0.333 | **1.000** | +0.667 |
| Recall | 1.000 | **0.933** | −0.067 |
| F1 | 0.500 | **0.966** | +0.466 |
| **FP Rate** | **1.000** | **0.000** | **−1.000** |
| Accuracy | 0.333 | **0.978** | +0.644 |
| Inference (ms) | 163 | **150** | −14 |

Confusion matrix (LoRA): **TN=30  FP=0  FN=1  TP=14**

Target thresholds (forest authority use case):
- ✅ Recall ≥ 0.85 → **0.933**
- ✅ FP Rate ≤ 0.15 → **0.000**

> The base model predicts *every* scene as fire (FP Rate = 1.0).
> LoRA fine-tuning eliminates false alarms entirely while maintaining high recall.

### LoRA Configuration
- Base model: `LiquidAI/LFM2.5-VL-450M`
- LoRA rank r=16, alpha=32, dropout=0.05
- Target modules: attention (q/k/v/out), FFN (w1/w2/w3), conv projection (in_proj), multimodal projector (linear_1/2)
- Epochs: 5, lr: 2e-4 (cosine), effective batch: 8
- Adapter weights: [🤗 HuggingFace — link TBD after upload]

### Reproduce Training
```bash
cd apps/fireedge

# 1. Collect dataset (SimSat + NASA FIRMS, ~1 hour)
uv run python -m finetune.dataset_builder

# 2. Train LoRA (RTX 5090, ~30 min)
uv run python -m finetune.train

# 3. Evaluate (base model comparison)
uv run python -m finetune.evaluate --run-base
```

---

## Performance

| Metric | Result | Target |
|---|---|---|
| Alert JSON size | **< 900 bytes** | < 2,048 bytes ✓ |
| Peak VRAM | **~900 MB** | < 2,048 MB ✓ |
| Inference time | **~150 ms** (model cached) | — |
| Compression vs raw image | **~4,000×** | — |
| Test set F1 (LoRA) | **0.966** | > 0.75 ✓ |
| Test set FP Rate (LoRA) | **0.000** | < 0.15 ✓ |

---

## Project Structure

```
fireedge/
├── demo.py                    # E2E demo: SimSat → SWIR → LFM → alert JSON
├── src/
│   ├── interfaces.py          # All type definitions & prompts
│   ├── data_fetcher.py        # SimSat API client (Sentinel-2 bands)
│   └── spectral.py            # SWIR false-color composite + spectral indices
├── finetune/
│   ├── dataset_builder.py     # FIRMS GT + SimSat → train/val/test JSONL
│   ├── train.py               # LoRA SFT training script
│   ├── evaluate.py            # Base vs fine-tuned evaluation + comparison chart
│   ├── config.py              # Hyperparameters
│   └── collator.py            # VLM batch collator
├── output/
│   └── fireedge-lora/
│       └── adapter/           # LoRA adapter weights (safetensors)
├── data/
│   └── finetune/
│       ├── dataset/           # 300 SWIR PNG scenes + records.jsonl
│       ├── hf_dataset/        # HuggingFace Dataset (train/val/test splits)
│       └── eval/              # results.json + base_vs_finetuned.png
├── docs/
│   ├── lean_canvas.md         # Business model & market analysis
│   ├── fire_detection_science.md
│   ├── poc_results.md
│   └── poc2_results.md
└── tests/
    └── test_spectral.py       # Spectral processing unit tests (6/6 passing)
```

---

## Hackathon Judging Alignment

| Criterion | Weight | FireEdge's Approach |
|---|---|---|
| **Satellite Imagery** | 10% | Sentinel-2 SWIR via SimSat (DPhi Space API) as core data source |
| **Innovation** | 35% | SWIR smoke-penetration + LoRA-fine-tuned LFM — combination impossible with either alone |
| **Technical Impl.** | 35% | E2E demo runs in one command; LoRA FT with 300 real scenes; quantitative improvement proven |
| **Demo & Presentation** | 20% | Architecture video covering problem → why now → why SWIR×LFM → what we proved |

---

*Built with [LFM 2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) by Liquid AI × [SimSat](https://github.com/DPhi-Space/SimSat) by DPhi Space*
