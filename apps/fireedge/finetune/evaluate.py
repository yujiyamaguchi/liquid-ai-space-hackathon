"""
FireEdge Base vs Fine-Tuned 精度比較
======================================
held-out test set と generalization test set (DIVERSE_NEG) で評価する。

評価の二層構造:
  Layer 1 - Test set (in-distribution):
    FIRMS地点由来の POS + temporal NEG + 一部 diverse NEG
    → Recall / Precision / F1 / FP Rate を base vs fine-tuned で比較

  Layer 2 - Generalization test (out-of-distribution):
    DIVERSE_NEG_LOCATIONS 16地点 (訓練データとは独立)
    → FP Rate のみ計測 (全件 GT=NO-FIRE)

使い方:
    cd apps/fireedge
    uv run python -m finetune.evaluate
    uv run python -m finetune.evaluate --adapter output/fireedge-lora/adapter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.interfaces import FIRE_DETECTION_SYSTEM_PROMPT, FIRE_DETECTION_USER_PROMPT

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
EVAL_OUT = ROOT / "data" / "build" / "eval"

# generalization test 用地点 (dataset_builder.py と同一リスト)
from finetune.dataset_builder import DIVERSE_NEG_LOCATIONS


# ===========================================================================
# 推論ユーティリティ
# ===========================================================================

def _extract_json(text: str) -> dict | None:
    import re
    for pattern in [r'\{[^{}]*\}', r'\{.*\}']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None


def _build_messages(image: Image.Image, nbr2: float = 0.0, ndvi: float = 0.0,
                    bai: float = 0.0, mean_swir22: float = 0.0,
                    fire_pixel_ratio: float = 0.0) -> list[dict]:
    user_text = FIRE_DETECTION_USER_PROMPT.format(
        nbr2=nbr2, ndvi=ndvi, bai=bai,
        mean_swir22=mean_swir22, fire_pixel_ratio=fire_pixel_ratio,
    )
    return [
        {"role": "system",  "content": [{"type": "text",  "text": FIRE_DETECTION_SYSTEM_PROMPT}]},
        {"role": "user",    "content": [{"type": "image", "image": image},
                                        {"type": "text",  "text": user_text}]},
    ]


def run_inference(model, processor, image: Image.Image, device: str = "cuda",
                  nbr2: float = 0.0, ndvi: float = 0.0, bai: float = 0.0,
                  mean_swir22: float = 0.0, fire_pixel_ratio: float = 0.0) -> dict:
    messages = _build_messages(image, nbr2, ndvi, bai, mean_swir22, fire_pixel_ratio)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    return _extract_json(raw) or {}


# ===========================================================================
# Layer 1: Test set 評価
# ===========================================================================

def evaluate_on_test(model, processor, test_ds, label: str,
                     device: str = "cuda") -> dict:
    """held-out test set で Recall/Precision/F1/FP Rate を計測。"""
    model.eval()
    preds, confs, gt_labels = [], [], []
    latencies = []
    json_ok = 0

    for i, ex in enumerate(test_ds):
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        t0 = time.perf_counter()
        result = run_inference(
            model, processor, img, device,
            nbr2=float(ex.get("nbr2", 0)),
            mean_swir22=float(ex.get("mean_swir22", 0)),
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        fire_pred  = bool(result.get("fire_detected", False))
        confidence = float(result.get("fire_confidence", 0.5))
        preds.append(int(fire_pred))
        confs.append(confidence)
        gt_labels.append(int(ex["label"]))
        if result:
            json_ok += 1

        if (i + 1) % 5 == 0:
            print(f"  [{label}] {i+1}/{len(test_ds)} ...", flush=True)

    preds_arr = np.array(preds)
    gt_arr    = np.array(gt_labels)

    tp = int(((preds_arr == 1) & (gt_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (gt_arr == 0)).sum())
    tn = int(((preds_arr == 0) & (gt_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (gt_arr == 1)).sum())
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "label":     label,
        "n":         len(gt_arr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision_score(gt_arr, preds_arr, zero_division=0),
        "recall":    recall_score(gt_arr, preds_arr, zero_division=0),
        "f1":        f1_score(gt_arr, preds_arr, zero_division=0),
        "fp_rate":   fp_rate,
        "accuracy":  float((preds_arr == gt_arr).mean()),
        "json_rate": json_ok / len(gt_arr),
        "lat_mean":  float(np.mean(latencies)),
        "lat_p95":   float(np.percentile(latencies, 95)),
        "preds":     preds_arr.tolist(),
        "confs":     np.array(confs).tolist(),
        "gt":        gt_arr.tolist(),
        "cm":        confusion_matrix(gt_arr, preds_arr).tolist(),
    }


# ===========================================================================
# Layer 2: Generalization test (FP Rate on out-of-distribution NEG)
# ===========================================================================

def evaluate_generalization(model, processor, device: str = "cuda") -> dict:
    """
    DIVERSE_NEG_LOCATIONS 16地点で FP Rate を計測。
    全件 GT=NO-FIRE なので FP Rate のみが評価指標。
    訓練データとは完全に独立。
    """
    import os
    from src.data_fetcher import SimSatClient
    from src.spectral import SpectralProcessor

    # .env ロード
    _env = ROOT / ".." / ".." / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    client   = SimSatClient()
    spectral = SpectralProcessor()
    WINDOW_SEC = 12 * 86400

    results = []
    for loc in DIVERSE_NEG_LOCATIONS:
        try:
            resp = client.fetch_fire_scene(
                lon=loc["lon"], lat=loc["lat"], timestamp=loc["ts"],
                size_km=5, window_seconds=WINDOW_SEC,
            )
        except Exception as e:
            results.append({"desc": loc["desc"], "status": "error", "pred": None})
            print(f"  ⚠️  [{loc['desc']}] error: {e}")
            continue

        if not resp.image_available or resp.image_array is None:
            results.append({"desc": loc["desc"], "status": "no_image", "pred": None})
            cc = f"{resp.cloud_cover:.0f}%" if resp.cloud_cover is not None else "?"
            print(f"  ⚠️  [{loc['desc']}] 画像なし (cc={cc})")
            continue

        scene = spectral.process(resp)
        t0 = time.perf_counter()
        result = run_inference(
            model, processor, scene.fire_composite, device,
            nbr2=float(scene.indices.nbr2),
            mean_swir22=float(scene.indices.mean_swir22),
        )
        ms = (time.perf_counter() - t0) * 1000

        fire_pred = bool(result.get("fire_detected", False))
        mark = "❌ FP" if fire_pred else "✅ TN"
        print(f"  {mark}  [{loc['desc']}]  pred={'FIRE' if fire_pred else 'NO-FIRE'} ({ms:.0f}ms)")
        results.append({"desc": loc["desc"], "status": "ok", "pred": fire_pred})

    valid = [r for r in results if r["status"] == "ok"]
    fp    = sum(1 for r in valid if r["pred"])
    fp_rate = fp / len(valid) if valid else 0.0

    return {
        "n_total":  len(DIVERSE_NEG_LOCATIONS),
        "n_valid":  len(valid),
        "n_fp":     fp,
        "fp_rate":  fp_rate,
        "details":  results,
    }


# ===========================================================================
# 出力・可視化
# ===========================================================================

def plot_comparison(base: dict, ft: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("FireEdge: Base LFM2.5-VL vs Fine-Tuned (Test Set)", fontsize=14, fontweight="bold")

    # Panel 1: メトリクスバーチャート
    ax = axes[0]
    metrics   = ["precision", "recall", "f1", "fp_rate", "accuracy"]
    labels    = ["Precision", "Recall", "F1", "FP Rate", "Accuracy"]
    x = np.arange(len(metrics))
    w = 0.35
    base_vals = [base[m] for m in metrics]
    ft_vals   = [ft[m]   for m in metrics]
    bars1 = ax.bar(x - w/2, base_vals, w, label="Base LFM2.5-VL", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + w/2, ft_vals,   w, label="FireEdge LoRA",  color="#e74c3c", alpha=0.85)
    # FP Rate 目標線
    ax.axhline(0.15, color="orange", lw=1.5, ls="--", label="FP Rate target (0.15)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Classification Metrics (Test Set)")
    ax.legend(fontsize=8)
    for bar, val in zip(list(bars1) + list(bars2), base_vals + ft_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Panel 2: 混同行列 (fine-tuned)
    ax2 = axes[1]
    cm = np.array(ft["cm"])
    im = ax2.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax2.set_title("Confusion Matrix (FireEdge LoRA)")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["NO-FIRE", "FIRE"]); ax2.set_yticklabels(["NO-FIRE", "FIRE"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax2)

    # Panel 3: サマリテーブル
    ax3 = axes[2]
    ax3.axis("off")

    def fmt_delta(f, b):
        d = f - b
        return f"+{d:.3f}" if d >= 0 else f"{d:.3f}"

    rows = [
        ["Metric",      "Base",                     "LoRA",                    "Δ"],
        ["Precision",   f"{base['precision']:.3f}",  f"{ft['precision']:.3f}",  fmt_delta(ft['precision'],  base['precision'])],
        ["Recall",      f"{base['recall']:.3f}",     f"{ft['recall']:.3f}",     fmt_delta(ft['recall'],     base['recall'])],
        ["F1",          f"{base['f1']:.3f}",          f"{ft['f1']:.3f}",          fmt_delta(ft['f1'],          base['f1'])],
        ["FP Rate",     f"{base['fp_rate']:.3f}",    f"{ft['fp_rate']:.3f}",    fmt_delta(ft['fp_rate'],    base['fp_rate'])],
        ["Accuracy",    f"{base['accuracy']:.3f}",   f"{ft['accuracy']:.3f}",   fmt_delta(ft['accuracy'],   base['accuracy'])],
        ["Latency(ms)", f"{base['lat_mean']:.0f}",   f"{ft['lat_mean']:.0f}",   fmt_delta(ft['lat_mean'],   base['lat_mean'])],
    ]
    tbl = ax3.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="center", loc="center",
                    colWidths=[0.28, 0.24, 0.24, 0.24])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for j in range(4):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Δ列: FP Rate は小さいほど良い (逆向き)
    fp_rate_row_idx = 4  # 1-based row index in table (header=0, Precision=1, ...)
    for i in range(1, len(rows)):
        d_str = rows[i][3]
        is_fp_rate = rows[i][0] == "FP Rate"
        better = d_str.startswith("+") and not is_fp_rate or d_str.startswith("-") and is_fp_rate
        color = "#27ae60" if better else "#e74c3c"
        tbl[i, 3].set_text_props(color=color, fontweight="bold")
    ax3.set_title("Performance Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    fig_path = out_dir / "base_vs_finetuned.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Eval] 比較図を保存: {fig_path}")
    return fig_path


def print_report(base: dict, ft: dict,
                 gen_base: dict | None = None, gen_ft: dict | None = None):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — Test Set (in-distribution)")
    print("=" * 60)
    hdr = f"{'Metric':<14} {'Base':>10} {'LoRA':>10} {'Delta':>10}"
    print(hdr); print("-" * 46)

    rows = [
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1"),
        ("fp_rate",   "FP Rate"),
        ("accuracy",  "Accuracy"),
        ("lat_mean",  "Latency(ms)"),
        ("lat_p95",   "Lat P95(ms)"),
    ]
    for key, name in rows:
        b, f = base[key], ft[key]
        sign = "+" if f >= b else ""
        if "lat" in key:
            print(f"{name:<14} {b:>10.0f} {f:>10.0f} {sign}{f-b:>9.0f}")
        else:
            print(f"{name:<14} {b:>10.3f} {f:>10.3f} {sign}{f-b:>9.3f}")

    print(f"\n  Confusion matrix (LoRA): TN={ft['tn']} FP={ft['fp']} FN={ft['fn']} TP={ft['tp']}")

    # 目標達成チェック
    targets = {"recall": (0.85, True), "fp_rate": (0.15, False)}
    print("\n  目標達成チェック:")
    for key, (thr, higher_better) in targets.items():
        val = ft[key]
        ok  = val >= thr if higher_better else val <= thr
        sym = "✅" if ok else "❌"
        print(f"  {sym}  {key}: {val:.3f} (目標={'≥' if higher_better else '≤'}{thr})")

    print("=" * 60)

    if gen_base and gen_ft:
        print("\n" + "=" * 60)
        print("GENERALIZATION TEST (out-of-distribution FP Rate)")
        print("=" * 60)
        print(f"  {'':30s} {'Base':>10} {'LoRA':>10}")
        print("-" * 52)
        b_rate = gen_base["fp_rate"]
        f_rate = gen_ft["fp_rate"]
        sign = "+" if f_rate >= b_rate else ""
        print(f"  {'FP Rate':<30} {b_rate:>10.3f} {f_rate:>10.3f}  {sign}{f_rate-b_rate:.3f}")
        print(f"  Valid samples: {gen_ft['n_valid']}/{gen_ft['n_total']}")
        ok = f_rate <= 0.30
        print(f"\n  {'✅' if ok else '❌'}  Generalization FP Rate: {f_rate:.2f} (目標 ≤0.30, poc2実績=0.57)")
        print("=" * 60)


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",      type=str,
                   default=str(ROOT / "output" / "fireedge-lora" / "adapter"),
                   help="LoRA adapter ディレクトリ")
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--skip-gen",     action="store_true",
                   help="generalization test をスキップ")
    p.add_argument("--skip-base",    action="store_true",
                   help="base model 評価をスキップ (adapter のみ評価)")
    args = p.parse_args()

    ds_path = ROOT / "data" / "build" / "hf_dataset"
    if not ds_path.exists():
        print(f"[ERROR] dataset が見つかりません: {ds_path}")
        sys.exit(1)

    test_ds = load_from_disk(str(ds_path / "test"))
    print(f"[Eval] test set: {len(test_ds)} 件 "
          f"(fire={sum(test_ds['label'])}, no-fire={len(test_ds)-sum(test_ds['label'])})")

    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    device = args.device

    # --- Base model ---
    base_results = None
    gen_base     = None
    if not args.skip_base:
        print("\n[Eval] Base model をロード中 ...")
        processor  = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        )
        print("[Eval] Base model — Test set 評価中 ...")
        base_results = evaluate_on_test(base_model, processor, test_ds, "Base LFM2.5-VL", device)

        if not args.skip_gen:
            print("\n[Eval] Base model — Generalization test ...")
            gen_base = evaluate_generalization(base_model, processor, device)

        del base_model
        torch.cuda.empty_cache()
    else:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # --- Fine-tuned model ---
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"[ERROR] adapter が見つかりません: {adapter_path}")
        print("  先に `uv run python -m finetune.train` を実行してください。")
        sys.exit(1)

    from peft import PeftModel
    print("\n[Eval] Fine-tuned model をロード中 ...")
    ft_base  = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_base, str(adapter_path))
    ft_model = ft_model.merge_and_unload()

    print("[Eval] Fine-tuned model — Test set 評価中 ...")
    ft_results = evaluate_on_test(ft_model, processor, test_ds, "FireEdge LoRA", device)

    gen_ft = None
    if not args.skip_gen:
        print("\n[Eval] Fine-tuned model — Generalization test ...")
        gen_ft = evaluate_generalization(ft_model, processor, device)

    del ft_model
    torch.cuda.empty_cache()

    # --- Report ---
    if base_results:
        print_report(base_results, ft_results, gen_base, gen_ft)
        plot_comparison(base_results, ft_results, EVAL_OUT)
    else:
        print("\n[Eval] Fine-tuned only:")
        for key in ["precision", "recall", "f1", "fp_rate", "accuracy"]:
            print(f"  {key}: {ft_results[key]:.3f}")
        if gen_ft:
            print(f"  Generalization FP Rate: {gen_ft['fp_rate']:.3f}")

    # --- JSON 保存 ---
    results_path = EVAL_OUT / "results.json"
    with open(results_path, "w") as fh:
        json.dump({
            "base":        base_results,
            "finetuned":   ft_results,
            "gen_base":    gen_base,
            "gen_ft":      gen_ft,
        }, fh, indent=2)
    print(f"\n[Eval] 結果を保存: {results_path}")


if __name__ == "__main__":
    main()
