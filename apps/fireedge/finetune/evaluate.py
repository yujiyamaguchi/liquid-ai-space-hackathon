"""
FireEdge Base vs Fine-Tuned 精度比較
======================================
validation セット上で base モデルと LoRA fine-tuned モデルを比較し、
hackathon 提出用の精度テーブルと図を生成する。

使い方:
    uv run python -m finetune.evaluate
    uv run python -m finetune.evaluate --adapter output/fireedge-lora/adapter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_ID   = "LiquidAI/LFM2.5-VL-450M"
EVAL_OUT   = ROOT / "data" / "finetune" / "eval"


# ------------------------------------------------------------------ inference

def _extract_json(text: str) -> dict | None:
    """生テキストから JSON を抽出。"""
    import re
    for pattern in [r'\{[^{}]*\}', r'\{.*\}']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None


def run_inference(model, processor, image: Image.Image, device: str = "cuda") -> dict:
    """1枚の画像に対して推論を実行し、パース済み JSON dict を返す。"""
    from interfaces import FIRE_DETECTION_SYSTEM_PROMPT, FIRE_DETECTION_USER_PROMPT

    messages = [
        {"role": "system",  "content": [{"type": "text",  "text": FIRE_DETECTION_SYSTEM_PROMPT}]},
        {"role": "user",    "content": [{"type": "image"}, {"type": "text", "text": FIRE_DETECTION_USER_PROMPT}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # 入力部分を除いた生成トークンのみデコード
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    parsed = _extract_json(raw)
    return parsed or {}


# ------------------------------------------------------------------ eval loop

def evaluate_model(model, processor, val_ds, label: str, device: str = "cuda") -> dict:
    """validation set 上で推論し、各種メトリクスを返す。"""
    model.eval()
    preds, confs, gt_labels = [], [], []
    json_ok, latencies = 0, []

    for i, ex in enumerate(val_ds):
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img["bytes"] if isinstance(img, dict) else img)

        t0 = time.perf_counter()
        result = run_inference(model, processor, img, device)
        latencies.append((time.perf_counter() - t0) * 1000)

        fire_pred = bool(result.get("fire_detected", False))
        confidence = float(result.get("fire_confidence", 0.5))
        preds.append(int(fire_pred))
        confs.append(confidence)
        gt_labels.append(int(ex["label"]))

        if result:
            json_ok += 1

        if (i + 1) % 5 == 0:
            print(f"  [{label}] {i+1}/{len(val_ds)} done ...", flush=True)

    preds = np.array(preds)
    gt    = np.array(gt_labels)
    confs = np.array(confs)

    return {
        "label":     label,
        "n":         len(gt),
        "precision": precision_score(gt, preds, zero_division=0),
        "recall":    recall_score(gt, preds, zero_division=0),
        "f1":        f1_score(gt, preds, zero_division=0),
        "accuracy":  float((preds == gt).mean()),
        "json_rate": json_ok / len(gt),
        "lat_mean":  float(np.mean(latencies)),
        "lat_std":   float(np.std(latencies)),
        "preds":     preds.tolist(),
        "confs":     confs.tolist(),
        "gt":        gt.tolist(),
        "cm":        confusion_matrix(gt, preds).tolist(),
    }


# ------------------------------------------------------------------ plotting

def plot_comparison(base: dict, ft: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("FireEdge: Base LFM2.5-VL vs Fine-Tuned Comparison", fontsize=14, fontweight="bold")

    # Panel 1: メトリクスバーチャート
    ax = axes[0]
    metrics = ["precision", "recall", "f1", "accuracy", "json_rate"]
    x = np.arange(len(metrics))
    w = 0.35
    base_vals = [base[m] for m in metrics]
    ft_vals   = [ft[m]   for m in metrics]
    bars1 = ax.bar(x - w/2, base_vals, w, label="Base LFM2.5-VL", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + w/2, ft_vals,   w, label="FireEdge LoRA",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1", "Accuracy", "JSON%"], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics")
    ax.legend()
    ax.axhline(1.0, color="gray", lw=0.5, ls="--")
    for bar, val in zip(list(bars1) + list(bars2), base_vals + ft_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Panel 2: 信頼度キャリブレーション (Reliability Diagram)
    ax2 = axes[1]
    for res, color, lbl in [(base, "#3498db", "Base"), (ft, "#e74c3c", "LoRA")]:
        confs = np.array(res["confs"])
        gt    = np.array(res["gt"])
        bins  = np.linspace(0, 1, 11)
        bin_acc, bin_conf, bin_n = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() > 0:
                bin_acc.append(gt[mask].mean())
                bin_conf.append(confs[mask].mean())
                bin_n.append(mask.sum())
        ax2.plot(bin_conf, bin_acc, "o-", color=color, label=lbl, markersize=6)
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax2.set_xlabel("Mean Confidence")
    ax2.set_ylabel("Fraction Positive")
    ax2.set_title("Reliability Diagram\n(Confidence Calibration)")
    ax2.legend()
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    # Panel 3: サマリテーブル
    ax3 = axes[2]
    ax3.axis("off")
    delta_p  = ft["precision"] - base["precision"]
    delta_r  = ft["recall"]    - base["recall"]
    delta_f1 = ft["f1"]        - base["f1"]
    delta_j  = ft["json_rate"] - base["json_rate"]
    delta_l  = ft["lat_mean"]  - base["lat_mean"]

    def fmt(v): return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

    table_data = [
        ["Metric",      "Base",                   "LoRA",               "Δ"],
        ["Precision",   f"{base['precision']:.3f}", f"{ft['precision']:.3f}", fmt(delta_p)],
        ["Recall",      f"{base['recall']:.3f}",    f"{ft['recall']:.3f}",    fmt(delta_r)],
        ["F1",          f"{base['f1']:.3f}",         f"{ft['f1']:.3f}",         fmt(delta_f1)],
        ["Accuracy",    f"{base['accuracy']:.3f}",  f"{ft['accuracy']:.3f}",  fmt(ft['accuracy']-base['accuracy'])],
        ["JSON Parse%", f"{base['json_rate']:.1%}",  f"{ft['json_rate']:.1%}",  fmt(delta_j)],
        ["Latency(ms)", f"{base['lat_mean']:.0f}",   f"{ft['lat_mean']:.0f}",   fmt(delta_l)],
    ]
    tbl = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc="center", loc="center",
                    colWidths=[0.28, 0.24, 0.24, 0.24])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    # ヘッダー行を強調
    for j in range(4):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Delta 列を色付け
    for i in range(1, len(table_data)):
        val_str = table_data[i][3]
        color = "#27ae60" if val_str.startswith("+") else "#e74c3c"
        tbl[i, 3].set_text_props(color=color, fontweight="bold")
    ax3.set_title("Performance Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    fig_path = out_dir / "base_vs_finetuned.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Eval] 図を保存: {fig_path}")
    return fig_path


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=str,
                   default=str(ROOT / "output" / "fireedge-lora" / "adapter"),
                   help="LoRA adapter の保存先ディレクトリ")
    p.add_argument("--device",  type=str, default="cuda")
    args = p.parse_args()

    ds_path = ROOT / "data" / "finetune" / "hf_dataset"
    if not ds_path.exists():
        print(f"[ERROR] dataset が見つかりません: {ds_path}")
        sys.exit(1)

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"[ERROR] adapter が見つかりません: {adapter_path}")
        print("  先に `uv run python -m finetune.train` を実行してください。")
        sys.exit(1)

    val_ds = load_from_disk(str(ds_path))["test"]
    print(f"[Eval] validation samples: {len(val_ds)} "
          f"(fire={sum(val_ds['label'])}, nofire={len(val_ds)-sum(val_ds['label'])})")

    device = args.device

    # --- Base model ---
    print("\n[Eval] Base model をロード中 ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    print("[Eval] Base model 評価中 ...")
    base_results = evaluate_model(base_model, processor, val_ds, "Base LFM2.5-VL", device)
    del base_model
    torch.cuda.empty_cache()

    # --- Fine-tuned model ---
    print("\n[Eval] Fine-tuned model をロード中 ...")
    from peft import PeftModel
    ft_base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_base, str(adapter_path))
    ft_model = ft_model.merge_and_unload()  # 推論時は merge して速度変化なし
    print("[Eval] Fine-tuned model 評価中 ...")
    ft_results = evaluate_model(ft_model, processor, val_ds, "FireEdge LoRA", device)
    del ft_model
    torch.cuda.empty_cache()

    # --- Report ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    header = f"{'Metric':<14} {'Base':>10} {'LoRA':>10} {'Delta':>10}"
    print(header)
    print("-" * 46)
    for key, name in [("precision","Precision"), ("recall","Recall"),
                      ("f1","F1"), ("accuracy","Accuracy"),
                      ("json_rate","JSON Rate"), ("lat_mean","Latency(ms)")]:
        b, f = base_results[key], ft_results[key]
        sign = "+" if f >= b else ""
        if key == "lat_mean":
            print(f"{name:<14} {b:>10.0f} {f:>10.0f} {sign}{f-b:>9.0f}")
        else:
            print(f"{name:<14} {b:>10.3f} {f:>10.3f} {sign}{f-b:>9.3f}")
    print("=" * 60)

    # --- Plot ---
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    plot_comparison(base_results, ft_results, EVAL_OUT)

    # --- Save JSON ---
    results_path = EVAL_OUT / "results.json"
    with open(results_path, "w") as fh:
        json.dump({"base": base_results, "finetuned": ft_results}, fh, indent=2)
    print(f"[Eval] 結果を保存: {results_path}")


if __name__ == "__main__":
    main()
