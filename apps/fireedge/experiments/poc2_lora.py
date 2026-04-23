"""
/poc2 Step 2 — 小規模 LoRA Fine-tuning
===========================================
LFM 2.5-VL-450M に 10〜20 サンプルで LoRA FT を行い、
few-shot ICL (Recall=0.00) との定量比較を行う。

データ設計:
  POS: 火災座標を FIRMS 検知日 +SHIFT_DAYS で SimSat に問い合わせ (Δ≥0 のみ採用)
  NEG: 同一座標を FIRMS 検知日 -NEG_OFFSET_DAYS で問い合わせ (180日前 = 火災シーズン外)
  プロンプト: 画像のみ (スペクトル指標・ルール記述なし)

完了条件 (CLAUDE.md /poc2 ②):
  LoRA FT の Recall が ICL baseline (0.00) を上回ること

実行:
    cd apps/fireedge
    uv run python experiments/poc2_lora.py
    uv run python experiments/poc2_lora.py --train 16 --test 8 --epochs 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))  # apps/fireedge/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

# .env をモジュールレベルで先にロードする
_env_path = os.path.normpath(os.path.join(ROOT_DIR, "../../.env"))
if os.path.exists(_env_path):
    for _line in open(_env_path):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# poc2_icl.py からデータ収集ロジックを再利用
from poc2_icl import (
    AREAS,
    FIRMS_KEY,
    NEG_OFFSET_DAYS,
    SHIFT_DAYS,
    SYSTEM_PROMPT,
    FIRMSEvent,
    Sample,
    collect_samples,
    fetch_firms,
    print_metrics,
    run_inference,
    select_top,
)
from src.data_fetcher import SimSatClient
from src.spectral import SpectralProcessor

# ===========================================================================
# 設定
# ===========================================================================

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

# ===========================================================================
# 汎化確認用: FIRMS 検知地点と無関係な多様な非火災地点
# ===========================================================================
DIVERSE_NEG_LOCATIONS = [
    # 森林 (Forest)
    {"desc": "Germany Black Forest (temperate forest)", "lat": 48.0,  "lon":   8.2,  "ts": "2026-04-15T10:00:00Z"},
    {"desc": "Canada boreal forest (Manitoba)",         "lat": 55.0,  "lon":-100.0,  "ts": "2026-04-15T18:00:00Z"},
    {"desc": "Brazil deep Amazon (no fire history)",    "lat": -2.0,  "lon": -62.0,  "ts": "2026-04-15T15:00:00Z"},
    # 農地・草地 (Agricultural / Grassland)
    {"desc": "France agricultural land",                "lat": 45.0,  "lon":   2.0,  "ts": "2026-04-15T10:00:00Z"},
    {"desc": "Japan rice paddies (Aichi)",              "lat": 35.0,  "lon": 137.0,  "ts": "2026-04-15T02:00:00Z"},
    {"desc": "Ireland grassland",                       "lat": 53.0,  "lon":  -8.0,  "ts": "2026-04-15T11:00:00Z"},
    # 砂漠・乾燥地 (Desert / Arid)
    {"desc": "Sahara Desert (Algeria)",                 "lat": 23.0,  "lon":   5.0,  "ts": "2026-04-15T10:00:00Z"},
    {"desc": "Arabian Peninsula (Saudi Arabia)",        "lat": 24.0,  "lon":  45.0,  "ts": "2026-04-15T08:00:00Z"},
    {"desc": "Australian outback (South Australia)",    "lat":-25.0,  "lon": 135.0,  "ts": "2026-04-15T02:00:00Z"},
    # 都市・郊外 (Urban / Suburban)
    {"desc": "Tokyo suburban area",                     "lat": 35.7,  "lon": 139.7,  "ts": "2026-04-15T02:00:00Z"},
    {"desc": "London suburbs",                          "lat": 51.5,  "lon":  -0.1,  "ts": "2026-04-15T11:00:00Z"},
    # 湿地・デルタ (Wetland / Delta)
    {"desc": "Bangladesh Ganges delta (wetland)",       "lat": 22.5,  "lon":  90.5,  "ts": "2026-04-15T05:00:00Z"},
    # サバンナ非火災期 (Savanna, non-fire season)
    {"desc": "Kenya savanna (rainy season, no fire)",   "lat":  1.0,  "lon":  37.0,  "ts": "2026-04-15T09:00:00Z"},
    # 海洋 (Ocean) ※ S2はopen oceanを系統的に撮像しないためスキップの可能性あり
    {"desc": "North Pacific Ocean",                     "lat": 35.0,  "lon": 160.0,  "ts": "2026-04-15T02:00:00Z"},
    {"desc": "Indian Ocean",                            "lat":-10.0,  "lon":  75.0,  "ts": "2026-04-15T07:00:00Z"},
    {"desc": "North Atlantic Ocean",                    "lat": 45.0,  "lon": -30.0,  "ts": "2026-04-15T12:00:00Z"},
]

TRAIN_PROMPT = """\
Examine this satellite false-color composite image (R=SWIR2.2μm, G=SWIR1.6μm, B=NIR).

Does this scene contain active fire or burn scar?
Respond with JSON only: {"fire_detected": true} or {"fire_detected": false}\
"""

# ===========================================================================
# 学習データ作成
# ===========================================================================

def make_training_examples(
    samples: list[Sample],
    processor: AutoProcessor,
    device: str,
) -> list[dict]:
    """
    各 Sample を chat 形式にトークナイズし、
    アシスタント部分のみ loss を計算するための labels を付与する。
    inputs の全キーをそのまま保持して model(**inputs) で渡す。
    """
    import json

    examples = []
    for s in samples:
        response = json.dumps({"fire_detected": bool(s.label)})

        messages_full = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": s.image},
                    {"type": "text", "text": TRAIN_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        # 全シーケンスをトークナイズ (アシスタント応答を含む)
        inputs = processor.apply_chat_template(
            messages_full,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # アシスタントターン開始位置を取得 (system + user のみ)
        prefix_inputs = processor.apply_chat_template(
            messages_full[:-1],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        prefix_len = prefix_inputs["input_ids"].shape[1]

        # labels: prefix 部分を -100 でマスクし、アシスタント部分のみ loss 計算
        labels = inputs["input_ids"].clone()
        labels[:, :prefix_len] = -100

        # inputs の全テンソルを device に移動し labels を追加
        ex = {}
        for k, v in inputs.items():
            ex[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        ex["labels"] = labels.to(device)
        examples.append(ex)

    return examples


def train_lora(
    model,
    processor: AutoProcessor,
    train_samples: list[Sample],
    epochs: int,
    lr: float,
    device: str,
) -> None:
    """LoRA FT のトレーニングループ。"""
    print(f"\n[LoRA] 学習開始: {len(train_samples)} サンプル × {epochs} epochs  lr={lr}")

    examples = make_training_examples(train_samples, processor, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        indices = list(range(len(examples)))
        np.random.shuffle(indices)

        for idx in indices:
            ex = examples[idx]
            optimizer.zero_grad()
            outputs = model(**ex)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(examples)
        print(f"  Epoch {epoch}/{epochs}  loss={avg:.4f}")

    model.eval()
    print("  学習完了")


# ===========================================================================
# 評価 (ICL と共通ロジック)
# ===========================================================================

def run_lora_inference(
    model,
    processor: AutoProcessor,
    test_samples: list[Sample],
    device: str,
) -> list[tuple[Sample, bool, float, str]]:
    """zero-shot 推論 (few-shot context なし)。"""
    from poc2_icl import _parse_result

    results = []
    for i, sample in enumerate(test_samples, 1):
        print(f"  [{i}/{len(test_samples)}] {sample.desc} (GT={'FIRE' if sample.label else 'NO-FIRE'})", end=" ... ")
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample.image},
                    {"type": "text", "text": TRAIN_PROMPT},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )

        input_len = inputs["input_ids"].shape[1]
        raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        fire_detected, fire_confidence = _parse_result(raw)
        ms = (time.perf_counter() - t0) * 1000
        print(f"Pred={'FIRE' if fire_detected else 'NO-FIRE'} conf={fire_confidence:.2f} ({ms:.0f}ms)")
        results.append((sample, fire_detected, fire_confidence, raw))

    return results


def run_generalization_check(
    model,
    processor: AutoProcessor,
    client: SimSatClient,
    spectral: SpectralProcessor,
    device: str,
) -> None:
    """
    FIRMS 検知地点と無関係な多様な非火災地点で推論し FP Rate を確認する。
    全件 GT=NO-FIRE なので FP Rate のみが評価指標。
    """
    from poc2_icl import _parse_result

    print(f"\n{'='*60}")
    print(f"  汎化確認: FIRMS 非関連地点での FP Rate")
    print(f"{'='*60}")

    fp = 0
    valid = 0
    WINDOW_SEC = 12 * 86400

    for loc in DIVERSE_NEG_LOCATIONS:
        resp = client.fetch_fire_scene(
            lon=loc["lon"], lat=loc["lat"], timestamp=loc["ts"],
            size_km=5, window_seconds=WINDOW_SEC,
        )
        if not resp.image_available or resp.image_array is None:
            cc = f"{resp.cloud_cover:.0f}%" if resp.cloud_cover is not None else "?"
            print(f"  ⚠️  [{loc['desc']}] 画像なし (cc={cc}) → スキップ")
            continue

        arr = resp.image_array  # (H, W, 6): ch0=swir22, ch1=swir16, ch2=nir
        fire_composite = np.stack([arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]], axis=-1)
        img = spectral._to_pil(fire_composite)
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": TRAIN_PROMPT},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=128,
                temperature=0.3, do_sample=True, top_p=0.9,
            )

        input_len = inputs["input_ids"].shape[1]
        raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        fire_detected, conf = _parse_result(raw)
        ms = (time.perf_counter() - t0) * 1000

        mark = "❌ FP" if fire_detected else "✅ TN"
        print(f"  {mark}  [{loc['desc']}]  Pred={'FIRE' if fire_detected else 'NO-FIRE'} conf={conf:.2f} ({ms:.0f}ms)")
        valid += 1
        if fire_detected:
            fp += 1

    if valid == 0:
        print("  有効な画像が取得できませんでした。")
        return

    fp_rate = fp / valid
    print(f"\n  ────────────────────────────────────────")
    print(f"  有効サンプル: {valid}件  FP: {fp}件  FP Rate: {fp_rate:.2f}")
    goal = fp_rate <= 0.15
    print(f"  精度目標 (FP Rate ≤ 0.15): {'✅ 達成' if goal else '❌ 未達'}")
    print(f"{'='*60}\n")


# ===========================================================================
# メイン
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=16,
                        help="学習に使うペア数 (pos+neg 合計 = train*2)")
    parser.add_argument("--test",  type=int, default=8,
                        help="評価に使うペア数")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr",     type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--days",   type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    n_total_pairs = args.train + args.test

    if not FIRMS_KEY:
        print("ERROR: FIRMS_MAP_KEY / FIRMS_API_KEY が設定されていません")
        sys.exit(1)

    # ------------------------------------------------------------------
    # データ収集
    # ------------------------------------------------------------------
    print(f"\n[1] FIRMS データ取得 (過去{args.days}日, 上位{n_total_pairs}件)")
    all_events: list[FIRMSEvent] = []
    for name, area_str in AREAS.items():
        evs = fetch_firms(area_str, args.days)
        print(f"    {name}: {len(evs)} ホットスポット")
        all_events.extend(evs)

    if not all_events:
        print("FIRMS データなし")
        sys.exit(1)

    # Δ<0 スキップを見越して 3 倍バッファ
    top_events = select_top(all_events, n_total_pairs * 3)
    client   = SimSatClient()
    spectral = SpectralProcessor()

    print(f"\n[2] SimSat 画像取得 (POS: +{SHIFT_DAYS}d, NEG: -{NEG_OFFSET_DAYS}d, 同一地点)")
    pairs = collect_samples(top_events, client, spectral)

    if len(pairs) < args.train + 1:
        print(f"\nERROR: 有効ペアが {len(pairs)} 件のみ (学習{args.train}+テスト1以上が必要)")
        sys.exit(1)

    train_pairs = pairs[:args.train]
    test_pairs  = pairs[args.train:args.train + args.test]

    train_samples: list[Sample] = []
    for pos, neg in train_pairs:
        train_samples.append(pos)
        train_samples.append(neg)

    test_samples: list[Sample] = []
    for pos, neg in test_pairs:
        test_samples.append(pos)
        test_samples.append(neg)

    print(f"\n  学習: {len(train_samples)} サンプル (POS:{args.train} / NEG:{args.train})")
    print(f"  テスト: {len(test_samples)} サンプル")

    # スペクトル分布確認
    pos_nbr2 = [s.nbr2_min for s in train_samples + test_samples if s.label]
    neg_nbr2 = [s.nbr2_min for s in train_samples + test_samples if not s.label]
    print(f"\n  [スペクトル分布確認 (Δ≥0 フィルター後)]")
    print(f"  POS NBR2_min: min={min(pos_nbr2):.3f}  mean={np.mean(pos_nbr2):.3f}  max={max(pos_nbr2):.3f}")
    print(f"  NEG NBR2_min: min={min(neg_nbr2):.3f}  mean={np.mean(neg_nbr2):.3f}  max={max(neg_nbr2):.3f}")
    sep = np.mean(pos_nbr2) < np.mean(neg_nbr2) - 0.05
    print(f"  分離判定: {'✅ POS < NEG' if sep else '⚠️  分離不十分'}")

    # ------------------------------------------------------------------
    # モデルロード
    # ------------------------------------------------------------------
    print(f"\n[3] モデルロード: {MODEL_ID}")
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=args.device, trust_remote_code=True
    )

    # LoRA 設定
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  VRAM (FT前): {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ------------------------------------------------------------------
    # LoRA 学習前の評価 (zero-shot baseline)
    # ------------------------------------------------------------------
    print(f"\n[4] 学習前評価 (zero-shot baseline)")
    model.eval()
    results_before = run_lora_inference(model, processor, test_samples, args.device)

    print(f"\n  === 学習前 (zero-shot) ===")
    print_metrics(results_before)

    # ------------------------------------------------------------------
    # LoRA 学習
    # ------------------------------------------------------------------
    train_lora(model, processor, train_samples, args.epochs, args.lr, args.device)
    print(f"  VRAM (FT後): {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ------------------------------------------------------------------
    # LoRA 学習後の評価
    # ------------------------------------------------------------------
    print(f"\n[5] 学習後評価")
    results_after = run_lora_inference(model, processor, test_samples, args.device)

    print(f"\n  === 学習後 (LoRA FT) ===")
    print_metrics(results_after)

    # ------------------------------------------------------------------
    # 比較サマリ
    # ------------------------------------------------------------------
    def metrics(results):
        tp = sum(1 for s, p, _, __ in results if s.label and p)
        fp = sum(1 for s, p, _, __ in results if not s.label and p)
        tn = sum(1 for s, p, _, __ in results if not s.label and not p)
        fn = sum(1 for s, p, _, __ in results if s.label and not p)
        recall   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fp_rate  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return recall, fp_rate

    r_before, fp_before = metrics(results_before)
    r_after,  fp_after  = metrics(results_after)

    print(f"\n{'='*60}")
    print(f"  /poc2 Step2 比較サマリ")
    print(f"{'='*60}")
    print(f"  ICL baseline (前回): Recall=0.00  FP Rate=0.00")
    print(f"  Zero-shot (FT前)   : Recall={r_before:.2f}  FP Rate={fp_before:.2f}")
    print(f"  LoRA FT (FT後)     : Recall={r_after:.2f}  FP Rate={fp_after:.2f}")

    # 完了条件: ICL baseline (Recall=0.00) を上回ること
    # zero-shot (FT前) は全件FIRE予測の退化解 (Recall=1.00, FP=1.00) になり得るため
    # 比較対象は ICL baseline とする
    ICL_BASELINE_RECALL = 0.00
    improved = r_after > ICL_BASELINE_RECALL and fp_after <= 0.5
    print(f"\n  → {'✅ GO — LoRA FT が ICL baseline を上回った。/poc2 完了条件② 達成。' if improved else '❌ 改善なし — エポック数・学習率・データ数を見直す。'}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 汎化確認: FIRMS 非関連地点での FP Rate
    # ------------------------------------------------------------------
    print(f"\n[6] 汎化確認 (FIRMS 検知地点と無関係な多様な非火災地点)")
    run_generalization_check(model, processor, client, spectral, args.device)


if __name__ == "__main__":
    main()
