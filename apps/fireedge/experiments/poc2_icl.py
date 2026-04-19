"""
/poc2 Step 1 — Few-shot In-Context Learning 検証
=================================================
LFM 2.5-VL-450M に /poc サンプルを context として与え (few-shot ICL)、
新規シーンの火災 / 非火災判定精度を計測する。

結果 (2026-04-19):
  Recall=0.00, FP Rate=0.00 — モデルは画像のみでは burn scar を識別できない。
  これを ICL baseline として /poc2 Step 2 (LoRA FT) の比較対象とする。

実行:
    cd apps/fireedge
    uv run python experiments/poc2_icl.py
    uv run python experiments/poc2_icl.py --shots 4 --top 10 --days 5
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
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

from src.data_fetcher import SimSatClient
from src.spectral import SpectralProcessor

# ===========================================================================
# 設定
# ===========================================================================

MODEL_ID    = "LiquidAI/LFM2.5-VL-450M"
FIRMS_BASE  = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_KEY   = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("FIRMS_API_KEY", "")

AREAS = {
    "west_africa": "5,5,30,15",
    "east_africa": "25,0,45,15",
    "seasia":      "95,5,140,20",
    "amazon":      "-70,-15,-45,5",
}
SHIFT_DAYS      = 2        # POS クエリ: 火災検知の N 日後を基準に SimSat へ問い合わせ (小さくして直後バイアス)
NEG_OFFSET_DAYS = 180      # NEG クエリ: 火災検知の N 日前 (SE アジアは 180 日前 ≈ 火災シーズン外)
WINDOW_SEC      = 12 * 86400
SIZE_KM         = 5

# ===========================================================================
# プロンプトテンプレート
# ===========================================================================

SYSTEM_PROMPT = """\
You are an expert satellite image analyst specializing in wildfire detection \
using multispectral remote sensing data.

You are analyzing a false-color composite image where:
- RED channel   = SWIR 2.2μm (B12): Active fire appears BRIGHT RED/ORANGE; \
burn scars appear DARK RED/BROWN
- GREEN channel = SWIR 1.6μm (B11): Thermal anomalies appear GREEN
- BLUE channel  = NIR 842nm  (B08): Healthy vegetation appears BLUE/GREEN

This composite penetrates smoke, so fires hidden under smoke are visible \
as bright red/orange areas.

Respond ONLY with a valid JSON object. No explanation or markdown.\
"""

FEW_SHOT_USER_TMPL = """\
Examine this satellite false-color composite image (R=SWIR2.2μm, G=SWIR1.6μm, B=NIR).

Does this scene contain active fire or burn scar? Answer FIRE or NO-FIRE.\
"""

QUERY_USER_TMPL = """\
Examine this satellite false-color composite image (R=SWIR2.2μm, G=SWIR1.6μm, B=NIR).

Does this scene contain active fire or burn scar?
Respond with JSON:
{{"fire_detected": <true|false>, "fire_confidence": <0.0-1.0>, "description": <string max 60 chars>}}\
"""


# ===========================================================================
# データクラス
# ===========================================================================

@dataclass
class FIRMSEvent:
    lat: float
    lon: float
    frp: float
    confidence: str
    acq_datetime: datetime

    @property
    def iso(self) -> str:
        return self.acq_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Sample:
    """1シーン分のデータ"""
    label: bool              # True = fire, False = no-fire (ground truth)
    lat: float
    lon: float
    image: Optional[Image.Image]   # SWIR疑似カラー (448×448 RGB)
    nbr2: float              # シーン平均 NBR2
    nbr2_min: float          # シーン最小 NBR2 ← /poc で有効だった指標
    mean_swir22: float       # シーン平均 SWIR22
    swir22_max: float        # シーン最大 SWIR22 ← /poc で有効だった指標
    fire_pixel_ratio: float
    image_datetime: str
    delta_days: float
    desc: str                # デバッグ用ラベル


# ===========================================================================
# FIRMS
# ===========================================================================

def fetch_firms(area_str: str, days: int) -> list[FIRMSEvent]:
    url = f"{FIRMS_BASE}/{FIRMS_KEY}/VIIRS_SNPP_NRT/{area_str}/{days}"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  [FIRMS] ERROR {area_str}: {e}")
        return []

    events = []
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            date_str = row["acq_date"]
            time_str = str(row["acq_time"]).zfill(4)
            dt = datetime(
                int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]),
                int(time_str[:2]), int(time_str[2:]), 0, tzinfo=timezone.utc
            )
            events.append(FIRMSEvent(
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
                frp=float(row.get("frp") or 0),
                confidence=str(row.get("confidence", "n")).strip().lower(),
                acq_datetime=dt,
            ))
        except Exception:
            continue
    return events


def select_top(events: list[FIRMSEvent], n: int) -> list[FIRMSEvent]:
    high  = sorted([e for e in events if e.confidence == "h"], key=lambda e: e.frp, reverse=True)
    other = sorted([e for e in events if e.confidence != "h"], key=lambda e: e.frp, reverse=True)
    return (high + other)[:n]


# ===========================================================================
# SimSat → PIL 変換
# ===========================================================================

def fetch_sample(
    client: SimSatClient,
    spectral: SpectralProcessor,
    lat: float, lon: float,
    query_ts: str,
    label: bool,
    desc: str,
    ref_datetime: Optional[datetime] = None,
) -> Optional[Sample]:
    """SimSat から画像を取得し Sample を返す。取得不可なら None。"""
    try:
        resp = client.fetch_fire_scene(
            lon=lon, lat=lat,
            timestamp=query_ts,
            size_km=SIZE_KM,
            window_seconds=WINDOW_SEC,
        )
    except Exception as e:
        print(f"      [SimSat] ERROR ({desc}): {e}")
        return None

    if not resp.image_available or resp.image_array is None:
        print(f"      [SimSat] 画像なし ({desc}): cc={resp.cloud_cover}")
        return None

    # スペクトル指標
    arr = resp.image_array
    s22, s16 = arr[:, :, 0], arr[:, :, 1]
    nbr2_arr = (s16 - s22) / (s16 + s22 + 1e-10)
    nbr2 = float(nbr2_arr.mean())
    nbr2_min = float(nbr2_arr.min())
    mean_swir22 = float(s22.mean())
    swir22_max = float(s22.max())
    fire_pixel_ratio = float(np.mean((nbr2_arr < -0.05) & (s22 > 0.15)))

    # PIL 画像 (SWIR22=R, SWIR16=G, NIR=B)
    scene = spectral.process(resp)

    # タイムラグ計算
    delta_days = 999.0
    if ref_datetime and resp.datetime:
        try:
            s2_dt = datetime.strptime(resp.datetime[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            delta_days = (s2_dt - ref_datetime).total_seconds() / 86400
        except Exception:
            pass

    return Sample(
        label=label,
        lat=lat, lon=lon,
        image=scene.fire_composite,
        nbr2=nbr2,
        nbr2_min=nbr2_min,
        mean_swir22=mean_swir22,
        swir22_max=swir22_max,
        fire_pixel_ratio=fire_pixel_ratio,
        image_datetime=resp.datetime or "",
        delta_days=delta_days,
        desc=desc,
    )


# ===========================================================================
# データ収集
# ===========================================================================

def collect_samples(
    top_events: list[FIRMSEvent],
    client: SimSatClient,
    spectral: SpectralProcessor,
) -> list[tuple[Sample, Sample]]:
    """
    正例・負例のペアリストを返す。

    POS: 火災座標 (lat, lon) を火災検知日 + SHIFT_DAYS で SimSat に問い合わせ。
         Δ >= 0 (火災後撮像) のみ採用。
    NEG: 同一座標 (lat, lon) を火災検知日 - NEG_OFFSET_DAYS で問い合わせ。
         時間的に離れることで burn scar がない状態の同一地点を取得する。
    """
    pairs: list[tuple[Sample, Sample]] = []

    for i, event in enumerate(top_events, 1):
        pos_ts = (event.acq_datetime + timedelta(days=SHIFT_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        neg_ts = (event.acq_datetime - timedelta(days=NEG_OFFSET_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n  [Event {i}/{len(top_events)}] lat={event.lat:.3f} lon={event.lon:.3f} "
              f"FRP={event.frp:.0f}MW")
        print(f"    POS ts={pos_ts}  NEG ts={neg_ts} ({NEG_OFFSET_DAYS}d 前)")

        # 正例: 火災後の同一地点
        pos = fetch_sample(
            client, spectral,
            lat=event.lat, lon=event.lon,
            query_ts=pos_ts,
            label=True,
            desc=f"POS#{i} FRP={event.frp:.0f}MW",
            ref_datetime=event.acq_datetime,
        )
        if pos is None:
            continue

        # Δ >= 0 フィルター: 火災後に撮像されたことを確認
        if pos.delta_days < 0:
            print(f"    POS#{i}: skip — Δ={pos.delta_days:+.1f}d < 0 (火災前撮像)")
            continue

        # 負例: 同一地点・180 日前 (火災シーズン外)
        neg = fetch_sample(
            client, spectral,
            lat=event.lat, lon=event.lon,
            query_ts=neg_ts,
            label=False,
            desc=f"NEG#{i} (same loc, -{NEG_OFFSET_DAYS}d)",
            ref_datetime=None,  # delta_days は NEG には意味を持たない
        )
        if neg is None:
            print(f"    NEG#{i}: 画像なし → スキップ")
            continue

        print(f"    POS: Δ={pos.delta_days:+.1f}d  NBR2_min={pos.nbr2_min:.3f}  SWIR22_max={pos.swir22_max:.3f}")
        print(f"    NEG:              NBR2_min={neg.nbr2_min:.3f}  SWIR22_max={neg.swir22_max:.3f}  img={neg.image_datetime[:10]}")
        pairs.append((pos, neg))
        time.sleep(0.5)

    return pairs


# ===========================================================================
# LFM-VL Few-shot 推論
# ===========================================================================

def build_few_shot_messages(
    shot_samples: list[Sample],
) -> list[dict]:
    """
    few-shot の user/assistant ターンを構築する。
    shot_samples は positive と negative が交互に並ぶことが望ましい。
    """
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    ]

    for s in shot_samples:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": s.image},
                {"type": "text",  "text":  FEW_SHOT_USER_TMPL},
            ],
        })

        # 画像ラベルのみ (指標値は含めない)
        if s.label:
            asst = json.dumps({
                "fire_detected": True,
                "fire_confidence": 0.90,
                "description": "Fire or burn scar detected in this scene.",
            })
        else:
            asst = json.dumps({
                "fire_detected": False,
                "fire_confidence": 0.90,
                "description": "No fire or burn scar detected in this scene.",
            })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": asst}],
        })

    return messages


def run_inference(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    few_shot_messages: list[dict],
    query: Sample,
    device: str = "cuda",
) -> tuple[bool, float, str]:
    """
    few-shot messages に query を追加して推論。
    Returns: (fire_detected, fire_confidence, raw_output)
    """
    query_text = QUERY_USER_TMPL
    messages = few_shot_messages + [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query.image},
                {"type": "text",  "text":  query_text},
            ],
        }
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
    return fire_detected, fire_confidence, raw


def _parse_result(raw: str) -> tuple[bool, float]:
    text = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return False, 0.0
    try:
        data = json.loads(match.group())
        return bool(data.get("fire_detected", False)), float(data.get("fire_confidence", 0.0))
    except Exception:
        return False, 0.0


# ===========================================================================
# 評価レポート
# ===========================================================================

def print_metrics(results: list[tuple[Sample, bool, float, str]]) -> None:
    """results: [(sample, pred_fire, pred_conf, raw_output)]"""
    print(f"\n{'='*70}")
    print("  /poc2 Few-shot ICL 結果")
    print(f"{'='*70}")

    tp = fp = tn = fn = 0
    for sample, pred, conf, raw in results:
        gt = sample.label
        correct = (pred == gt)
        status = "✅" if correct else "❌"
        label_str = "FIRE  " if gt else "NO-FIRE"
        pred_str  = "FIRE  " if pred else "NO-FIRE"
        print(f"  {status} {sample.desc:<20}  GT={label_str}  Pred={pred_str}  conf={conf:.2f}")
        if gt and pred:     tp += 1
        elif not gt and pred: fp += 1
        elif gt and not pred: fn += 1
        else:               tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall   = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    fp_rate  = fp / (fp + tn)    if (fp + tn) > 0 else 0.0

    print(f"\n  {'─'*40}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}  (total={total})")
    print(f"  Accuracy : {accuracy:.2f}")
    print(f"  Recall   : {recall:.2f}  (目標: ≥ 0.70)")
    print(f"  Precision: {precision:.2f}")
    print(f"  FP Rate  : {fp_rate:.2f}  (目標: ≤ 0.50)")

    recall_ok  = recall  >= 0.70
    fp_rate_ok = fp_rate <= 0.50
    go = recall_ok and fp_rate_ok
    print(f"\n  Recall {'✅' if recall_ok else '❌'}  FP Rate {'✅' if fp_rate_ok else '❌'}")
    print(f"  → {'✅ GO — /poc2 完了条件① 達成。LoRA FT (Step2) へ進む。' if go else '❌ NO-GO — ショット数・プロンプト設計を見直す。'}")
    print(f"{'='*70}\n")


# ===========================================================================
# メイン
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=4,
                        help="Few-shot に使うサンプル数 (pos+neg 合計, 偶数推奨)")
    parser.add_argument("--top",   type=int, default=8,
                        help="FIRMS 上位イベント数 (shots//2 + テスト分を確保)")
    parser.add_argument("--days",  type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not FIRMS_KEY:
        print("ERROR: FIRMS_MAP_KEY が設定されていません")
        sys.exit(1)

    n_shot_pairs = args.shots // 2   # 各 pos/neg ペアから1サンプルずつ
    print(f"  Few-shot: {args.shots} サンプル ({n_shot_pairs} positive + {n_shot_pairs} negative)")

    # ------------------------------------------------------------------
    # データ収集
    # ------------------------------------------------------------------
    print(f"\n[1] FIRMS データ取得 (過去{args.days}日, 上位{args.top}件)")
    all_events: list[FIRMSEvent] = []
    for name, area_str in AREAS.items():
        evs = fetch_firms(area_str, args.days)
        print(f"    {name}: {len(evs)} ホットスポット")
        all_events.extend(evs)

    if not all_events:
        print("FIRMS データなし")
        sys.exit(1)

    top_events = select_top(all_events, args.top)
    print(f"\n  上位 {len(top_events)} イベント:")
    for i, e in enumerate(top_events, 1):
        print(f"    {i}. lat={e.lat:.3f} lon={e.lon:.3f}  FRP={e.frp:.0f}MW  {e.iso}")

    client   = SimSatClient()
    spectral = SpectralProcessor()

    print(f"\n[2] SimSat 画像取得 (POS: +{SHIFT_DAYS}d, NEG: -{NEG_OFFSET_DAYS}d, 同一地点)")
    pairs = collect_samples(top_events, client, spectral)

    if len(pairs) < n_shot_pairs + 1:
        print(f"\nERROR: 有効なペアが {len(pairs)} 件のみ。"
              f"few-shot {n_shot_pairs} + テスト 1 件以上が必要。")
        print("  --days を増やすか --shots を減らしてください。")
        sys.exit(1)

    # few-shot / test 分割
    shot_pairs = pairs[:n_shot_pairs]
    test_pairs = pairs[n_shot_pairs:]

    # few-shot samples: pos/neg を交互に並べる
    shot_samples: list[Sample] = []
    for pos, neg in shot_pairs:
        shot_samples.append(pos)
        shot_samples.append(neg)

    test_samples: list[Sample] = []
    for pos, neg in test_pairs:
        test_samples.append(pos)
        test_samples.append(neg)

    print(f"\n  Few-shot: {len(shot_samples)} サンプル (ポジ/ネガ各 {n_shot_pairs})")
    print(f"  テスト  : {len(test_samples)} サンプル")

    # スペクトル分布レポート (全ペア対象)
    all_samples = shot_samples + test_samples
    pos_nbr2 = [s.nbr2_min for s in all_samples if s.label]
    neg_nbr2 = [s.nbr2_min for s in all_samples if not s.label]
    print(f"\n  [スペクトル分布確認 (Δ≥0 フィルター後)]")
    print(f"  POS NBR2_min: min={min(pos_nbr2):.3f}  mean={np.mean(pos_nbr2):.3f}  max={max(pos_nbr2):.3f}  (n={len(pos_nbr2)})")
    print(f"  NEG NBR2_min: min={min(neg_nbr2):.3f}  mean={np.mean(neg_nbr2):.3f}  max={max(neg_nbr2):.3f}  (n={len(neg_nbr2)})")
    sep = np.mean(pos_nbr2) < np.mean(neg_nbr2) - 0.05
    print(f"  分離判定: {'✅ POS < NEG (burn scar シグナルあり)' if sep else '⚠️  分離不十分'}")

    # ------------------------------------------------------------------
    # モデルロード
    # ------------------------------------------------------------------
    print(f"\n[3] モデルロード: {MODEL_ID}")
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=args.device, trust_remote_code=True
    )
    model.eval()
    print(f"  ロード完了  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Few-shot メッセージ構築
    # ------------------------------------------------------------------
    print(f"\n[4] Few-shot context 構築 ({len(shot_samples)} サンプル)")
    few_shot_messages = build_few_shot_messages(shot_samples)
    print(f"  メッセージターン数: {len(few_shot_messages)} "
          f"(system + {len(shot_samples)*2} user/asst)")

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------
    print(f"\n[5] 推論 ({len(test_samples)} テストサンプル)")
    results: list[tuple[Sample, bool, float, str]] = []
    for i, sample in enumerate(test_samples, 1):
        print(f"  [{i}/{len(test_samples)}] {sample.desc} (GT={'FIRE' if sample.label else 'NO-FIRE'})", end=" ... ")
        t0 = time.perf_counter()
        pred, conf, raw = run_inference(model, processor, few_shot_messages, sample, args.device)
        ms = (time.perf_counter() - t0) * 1000
        print(f"Pred={'FIRE' if pred else 'NO-FIRE'} conf={conf:.2f} ({ms:.0f}ms)")
        results.append((sample, pred, conf, raw))

    # ------------------------------------------------------------------
    # 評価レポート
    # ------------------------------------------------------------------
    print_metrics(results)

    # 生出力サンプルを表示
    print("  [Raw LLM Output サンプル (最初の1件)]")
    if results:
        print(f"  {results[0][3][:300]}")


if __name__ == "__main__":
    env = os.path.normpath(os.path.join(SCRIPT_DIR, "../../.env"))
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    main()
