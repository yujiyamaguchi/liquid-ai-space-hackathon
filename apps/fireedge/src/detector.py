"""
[3] FireDetector — LFM 2.5-VL-450M 推論エンジン
================================================
ProcessedScene を受け取り、LFM 2.5-VL に疑似カラー合成画像を渡して
構造化 JSON アラートを生成する。

モデル: LiquidAI/LFM2.5-VL-450M
入力 : fire_composite (PIL 448×448 RGB) + スペクトル指標
出力 : FireDetectionResult (JSON パース済み)
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import (
    FIRE_DETECTION_SYSTEM_PROMPT,
    FIRE_DETECTION_USER_PROMPT,
    FireDetectionResult,
    FireSeverity,
    LFMInferenceConfig,
    ProcessedScene,
    SpreadDirection,
)

ALERT_CONFIDENCE_THRESHOLD = 0.6


class FireDetector:
    """LFM 2.5-VL-450M を用いた煙透過型火災検知モジュール。"""

    def __init__(self, config: LFMInferenceConfig | None = None) -> None:
        self.cfg = config or LFMInferenceConfig()
        self._model: Optional[AutoModelForImageTextToText] = None
        self._processor: Optional[AutoProcessor] = None

    # ------------------------------------------------------------------
    # モデルロード (遅延初期化)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """モデルと processor を GPU にロードする。初回のみ実行。"""
        if self._model is not None:
            return

        print(f"[FireDetector] Loading {self.cfg.model_id} ...")
        dtype = torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float16

        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model_id, trust_remote_code=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model_id,
            dtype=dtype,
            device_map=self.cfg.device,
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"[FireDetector] Model loaded on {self.cfg.device}.")

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def detect(self, scene: ProcessedScene) -> FireDetectionResult:
        """
        疑似カラー合成画像 + スペクトル指標 → FireDetectionResult

        Args:
            scene: SpectralProcessor が生成した ProcessedScene

        Returns:
            パース・バリデーション済みの FireDetectionResult
        """
        self.load()

        user_prompt = FIRE_DETECTION_USER_PROMPT.format(
            nbr2=scene.indices.nbr2,
            ndvi=scene.indices.ndvi,
            bai=scene.indices.bai,
            mean_swir22=scene.indices.mean_swir22,
            fire_pixel_ratio=scene.indices.fire_pixel_ratio,
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": FIRE_DETECTION_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": scene.fire_composite},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # processor でトークナイズ
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.cfg.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=(self.cfg.temperature > 0),
            )
        inference_ms = (time.perf_counter() - t0) * 1000

        # 入力トークンを除いて生成部分のみデコード
        input_len = inputs["input_ids"].shape[1]
        raw_text = self._processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        return self._parse_output(raw_text, inference_ms)

    # ------------------------------------------------------------------
    # 出力パース
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(raw: str, inference_ms: float) -> FireDetectionResult:
        """
        LLM の生テキストから JSON を抽出してバリデーションする。
        JSON が壊れていた場合は安全側に倒したデフォルト値を返す。
        """
        data = _extract_json(raw)

        fire_detected  = bool(data.get("fire_detected", False))
        fire_confidence = float(data.get("fire_confidence", 0.0))

        # severity
        try:
            severity = FireSeverity(data.get("severity", "NONE"))
        except ValueError:
            severity = FireSeverity.NONE

        # spread_direction
        spread_raw = data.get("spread_direction")
        try:
            spread = SpreadDirection(spread_raw) if spread_raw else None
        except ValueError:
            spread = None

        # fire_front_bbox
        bbox_raw = data.get("fire_front_bbox")
        if bbox_raw and len(bbox_raw) == 4:
            bbox: Optional[tuple[int, int, int, int]] = tuple(int(v) for v in bbox_raw)
        else:
            bbox = None

        # alert_recommended: モデル判断 OR 信頼度閾値超え
        alert = bool(data.get("alert_recommended", False)) or (
            fire_detected and fire_confidence >= ALERT_CONFIDENCE_THRESHOLD
        )

        def _float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        return FireDetectionResult(
            smoke_detected=bool(data.get("smoke_detected", False)),
            smoke_confidence=_float(data.get("smoke_confidence")),
            smoke_area_fraction=_float(data.get("smoke_area_fraction")),
            fire_detected=fire_detected,
            fire_confidence=fire_confidence,
            fire_area_ha=_float(data.get("fire_area_ha")),
            fire_front_bbox=bbox,
            spread_direction=spread,
            severity=severity,
            alert_recommended=alert,
            description=str(data.get("description") or "")[:100],
            raw_llm_output=raw,
            inference_time_ms=inference_ms,
        )


# ------------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    LLM 出力テキストから JSON オブジェクトを抽出する。
    マークダウンコードブロック (```json ... ```) にも対応。
    """
    # ```json ... ``` または ``` ... ``` を除去
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # 最初の { ... } を貪欲マッチ
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}
