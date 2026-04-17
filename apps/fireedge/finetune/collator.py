"""
FireEdge VLM Data Collator
===========================
SFTTrainer 向け: system+user トークンの loss を -100 でマスクし、
assistant 応答部分のみ学習対象にする。

確認済みチャットテンプレート形式:
  <|startoftext|><|im_start|>system\n...<|im_end|>\n
  <|im_start|>user\n<image>...<|im_end|>\n
  <|im_start|>assistant\n{JSON}<|im_end|>\n
"""
from __future__ import annotations

import json
from typing import Any

import torch


def _find_last_subseq(seq: list[int], subseq: list[int]) -> int:
    """seq の中で subseq が最後に始まるインデックスを返す。見つからなければ -1。"""
    n, m = len(seq), len(subseq)
    for i in range(n - m, -1, -1):
        if seq[i : i + m] == subseq:
            return i
    return -1


class VLMFireCollator:
    """
    messages_json (str) + image (PIL) のバッチ → モデル入力バッチ。

    損失マスク戦略:
      - system + user トークン → labels = -100 (損失計算除外)
      - assistant 応答トークン → labels = input_ids (損失計算対象)
    """

    def __init__(self, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length

        # assistant 開始を示すトークン列を事前計算
        # チャットテンプレートから "<|im_start|>assistant\n" のトークン ID を取得
        asst_header = "<|im_start|>assistant\n"
        self._asst_ids: list[int] = processor.tokenizer.encode(
            asst_header, add_special_tokens=False
        )

    # ------------------------------------------------------------------

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts, images_list = [], []

        for ex in examples:
            messages = json.loads(ex["messages_json"])
            img = ex["image"]  # PIL Image

            # フルテキスト (assistant 応答含む)
            text_full = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text_full)
            images_list.append([img])

        # processor でトークナイズ + 画像処理を一括実行
        batch = self.processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = batch["input_ids"]          # (B, L)
        attention_mask = batch["attention_mask"] # (B, L)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100       # パディング部分を除外

        # assistant 応答開始位置より前を -100 でマスク
        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()
            pos = _find_last_subseq(seq, self._asst_ids)
            if pos >= 0:
                # assistant ヘッダーの直後から損失計算開始
                labels[i, : pos + len(self._asst_ids)] = -100
            else:
                # 見つからなければ全トークンを除外 (安全フォールバック)
                labels[i, :] = -100

        batch["labels"] = labels
        return batch
