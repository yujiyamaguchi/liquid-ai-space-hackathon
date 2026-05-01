"""
FireEdge LoRA Fine-Tuning
==========================
LFM2.5-VL-450M を SWIR 火災検知データセットで LoRA SFT する。

使い方:
    uv run python -m finetune.train
    uv run python -m finetune.train --epochs 5 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from finetune.collator import VLMFireCollator
from finetune.config import FinetuneConfig


# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--output",        type=str,   default=None)
    p.add_argument("--dataset",       type=str,   default=None)
    p.add_argument("--no-mask-asst",  action="store_true", default=False,
                   help="system+user トークンも loss 対象にする (full_seq 実験)")
    p.add_argument("--run-name",      type=str,   default=None,
                   help="実験名 (output subdir)。未指定時は mask_asst / full_seq を自動設定")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = FinetuneConfig()

    if args.epochs:       cfg.num_train_epochs = args.epochs
    if args.lr:           cfg.learning_rate = args.lr
    if args.output:       cfg.output_dir = args.output
    if args.dataset:      cfg.dataset_dir = args.dataset
    if args.no_mask_asst: cfg.mask_non_assistant = False
    if args.run_name:
        cfg.run_name = args.run_name
    elif not cfg.run_name:
        cfg.run_name = "mask_asst" if cfg.mask_non_assistant else "full_seq"

    print("=" * 60)
    print("FireEdge LoRA Fine-Tuning")
    print(f"  model     : {cfg.model_id}")
    print(f"  run_name  : {cfg.run_name}")
    print(f"  mask_asst : {cfg.mask_non_assistant}")
    print(f"  output    : {cfg.output_dir}/{cfg.run_name}")
    print(f"  epochs    : {cfg.num_train_epochs}")
    print(f"  lr        : {cfg.learning_rate}")
    print(f"  lora_r    : {cfg.lora_r}")
    print("=" * 60)

    # ---------------------------------------------------------------- dataset
    ds_path = Path(cfg.dataset_dir)
    if not ds_path.exists():
        print(f"[ERROR] dataset が見つかりません: {ds_path}")
        print("  先に `uv run python -m finetune.dataset_builder` を実行してください。")
        sys.exit(1)

    train_ds = load_from_disk(str(ds_path / "train"))
    val_ds   = load_from_disk(str(ds_path / "val"))
    print(f"[Train] train={len(train_ds)}, val={len(val_ds)}")
    print(f"[Train] fire in train: {sum(train_ds['label'])}/{len(train_ds)}")
    print(f"[Train] fire in val:   {sum(val_ds['label'])}/{len(val_ds)}")

    # ------------------------------------------------------------------ model
    print("\n[Train] モデルをロード中 ...")
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # gradient checkpointing 必須

    # ------------------------------------------------------------------ LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --------------------------------------------------------------- collator
    collator = VLMFireCollator(processor, max_length=cfg.max_seq_length,
                               mask_non_assistant=cfg.mask_non_assistant)

    # ------------------------------------------------------------ SFT config
    output_dir = Path(ROOT) / cfg.output_dir / cfg.run_name
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="none",
        seed=cfg.seed,
        # SFT 固有: テキストフィールドは collator で処理するので None
        dataset_text_field=None,
        max_length=cfg.max_seq_length,  # trl>=1.1.0: max_seq_length → max_length
    )

    # ---------------------------------------------------------------- trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
    )

    print("\n[Train] 学習開始 ...")
    trainer.train()

    # ------------------------------------------------------------------ save
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Train] アダプタ保存中: {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    # 学習曲線を JSON で保存
    log_history = trainer.state.log_history
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print("[Train] 完了!")
    print(f"  adapter   : {adapter_dir}")
    print(f"  train log : {output_dir / 'train_log.json'}")


if __name__ == "__main__":
    main()
