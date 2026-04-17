"""
FireEdge Fine-Tuning Configuration
===================================
全ハイパーパラメータを一箇所で管理。
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FinetuneConfig:
    # ------------------------------------------------------------------ model
    model_id: str = "LiquidAI/LFM2.5-VL-450M"
    output_dir: str = "output/fireedge-lora"
    seed: int = 42

    # ---------------------------------------------------------------- dataset
    # 取得シーン数 (fire / no-fire 各上限)
    n_fire_scenes: int = 40
    n_nofire_scenes: int = 40
    val_split: float = 0.15          # 15% を validation に
    max_cloud_cover: float = 50.0    # 雲量上限 [%]
    simsat_base_url: str = "http://localhost:9005"
    dataset_dir: str = "data/finetune/dataset"
    image_size: int = 448

    # ------------------------------------------------------------------ LoRA
    # Linear suffix 一覧 (conv.conv は Conv1d なので除外)
    lora_target_modules: list[str] = field(default_factory=lambda: [
        # LM attention layers (7 full-attention layers)
        "q_proj", "k_proj", "v_proj", "out_proj",
        # LM FFN (全16層)
        "w1", "w2", "w3",
        # LM hybrid conv projection (9 conv layers)
        "in_proj",
        # Multimodal projector (image→text grounding の要)
        "linear_1", "linear_2",
    ])
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # --------------------------------------------------------------- training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8   # effective batch = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    eval_steps: int = 20
    save_steps: int = 20
    save_total_limit: int = 2
    dataloader_num_workers: int = 0   # 0 = main process (安定)
