# FireEdge — 損失マスク戦略 実験レポート

*Loss Masking Strategy: Assistant-Only vs Full-Sequence*

---

## 概要

SFT (Supervised Fine-Tuning) において損失計算の対象トークンをどう選ぶかは、  
モデルが「何を学習するか」を根本的に左右する。  
本実験では以下2つの戦略を比較し、どちらが火災検知タスクに適切かを定量的に検証した。

| 実験名 | 戦略 | 損失対象トークン |
|---|---|---|
| `mask_asst` | assistant のみ（デフォルト） | assistant 応答 `{"fire_detected": true/false}` のみ |
| `full_seq` | 全シーケンス | system + user + assistant すべて |

---

## 実験条件

- **モデル**: LFM 2.5-VL-450M
- **アダプタ**: LoRA r=16, alpha=32, dropout=0.05
- **データセット**: POS 100件 + firms_NEG 100件 + diverse_NEG 100件 = 計300件  
  Train 70% / Val 15% / Test 15%（stratified）
- **エポック**: 5
- **学習率**: 2e-4（cosine scheduler, warmup_ratio=0.05）
- **実効バッチサイズ**: 8（per_device=1 × gradient_accumulation=8）
- **実行環境**: NVIDIA RTX 5090 (24GB VRAM)

---

## 学習曲線

### Train Loss

| Step | mask_asst | full_seq |
|-----:|----------:|---------:|
|    5 |  0.3114   |  8.0121  |
|   20 |  0.0687   |  0.8152  |
|   40 |  0.0256   |  0.0528  |
|   60 |  0.0136   |  0.0189  |
|   80 |  0.0047   |  0.0067  |
|  100 |  0.0019   |  0.0034  |
|  135 |  0.0007   |  0.0030  |

### Eval Loss (Validation Set)

| Step | mask_asst | full_seq |
|-----:|----------:|---------:|
|   20 |  0.0660   |  0.3858  |
|   40 |  0.0289   |  0.0454  |
|   60 |  0.0191   |  0.0157  |
|   80 |  **0.0158** (best) |  0.0049  |
|  100 |  0.0186   |  0.0030  |
|  120 |  0.0202   |  **0.0026** (best) |
|  135 |  0.0202   |  0.0026  |

**観察:**

- `mask_asst` は step 80 でeval_loss が底打ち（0.0158）し、その後わずかに上昇（過学習の兆候）。  
  `load_best_model_at_end=True` により step 80 のチェックポイントが採用された。
- `full_seq` はeval_loss が step 120 まで単調に低下し続け 0.0026 に達した。  
  一見「過学習なし」に見えるが、これは system + user トークン（固定テンプレート）を  
  丸暗記することで損失が下がり続けているためであり、検知精度の向上を意味しない。

---

## テストセット評価結果

| Metric | mask_asst | full_seq | 目標 |
|--------|----------:|---------:|:----:|
| Precision | **1.000** | 0.000 | — |
| Recall | **0.933** | 0.000 | ≥ 0.85 |
| F1 | **0.966** | 0.000 | — |
| FP Rate | **0.000** | 0.000 | ≤ 0.15 |
| Accuracy | **0.978** | 0.667 | — |
| Latency (mean ms) | 149.5 | 184.1 | — |
| Latency (P95 ms) | 184.9 | 219.7 | — |
| JSON 解析成功率 | 1.000 | 1.000 | — |

### 混同行列

**mask_asst:**
```
              Predicted
              NO-FIRE   FIRE
True NO-FIRE    30        0
True FIRE        1       14
```

**full_seq:**
```
              Predicted
              NO-FIRE   FIRE
True NO-FIRE    30        0
True FIRE       15        0   ← 全件 no-fire 予測に崩壊
```

---

## 考察

### full_seq が崩壊した理由

チャットテンプレートの構造を確認すると、1サンプルのトークン構成は以下のとおり（実測値）:

```
<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n          ← ~200 tokens (固定)
<|im_start|>user\n[画像トークン×196]{USER_PROMPT}<|im_end|> ← 196 tokens (可変) + ~42 tokens (固定)
<|im_start|>assistant\n{"fire_detected": true/false}<|im_end|> ← ~10 tokens (学習対象)
合計 451 tokens
```

`full_seq` での損失を性質別に整理すると:

| 区分 | トークン数 | 性質 | full_seq での挙動 |
|---|---:|---|---|
| system + user テキスト | 242 | **固定**（全サンプル同一） | 丸暗記で損失をゼロにできる → 勾配を能動的に食いつぶす |
| 画像トークン | 196 | **可変**（サンプルごとに異なる） | 暗記不可だが task と無関係な予測に勾配が散る |
| assistant 応答 | 10 | **学習対象** | 全体に占める割合 2.2% → 有効な勾配がほぼ届かない |

固定テキスト（242T）の暗記が最もコスパの高い勾配降下方向になるため、  
`{"fire_detected": true/false}` の2択を学ぶ前に損失が収束してしまった。

最終的に訓練データの `no-fire` 比率（約67%）に合わせて  
「常に `false` を返す」バイアスに収束した。

### mask_asst が有効な理由

assistant 応答トークンのみに損失を絞ることで:
1. **勾配が火災判定に集中する** — テンプレート丸暗記に勾配が消費されない
2. **信号対雑音比が改善する** — 学習対象の10トークンが毎ステップ有効に更新される
3. **適切なタイミングで早期停止できる** — eval loss が過学習を正しく検出できる

---

## 結論

> **assistant 応答トークンのみに損失を絞る `mask_asst` が正しいアプローチであることを定量的に確認した。**

`full_seq` は eval loss が 0.003 まで下がるが、これはテンプレート暗記によるもので  
検知精度を一切改善しない（Recall 0.000、全件崩壊）。  
`mask_asst` は Recall 0.933 / FP Rate 0.000 を達成し、設定目標を満たしている。

本実験の知見は、VLM を分類タスクに SFT する際の一般的な教訓として適用できる:  
**質問・指示が固定テンプレートである場合、full-sequence loss は有害。**

---

*実験データ保存先:*
- `output/fireedge-lora/mask_asst/` → `data/finetune/eval/results.json`（mask_asst）
- `output/fireedge-lora/full_seq/` → `data/finetune/eval/full_seq/results.json`（full_seq）
