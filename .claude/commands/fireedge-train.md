# FireEdge LoRA ファインチューニング

FireEdge の LoRA 学習 → evaluate を実行する。
train.py 実行 → ログ保存 → evaluate.py 実行 → 結果表示。

## 引数

`$ARGUMENTS` に以下を渡せる（省略時はデフォルト値）:
- `--epochs N`   : エポック数 (default: 5)
- `--lr F`       : 学習率 (default: 2e-4)
- `--run-base`   : evaluate 時に base model も評価して比較表を出力する（時間がかかる）

例: `/fireedge-train --epochs 3 --run-base`

## 手順

1. **HF dataset 存在確認**

   以下のディレクトリが存在しなければ中断してユーザーに報告する。
   ```
   apps/fireedge/data/finetune/hf_dataset/train
   apps/fireedge/data/finetune/hf_dataset/val
   apps/fireedge/data/finetune/hf_dataset/test
   ```
   存在しない場合: 「先に `/fireedge-collect` を実行してください」と伝える。

2. **ログディレクトリ作成**
   ```
   mkdir -p apps/fireedge/data/finetune/logs
   ```

3. **train.py 実行（ログ tee 保存）**

   ログファイル名: `data/finetune/logs/YYYYMMDD_HHMMSS_train.log`
   （`apps/fireedge` 基点の相対パス）
   train.py に渡す引数は `--epochs` と `--lr` のみ（`--run-base` は evaluate に渡す）。

   ```bash
   cd apps/fireedge
   LOG=data/finetune/logs/$(date +%Y%m%d_%H%M%S)_train.log
   {
     echo "=== fireedge-train ==="
     echo "date: $(date -Iseconds)"
     echo "cmd: uv run python -m finetune.train $ARGUMENTS"
     echo "======================"
   } | tee "$LOG"
   uv run python -m finetune.train $TRAIN_ARGS 2>&1 | tee -a "$LOG"
   ```
   ※ `$TRAIN_ARGS` には `$ARGUMENTS` から `--run-base` を除いた引数を渡す。
   ※ 学習は RTX 5090 で 20〜40 分かかる見込み。バックグラウンド実行を使うこと。

4. **adapter 存在確認**

   学習完了後、以下が存在することを確認する。
   ```
   apps/fireedge/output/fireedge-lora/adapter/
   ```
   存在しなければ中断してエラーログをユーザーに示す。

5. **evaluate.py 実行**

   `--run-base` が指定された場合は base model も評価する（比較表・グラフを生成）。
   指定なし（デフォルト）はファインチューニング済みモデルのみ評価する。

   ```bash
   cd apps/fireedge
   LOG=data/finetune/logs/$(date +%Y%m%d_%H%M%S)_eval.log
   {
     echo "=== fireedge-evaluate ==="
     echo "date: $(date -Iseconds)"
     echo "========================="
   } | tee "$LOG"
   uv run python -m finetune.evaluate $EVAL_ARGS 2>&1 | tee -a "$LOG"
   ```
   ※ `$EVAL_ARGS`: `--run-base` が指定された場合は `--run-base`、それ以外は空。

6. **結果表示とログパスをユーザーに伝える**

   evaluate の標準出力に含まれる以下を整形してユーザーに伝える:
   - Precision / Recall / F1 / FP Rate の Base vs LoRA 比較表
   - 目標達成チェック (Recall ≥ 0.85 / FP Rate ≤ 0.15)
   - 比較グラフ保存先: `apps/fireedge/data/finetune/eval/base_vs_finetuned.png`
   - train / eval ログのパス
