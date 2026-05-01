# FireEdge LoRA ファインチューニング

FireEdge の LoRA 学習 → evaluate を実行する。
train.py 実行 → ログ保存 → evaluate.py 実行 → 結果表示。

## 引数

`$ARGUMENTS` に以下を渡せる（省略時はデフォルト値）:
- `--epochs N`      : エポック数 (default: 5)
- `--lr F`          : 学習率 (default: 2e-4)
- `--no-mask-asst`  : system+user トークンも loss 対象にする（full_seq 実験）
- `--run-name NAME` : 実験名。省略時は `mask_asst` または `full_seq` を自動設定
- `--run-base`      : evaluate 時に base model も評価して比較表を出力する（時間がかかる）

例: `/fireedge-train --epochs 3 --run-base`
例: `/fireedge-train --no-mask-asst --epochs 5`
例: `/fireedge-train --no-mask-asst --run-name full_seq_lr1e4 --lr 1e-4`

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

3. **run_name の導出と train.py 実行（ログ tee 保存）**

   `$ARGUMENTS` から `--run-name` を抽出する。未指定の場合:
   - `--no-mask-asst` が含まれていれば `RUN_NAME=full_seq`
   - そうでなければ `RUN_NAME=mask_asst`

   ログファイル名: `data/finetune/logs/YYYYMMDD_HHMMSS_{RUN_NAME}_train.log`
   （`apps/fireedge` 基点の相対パス）
   train.py に渡す引数は `--run-base` を除いた全引数。

   ```bash
   cd apps/fireedge
   LOG=data/finetune/logs/$(date +%Y%m%d_%H%M%S)_${RUN_NAME}_train.log
   {
     echo "=== fireedge-train ==="
     echo "date: $(date -Iseconds)"
     echo "run_name: ${RUN_NAME}"
     echo "cmd: uv run python -m finetune.train $TRAIN_ARGS"
     echo "======================"
   } | tee "$LOG"
   uv run python -m finetune.train $TRAIN_ARGS 2>&1 | tee -a "$LOG"
   ```
   ※ `$TRAIN_ARGS` には `$ARGUMENTS` から `--run-base` を除いた引数を渡す。
   ※ 学習は RTX 5090 で 20〜40 分かかる見込み。バックグラウンド実行を使うこと。

4. **adapter 存在確認**

   学習完了後、以下が存在することを確認する（`{RUN_NAME}` は Step 3 で導出した値）。
   ```
   apps/fireedge/output/fireedge-lora/{RUN_NAME}/adapter/
   ```
   存在しなければ中断してエラーログをユーザーに示す。

5. **evaluate.py 実行**

   `--run-base` が指定された場合は base model も評価する（比較表・グラフを生成）。
   指定なし（デフォルト）はファインチューニング済みモデルのみ評価する。

   `$EVAL_ARGS` に必ず `--adapter output/fireedge-lora/{RUN_NAME}/adapter --run-name {RUN_NAME}` を含める。
   `--run-base` が指定された場合はさらに `--run-base` を追加。

   ```bash
   cd apps/fireedge
   LOG=data/finetune/logs/$(date +%Y%m%d_%H%M%S)_${RUN_NAME}_eval.log
   {
     echo "=== fireedge-evaluate ==="
     echo "date: $(date -Iseconds)"
     echo "run_name: ${RUN_NAME}"
     echo "========================="
   } | tee "$LOG"
   uv run python -m finetune.evaluate $EVAL_ARGS 2>&1 | tee -a "$LOG"
   ```

6. **結果表示とログパスをユーザーに伝える**

   evaluate の標準出力に含まれる以下を整形してユーザーに伝える:
   - Precision / Recall / F1 / FP Rate の Base vs LoRA 比較表
   - 目標達成チェック (Recall ≥ 0.85 / FP Rate ≤ 0.15)
   - 比較グラフ保存先: `apps/fireedge/data/finetune/eval/{RUN_NAME}/base_vs_finetuned.png`
   - train / eval ログのパス
