# FireEdge データセット収集

FireEdge のファインチューニング用データセットを収集する。
SimSat チェック → dataset_builder 実行 → ログ保存 → report 表示。

## 引数

`$ARGUMENTS` に以下を渡せる（省略時はデフォルト値）:
- `--n-pos N`      : FIRMS POS 件数 (default: 100)
- `--no-diverse`   : diverse NEG をスキップ
- `--min-frp F`    : FRP 最低閾値 [MW] (default: 0.0)

例: `/fireedge-collect --n-pos 50 --no-diverse`

## 手順

1. **SimSat ヘルスチェック**
   ```
   curl -s http://localhost:9005/
   ```
   `{"message":"Simulation API is online"}` 以外なら中断してユーザーに報告する。

2. **ログディレクトリ作成**
   ```
   mkdir -p apps/fireedge/data/finetune/logs
   ```

3. **dataset_builder 実行（ログ tee 保存）**

   ログファイル名: `apps/fireedge/data/finetune/logs/YYYYMMDD_HHMMSS.log`
   （`date +%Y%m%d_%H%M%S` で生成）

   ログ先頭に実行コマンドを記録してから tee で実行する:
   ```bash
   LOG=apps/fireedge/data/finetune/logs/$(date +%Y%m%d_%H%M%S).log
   {
     echo "=== fireedge-collect ==="
     echo "date: $(date -Iseconds)"
     echo "cmd: uv run python -m finetune.dataset_builder $ARGUMENTS"
     echo "========================"
   } | tee "$LOG"
   cd apps/fireedge && uv run python -m finetune.dataset_builder $ARGUMENTS 2>&1 | tee -a "../$LOG"
   ```
   ※ working directory が異なる場合はパスを調整すること。

4. **終了後 report 表示**
   ```
   cd apps/fireedge && uv run python -c "
   from finetune.dataset_builder import DatasetBuilder
   DatasetBuilder().report()
   "
   ```

5. **ログパスをユーザーに伝える**
   保存先ログファイルのパスを表示する。
