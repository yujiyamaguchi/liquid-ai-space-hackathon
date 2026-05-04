# FireGuard /poc — 発火前スペクトル分離確認

発火前 NDMI シグナルが POS/NEG チャパラルで分離できるかを検証する。
SimSat チェック → poc.py 実行（ログ tee 保存）→ report.md 表示。

## 引数

`$ARGUMENTS` に以下を渡せる（省略時はデフォルト値）:
- `--run-name NAME` : 実験名。省略時はタイムスタンプ (`YYYYMMDD_HHMMSS`) を自動設定

例: `/fireguard-poc`
例: `/fireguard-poc --run-name v3_rerun`

## 手順

1. **SimSat ヘルスチェック**
   ```
   curl -s http://localhost:9005/
   ```
   `{"message":"Simulation API is online"}` 以外なら中断してユーザーに報告する。

2. **run_name 決定・ログ/出力ディレクトリ作成**

   `$ARGUMENTS` から `--run-name` を抽出する。未指定の場合は `$(date +%Y%m%d_%H%M%S)` を使用。
   出力先: `apps/fireguard/data/poc/{RUN_NAME}/`
   ログ先: `apps/fireguard/data/poc/logs/`

   ```bash
   RUN_NAME=$(echo "$ARGUMENTS" | grep -oP '(?<=--run-name )\S+' || echo "")
   if [ -z "$RUN_NAME" ]; then RUN_NAME=$(date +%Y%m%d_%H%M%S); fi
   mkdir -p apps/fireguard/data/poc/logs
   mkdir -p "apps/fireguard/data/poc/${RUN_NAME}"
   ```

3. **poc.py 実行（ログ tee 保存）**

   ログファイル名: `apps/fireguard/data/poc/logs/{YYYYMMDD_HHMMSS}_{RUN_NAME}.log`

   ```bash
   LOG=apps/fireguard/data/poc/logs/$(date +%Y%m%d_%H%M%S)_${RUN_NAME}.log
   {
     echo "=== fireguard-poc ==="
     echo "date: $(date -Iseconds)"
     echo "run_name: ${RUN_NAME}"
     echo "cmd: uv run python poc.py --run-name ${RUN_NAME}"
     echo "====================="
   } | tee "$LOG"
   cd apps/fireguard && uv run python poc.py --run-name "${RUN_NAME}" 2>&1 | tee -a "../../$LOG"
   ```

4. **report.md の内容をユーザーに表示**

   ```bash
   cat "apps/fireguard/data/poc/${RUN_NAME}/report.md"
   ```

5. **ログ・レポートパスをユーザーに伝える**

   - ログ: `apps/fireguard/data/poc/logs/{timestamp}_{RUN_NAME}.log`
   - レポート: `apps/fireguard/data/poc/{RUN_NAME}/report.md`
   - 画像: `apps/fireguard/data/poc/{RUN_NAME}/images/`
   - 図: `apps/fireguard/data/poc/{RUN_NAME}/figures/`
