# 衛星オンボードハードウェア仕様

ハッカソン提出物の実行環境として想定するハードウェアスペックをまとめる。

---

## 確定スペック

| 項目 | 値 | 備考 |
|---|---|---|
| **SoC** | NVIDIA Jetson Orin NX | DPhi Space 実行環境として明記 |
| **AI 演算性能** | 175 TOPS | INT8 換算 |
| **GPU** | 1024 コア（Ampere アーキテクチャ） | CUDA 対応 |
| **RAM** | 16 GB | Winner Prize の "NVIDIA Orin 16GB" と一致 |
| **オンボードストレージ** | テラバイト単位 | DPhi Space 実行環境ページ「Terabytes of Storage」記載。Winner Prize の「1GB/月」はクレジット割り当て量であり、ハードウェア実容量とは別 |
| **データアップリンク** | 5 MB | 衛星へのアップロード上限（Winner Prize） |
| **データダウンリンク** | 10 MB | 衛星からのダウンロード上限（Winner Prize） |
| **GPU 使用時間** | 5 GPU 時間 | Winner Prize クレジット |

---

## 出典

### 1. DPhi Space ソフトウェア実行環境ページ
- URL: https://software.dphispace.com/
- 記載内容（原文抜粋）:
  ```
  Jetson Orin NX
  AI Perf  175 TOPS
  GPU      1024 Ampere
  RAM      16 GB
  Terabytes of Storage
  ```

### 2. ハッカソン審査基準・賞品ドキュメント（Judging Criteria and Prizes）
- URL: https://docs.google.com/document/d/1hG569zx0laVWiVKP1ek7U0T_VrrydEBF/edit
- Winner Prize の記載（原文抜粋）:
  ```
  Cash Prize $5'000
  Credits for Software Execution in Space
  5 GPU hours (NVIDIA Orin 16GB)
  5MB of data upload to satellite
  10MB of data download from satellite
  1GB of in-space storage for 1 month
  All historic images of fisheye camera on the satellite.
  Access to all public Docker images and LLMs preloaded on the satellite.
  7 days of testing on the satellite ground compute server.
  ```

---

## 設計上の制約

### メモリ制約

| 用途 | 想定 | 根拠 |
|---|---|---|
| LFM 2.5-VL-450M 推論 | ~1 GB VRAM | bfloat16 で 450M params ≈ 900MB |
| 画像バッファ（入力） | ~20〜100 MB | Sentinel-2 タイル（256×256 px × 13バンド × uint16 = 1.7 MB/タイル）。LFM 推論直前に bfloat16/float16 へキャスト |
| 推論オーバーヘッド | ~500 MB | KV cache・アクティベーション等 |
| **合計（推論時ピーク）** | **~2 GB** | 16GB RAM に対して十分余裕あり |

→ **LFM 2.5-VL-450M は 16GB RAM の Jetson Orin NX で実行可能**

### ストレージ設計

**モデル重みについて**: LFM ベースモデルおよび LoRA アダプタは Docker イメージとして衛星にプリロードされる想定（Winner Prize: "Access to all public Docker images and LLMs preloaded on the satellite"）。動的なアップリンクは不要。

**ストレージ容量の解釈**: DPhi Space の実行環境ページには「Terabytes of Storage」と明記されており、ハードウェア実容量はテラバイト単位。Winner Prize に記載の「1GB of in-space storage for 1 month」は賞品として付与される利用クレジット（テスト期間中の割り当て量）であり、衛星ハードウェアの実容量とは別物。

ハッカソンの与件（"What if satellites had onboard compute?"）の文脈では、テラバイト級ストレージを前提に設計してよい。過去の撮像履歴（フィッシュアイカメラ等）をオンボードに保持し、軌道上でのタイムシリーズ比較・ベースライン管理が可能になる。これがオンボードAIの核心的な価値。

#### キャッシュ用途と推定サイズ

| 用途 | 内容 | 推定サイズ | 対応アプリ |
|---|---|---|---|
| **AOI ベースライン画像** | 定点監視対象（バース・ヤード等）の T0（正常時）タイルを保存。通過ごとに T0 vs 現在値を軌道上で差分比較 | 50 AOI × 256×256 × 5band × uint16 = **~33 MB** | HarborLens, PortPulse, DarkFleet |
| **スペクトル統計 時系列** | タイルごとの過去 3〜5 パス分の NBR/NDVI/NDWI の平均・標準偏差を保存。生画像より 100x 小さい | 1,500 タイル × 5 stats × 5 passes × float32 = **~1.5 MB/シーン** | FireGuard, CoralShield, InfraWatch |
| **Few-shot 参照画像** | LFM への in-context learning 用に正例・負例タイルを保存。毎パスでプロンプトに添付 | 4 クラス × 5 例 × 224×224 × 3band × uint8 = **~18 MB** | OrbitTriage, FireGuard, CoralShield |
| **アラート送信待ちキュー** | 地上局接触まで生成済みアラートを保持 | 2KB × 5,000 件 = **~10 MB** | 全アプリ共通 |
| **LFM 入力画像バッファ** | 直前パスの入力タイルを保持。リビジット時の再推論コスト削減 | 10 タイル × 1.3 MB = **~13 MB** | 全アプリ共通 |

**合計見積**: 33（AOIベースライン）+ 18（Few-shot）+ 10（アラートキュー）+ 13（LFM入力バッファ）+ 時系列統計（シーン数による）≈ **80〜200 MB**。テラバイト級ストレージに対しては余裕が大きく、過去撮像の全タイル保持・長期タイムシリーズ蓄積なども十分可能。

### 帯域幅制約

| 方向 | 上限 | 設計方針 |
|---|---|---|
| アップリンク（地上→衛星） | 5 MB | AOI 定義 JSON・閾値パラメータ更新（数KB〜数十KB）は収まる。モデル重みは Docker イメージに同梱済みのため不要 |
| ダウンリンク（衛星→地上） | 10 MB | アラート JSON（2KB × 5,000 件以内）は収まる。高リスクタイル画像（JPEG 圧縮 ~50KB × 20 枚 = 1MB）の添付も可能 |

---

## CLAUDE.md との照合

CLAUDE.md に記載の `LFM 2.5-VL-450M (Grounding & Structured Output 優先)` および `< 1GB VRAM` という制約はこのスペックで成立する。

開発環境（RTX 5090 / 24GB VRAM）との差分:
- Jetson Orin NX は CUDA Ampere だが、RTX 5090 より推論速度は遅い
- 推論レイテンシ目標（< 30秒/シーン）は Jetson Orin NX で別途計測が必要

### 量子化の要否

| 観点 | 結論 | 根拠 |
|---|---|---|
| **メモリ** | **不要** | bfloat16 で ~900MB。OS・KV cache 込みでも ~4〜5GB。16GB 統合メモリに余裕あり |
| **レイテンシ** | **任意** | 175 TOPS は INT8（= Q8）計測値。bfloat16 で動かすとこのピーク性能を引き出せない。INT8/Q8 に量子化すると 1.5〜2x 速くなる可能性があるが、450M パラメータで <30秒/シーンの目標は bfloat16 でも達成できる見込み |
| **エッジ慣習** | **INT8/Q8 がデファクト** | エッジ VLM では GGUF Q8_0（= INT8 量子化）が標準的な配布形式。LFM 2.5-VL の公式推奨デプロイ形式に従うのが確実 |

補足: INT8 と Q8（GGUF Q8_0）は同一の量子化精度（8ビット整数）。Q8_0 はブロックごとのスケール係数を持つ実装形式の名称。

→ **量子化しなければならない理由はない**。実装方針としては、LiquidAI が提供する公式推論フォーマット（bfloat16 または INT8/Q8）をそのまま使い、Jetson 上で実測してから判断する。推論が 30 秒を超える場合に限り INT8/Q8 への移行を検討。
