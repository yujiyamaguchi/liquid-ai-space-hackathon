# アプリ案比較表

> **ハードウェア基準**: Jetson Orin NX（RAM 16GB、175 TOPS INT8、1GB in-space storage）。詳細は [docs/onboard_hardware.md](../docs/onboard_hardware.md) を参照。

## 進行中・検討中

| | **FireEdge** | **FireGuard** | **DarkFleet** | **HarborLens** | **PortPulse** | **InfraWatch** | **CoralShield** | **OrbitTriage** |
|---|---|---|---|---|---|---|---|---|
| **ドメイン** | 野火検知（発火後） | **山火事リスク予測（発火前）** | ダーク船舶追跡 | 港湾タンク在庫監視 | 港湾/製造拠点のサプライチェーン予兆 | 洪水後インフラ損傷 | サンゴ礁白化 | 観測優先度トリアージ |
| **Lean Canvas** | [fireedge/docs/](fireedge/docs/lean_canvas.md) | [fireguard/docs/](fireguard/docs/lean_canvas.md) | [darkfleet/docs/](darkfleet/docs/lean_canvas.md) | [harborlens/docs/](harborlens/docs/lean_canvas.md) | [portpulse/docs/](portpulse/docs/lean_canvas.md) | [infrawatch/docs/](infrawatch/docs/lean_canvas.md) | [coralshield/docs/](coralshield/docs/lean_canvas.md) | [orbittriage/docs/](orbittriage/docs/lean_canvas.md) |
| **使用バンド** | B12/B11/B8 (SWIR22・SWIR16・NIR) | **B8A/B12(NBR)・B8/B4(NDVI)・B11 時系列** | B4/B3/B2 (RGB 10m) | B2〜B12 全般 | B4/B3/B2 (RGB 10m) | B4/B3/B8/B11/B12 | B2/B3/B4/B8A | B4/B3/B2 + ドメイン依存 |
| **GT ソース** | NASA FIRMS VIIRS | **FIRMS 座標 × 発火 14〜28日前の SimSat 画像** | AIS データ + SimSat画像照合 | AIS 船舶動静 + SimSat 在庫変化の照合 | Marine Cadastre 歴史 AIS + IEA/METI 輸入統計（事後相関） | Copernicus EMS + 中小規模洪水は OSM 道路閉鎖データ | NOAA / GBRMPA 2024年白化データ | FIRMS + Copernicus EMS（共有） |
| **Early Adopter** | 衛星オペレーター・民間林業会社 | **P&C 損害保険会社（Swiss Re・Tokio Marine 等）** | 制裁コンプライアンス SaaS・トレードファイナンス銀行 | シンガポール系トレードファイナンス銀行 | トレードファイナンス銀行・代替データプロバイダー | 自然災害保険会社（査定部門） | GBRMPA | DPhi Space / 小型コンステレーション |
| **ストレージ活用** | アラートキュー（~10MB） | **NBR/NDVI 時系列統計（~1.5MB/シーン）+ Few-shot参照画像** | AOI ベースライン画像（T0参照） | **T0 タンク画像（差分検知の基準、~33MB）** | AOI ベースライン + 入港頻度時系列 | 洪水前基準道路画像 | 礁ごとのB3/B4基準値 + Few-shot参照 | **Few-shot参照画像（4クラス×5枚 ~18MB）** |
| **ダウンリンク形式** | JSON 2KB + SWIR thumbnail（任意 ~50KB） | GeoJSON リスクマップ（~10〜50KB） | JSON 2KB + RGB thumbnail（任意 ~50KB） | JSON 2KB（在庫変動アラート） | JSON + 入港頻度時系列（~5〜10KB） | GeoJSON リスクマップ（~10〜50KB） | GeoJSON 白化マスク（~10〜50KB） | JSON 優先度リスト（~2KB） |
| **Critic スコア** | **69点** | **65点** | **67点** | **63点** | **63点** | **58点** | **57点** | **62点** |
| **Critic 判定** | **Go** | **Go** | **Go** | **Go** | **Go** | **Go** | **Go** | **Go** |
| **フェーズ** | **/poc2 完了** | **/ideate 完了** | /ideate 完了 | /ideate 完了 | /ideate 完了 | /ideate 完了 | /ideate 完了 | /ideate 完了 |

### FireGuard Critic 変遷
- 初回評価: **62点・Go（条件付き）**（LFM vs RF の差別化不明・デモ地域が PoC リスク高・直接競合未記載）
- 修正後: **65点・Go**（LFM 役割を時系列パターン認識に特化・デモをカリフォルニアに変更・Planet Analytics / Descartes Labs との比較追加）

### DarkFleet Critic 変遷
- 初回（修正前）: 62点推定・**要修正**（解像度制約で漁船対象不可、リビジット問題）
- 修正後（lean_canvas 反映済み）: **67点・Go**（スコープを港湾静止大型タンカーに絞り、両致命的弱点を解消）
- 残課題（PoC フェーズで検証）: 船種分類の解像度限界、FT 用 GT データ収集、AIS 外部依存

### OrbitTriage Critic 変遷
- 初回: 53点・**要修正**（デモ不可視・Early Adopter 曖昧・GT 未定義・φ-sat 差別化不足）
- 修正後: **62点・Go**（4点すべて lean_canvas に反映済み）
- 主な修正: DPhi Space を Early Adopter #1 に設定・φ-sat 帯域削減率 25% vs 80% の定量比較追加・GT を FireEdge+InfraWatch と共有する設計明記・デモシナリオ固定

### FireEdge Critic 変遷
- 正式評価: **69点・Go**（全案トップ）
- 残課題（/finetune 対処中）: FPR 目標 0.15 未達（現状 0.244〜0.300）。active fire only + SWIR+RGB 2枚で解決見込み

### HarborLens Critic 変遷
- 初回評価: **60点・Go（条件付き）**（自己 GT 循環・LFM 役割曖昧・荷役活動の 10m 問題）
- 修正後: **63点・Go**（UVP を floating roof 油タンク特化・AIS 外部 GT・LFM を文脈解釈に明記）

### InfraWatch Critic 変遷
- 初回評価: **53点・要修正**（光学センサーの雲問題で速度 UVP 崩壊・EMS 差別化なし・LFM 役割不明）
- 修正後: **58点・Go**（UVP を保険査定特化・EMS 差別化追加・LFM を severity 分類に特化）

### CoralShield Critic 変遷
- 初回評価: **52点・要修正**（水柱補正リスク・LFM 役割不明・リアルタイム主張が雲で崩れる）
- 修正後: **57点・Go**（UVP 修正・PoC 第一ゲート追加・LFM を false positive フィルタに特化）

### PortPulse Critic 変遷
- 初回評価: **57点・要修正**（コンテナ密度 10m 不可・LFM 役割曖昧・オンボード必然性なし）
- /ideate 完了後: **63点・Go**（タンカー入港頻度特化・LFM 文脈解釈・追加撮像リクエスト中継を設計）
- Early Adopter: 日系総合商社エネルギートレーディング部門（三菱・三井・住友）
- デモシナリオ固定: 2022年3月ロシア LNG 制裁時の京葉シーバース等の入港頻度変化（IEA 統計より 21 日前）

---

## No-Go 判定（技術的に実現不可 or 既存案と重複）

| | **停電地域把握** | **津波検知** | **降雪領域観測** | **火山噴火危険度** | **ヌーの群れ移動監視** | **鉄道貨物車両監視** |
|---|---|---|---|---|---|---|
| **ドメイン** | 停電範囲の衛星検知 | 津波発生・進行状況の検知 | 山岳積雪範囲の時系列変化 | 噴煙量から噴火危険度を定量化 | ヌー移動 + 人工的干渉検知 | 鉄道貨物車両の運用状況把握 |
| **No-Go 理由** | Sentinel-2 は昼間光学センサー。夜間照明データ（VIIRS DNB 等）は取得不可 | ① 津波波高の光学検出は不可能 ② 5〜10日リビジットでは実時間追跡不可 ③ 浸水後の陸上被害評価は InfraWatch と重複 | NDSI `(B3-B11)/(B3+B11)` は教科書的手法。Copernicus HRL Snow & Ice が無償提供済み。LFM が既存製品を超える付加価値を示せない | SO₂/火山ガス検出は UV センサー（Sentinel-5P）が必要で Sentinel-2 光学では不可。熱異常検出は MIROVA / MODVOLC が商用展開済み。噴火予測誤報の法的責任問題 | ヌー個体は全長 ~1.5m。10m/pixel では個体検出不可（間接的な植生変化でしか測れない）。市場は保全 NGO に限定で支払い意欲が低い | 標準貨車 ~15m → 10m/pixel で 1〜2px。安定した検出が不可能。学習データセットも業界全体で不足 |
| **代替案** | VIIRS DNB / DMSP-OLS センサーを持つ別衛星プラットフォームで再検討 | 事後的な海岸線変化・浸水域検出は InfraWatch に統合可能 | 水資源管理の文脈でSARとの組み合わせを検討するか、LFM なしのパイプラインで再設計 | 溶岩流・火砕流の流下方向予測に絞り、事後対応（避難支援）に限定すれば成立可能性あり | 植生 NDVI 変化から生態系圧力を測る方向に変換。人工干渉 vs 気候変動の分離が研究課題として成立 | Planet Labs の 3m 解像度なら 15m 貨車 = 5px で検出可能。SimSat（Sentinel-2）では不可 |

---

## バンド選定の根拠

sentinel2_guide.md §10 の用途別バンド組み合わせに基づく。

| アプリ | バンド | 用途 | SimSat 名 |
|---|---|---|---|
| FireEdge | B12, B11, B8 | 疑似カラー合成 (R=SWIR22, G=SWIR16, B=NIR)。burn scar が暗い赤〜茶色に見える | `swir22, swir16, nir` |
| FireGuard | B8A, B12, B8, B4, B11 | NBR=(B8A-B12)/(B8A+B12) 植生含水率、NDVI=(B8-B4)/(B8+B4) 植生活性度、B11 水分ストレス。3〜5パスの時系列統計をストレージにキャッシュ | `nir08, swir22, nir, red, swir16` |
| DarkFleet | B4, B3, B2 | 10m RGB。船舶の形状・輝度を最高解像度で捉える | `red, green, blue` |
| PortPulse | B4, B3, B2 | 10m RGB。大型タンカー（全長 100m+ = 10px+）の入港頻度カウント | `red, green, blue` |
| InfraWatch | B4, B3, B8, B11, B12 | RGB + 浸水域 (NDWI) + 植生変化 (NDVI) + 構造物損傷 (SWIR) | `red, green, nir, swir16, swir22` |
| CoralShield | B2, B3, B4, B8A | 水中透過が最も高い可視〜NIR。白化は可視 RGB + 水深補正に有効 | `blue, green, red, nir08` |
| HarborLens | B2〜B12 全般 | タンク屋根の影・反射から容量を推定。SWIR で金属反射の識別 | 複数バンド |
| OrbitTriage | B4/B3/B2 + イベント依存 | シーン分類。クライシスの種別によりバンドを切り替え | 複数バンド |
