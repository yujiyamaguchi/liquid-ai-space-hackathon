# アプリ案比較表

## 進行中・検討中

| | **FireEdge** | **DarkFleet** | **HarborLens** | **InfraWatch** | **CoralShield** | **OrbitTriage** |
|---|---|---|---|---|---|---|
| **ドメイン** | 野火検知 | ダーク船舶追跡 | 港湾タンク在庫監視 | 洪水後インフラ損傷 | サンゴ礁白化 | 観測優先度トリアージ |
| **Lean Canvas** | [fireedge/docs/](fireedge/docs/lean_canvas.md) | [darkfleet/docs/](darkfleet/docs/lean_canvas.md) | [harborlens/docs/](harborlens/docs/lean_canvas.md) | [infrawatch/docs/](infrawatch/docs/lean_canvas.md) | [coralshield/docs/](coralshield/docs/lean_canvas.md) | [orbittriage/docs/](orbittriage/docs/lean_canvas.md) |
| **使用バンド** | B12/B11/B8 (SWIR22・SWIR16・NIR) | B4/B3/B2 (RGB 10m) | B2〜B12 全般 | B4/B3/B8/B11/B12 | B2/B3/B4/B8A | B4/B3/B2 + ドメイン依存 |
| **GT ソース** | NASA FIRMS VIIRS | AIS データ + SimSat画像照合 | SimSat 時系列差分（自己 GT） | Copernicus EMS | NOAA / GBRMPA 2024年白化データ | FIRMS + Copernicus EMS（共有） |
| **Early Adopter** | 衛星オペレーター・民間林業会社 | 制裁コンプライアンス SaaS・トレードファイナンス銀行 | シンガポール系トレードファイナンス銀行 | 自然災害保険会社（査定部門） | GBRMPA | DPhi Space / 小型コンステレーション |
| **Critic スコア** | **68点** | **62点（推定）** | 60点 | 53点 | 51点 | 52点（推定） |
| **Critic 判定** | — | **要修正** | — | — | — | **要修正** |
| **フェーズ** | **/poc2 完了** | /ideate 完了 | /ideate 完了 | /ideate 完了 | /ideate 完了 | /ideate 完了 |

## No-Go 判定（技術的に実現不可 or 既存案と重複）

| | **停電地域把握** | **津波検知** |
|---|---|---|
| **ドメイン** | 停電範囲の衛星検知 | 津波発生・進行状況の検知 |
| **No-Go 理由** | Sentinel-2 は昼間光学センサー。夜間照明データ（VIIRS DNB 等）は取得不可。SimSat API では実現不可能 | ① 津波波高の光学検出は不可能 ② 5〜10日リビジットでは実時間追跡不可 ③ 浸水後の陸上被害評価は InfraWatch と重複 |
| **代替案** | VIIRS DNB / DMSP-OLS センサーを持つ別衛星プラットフォームで再検討 | 事後的な海岸線変化・浸水域検出は InfraWatch に統合可能 |

---

## バンド選定の根拠

sentinel2_guide.md §9 の用途別バンド組み合わせに基づく。

| アプリ | バンド | 用途 | SimSat 名 |
|---|---|---|---|
| FireEdge | B12, B11, B8 | 疑似カラー合成 (R=SWIR22, G=SWIR16, B=NIR)。burn scar が暗い赤〜茶色に見える | `swir22, swir16, nir` |
| DarkFleet | B4, B3, B2 | 10m RGB。船舶の形状・輝度を最高解像度で捉える | `red, green, blue` |
| InfraWatch | B4, B3, B8, B11, B12 | RGB + 浸水域 (NDWI) + 植生変化 (NDVI) + 構造物損傷 (SWIR) | `red, green, nir, swir16, swir22` |
| CoralShield | B2, B3, B4, B8A | 水中透過が最も高い可視〜NIR。白化は可視 RGB + 水深補正に有効 | `blue, green, red, nir08` |
| HarborLens | B2〜B12 全般 | タンク屋根の影・反射から容量を推定。SWIR で金属反射の識別 | 複数バンド |
| OrbitTriage | B4/B3/B2 + イベント依存 | シーン分類。クライシスの種別によりバンドを切り替え | 複数バンド |
