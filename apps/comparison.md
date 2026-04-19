# アプリ案比較表

| | **FireEdge** | **InfraWatch** | **CoralShield** | **HarborLens** |
|---|---|---|---|---|
| **ドメイン** | 野火検知 | 洪水後インフラ損傷 | サンゴ礁白化 | 港湾タンク在庫監視 |
| **Lean Canvas** | [fireedge/docs/](fireedge/docs/lean_canvas.md) | [infrawatch/docs/](infrawatch/docs/lean_canvas.md) | [coralshield/docs/](coralshield/docs/lean_canvas.md) | [harborlens/docs/](harborlens/docs/lean_canvas.md) |
| **使用バンド** | B12/B11/B8 (SWIR22・SWIR16・NIR) | B4/B3/B8/B11/B12 | B2/B3/B4/B8A | B2〜B12 全般 |
| **GT ソース** | NASA FIRMS VIIRS | Copernicus EMS | NOAA / GBRMPA 2024年白化データ | SimSat 時系列差分（自己 GT） |
| **Early Adopter** | 衛星オペレーター・民間林業会社 | 自然災害保険会社（査定部門） | GBRMPA | シンガポール系トレードファイナンス銀行 |
| **Critic スコア** | **68点** | 53点 | 51点 | 60点 |
| **フェーズ** | **/poc2 完了** | /ideate 完了 | /ideate 完了 | /ideate 完了 |

## バンド選定の根拠

sentinel2_guide.md §9 の用途別バンド組み合わせに基づく。

| アプリ | バンド | 用途 | SimSat 名 |
|---|---|---|---|
| FireEdge | B12, B11, B8 | 疑似カラー合成 (R=SWIR22, G=SWIR16, B=NIR)。burn scar が暗い赤〜茶色に見える | `swir22, swir16, nir` |
| InfraWatch | B4, B3, B8, B11, B12 | RGB + 浸水域 (NDWI) + 植生変化 (NDVI) + 構造物損傷 (SWIR) | `red, green, nir, swir16, swir22` |
| CoralShield | B2, B3, B4, B8A | 水中透過が最も高い可視〜NIR。白化は可視 RGB + 水深補正に有効 | `blue, green, red, nir08` |
| HarborLens | B2〜B12 全般 | タンク屋根の影・反射から容量を推定。SWIR で金属反射の識別 | 複数バンド |
