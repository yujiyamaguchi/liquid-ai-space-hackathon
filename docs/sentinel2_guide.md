# Sentinel-2 ガイド — ミッション概要・バンド仕様・SimSat との関係

> **出典**: 特記なき場合 ESA/Copernicus 公式ドキュメント  
> [Sentinel-2 Mission Overview](https://sentiwiki.copernicus.eu/web/s2-mission) | [Band Combinations](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel/sentinel-2/)

---

## 1. Sentinel-2 とは

Sentinel-2 は欧州宇宙機関 (ESA) と欧州委員会が運営する **Copernicus プログラム**の地球観測ミッション。  
現在 3 機の衛星 (Sentinel-2A / 2B / 2C) が同一軌道面を異なる位相で周回しており、地球全体を定期的にカバーしている。

- **軌道高度**: 786 km (太陽同期軌道)
- **軌道傾斜角**: 98.6°（全緯度カバー）
- **データアクセス**: 完全無償公開 — Copernicus Data Space Ecosystem ([dataspace.copernicus.eu](https://dataspace.copernicus.eu/))

出典: ESA Sentinel-2 Mission — [sentiwiki.copernicus.eu/web/s2-mission](https://sentiwiki.copernicus.eu/web/s2-mission)

---

## 2. 太陽同期軌道と昼間専用観測（重要）

Sentinel-2 が採用する**太陽同期軌道 (Sun-Synchronous Orbit, SSO)** とは、軌道面が地球の公転に合わせて年間 360° 回転し、衛星が**常に太陽と同じ相対角度を保ちながら**通過するよう軌道傾斜を調整した軌道である。

**実用上の意味:**

| 項目 | 内容 |
|---|---|
| **地方太陽時の通過時刻** | 地球上どの地点でも **~10:30（降交点・午前）** に通過する |
| **夜間観測** | **不可能**。MSI (MultiSpectral Instrument) は太陽光の反射を測定する受動光学センサーであり、夜間は照明光がないためデータを取得しない |
| **均一な照明条件** | 常に同じ太陽高度角で撮影されるため、時期・場所をまたいだ画像の定量比較が容易 |
| **夜間アーカイブ** | 存在しない。SimSat が返す画像は必ず昼間撮影のものになる |

**他センサーとの比較例:**

VIIRS (FIRMS) は熱赤外チャネルを持つ気象衛星系センサーであり、昼夜を問わず複数回通過して地表温度・熱異常を検出できる（地方時 ~13:30 + 夜間複数回）。Sentinel-2 MSI とは観測原理が異なり、「昼間の高解像度反射光観測」と「昼夜の熱赤外観測」は補完関係にある。アプリケーション設計時は、この観測時間帯の違いを考慮してデータソースを組み合わせること。

---

## 3. 再訪問頻度（同一地点に何日おきに戻るか）

| 衛星構成 | 赤道付近 | 中緯度 (45°N/S付近) |
|---|---|---|
| 1 機 (2A のみ) | 10 日 | 10 日 |
| 2 機 (2A + 2B) | 5 日 | 2〜3 日 |
| **3 機 (2A + 2B + 2C)** | **約 3 日** | **1〜2 日** |

> **2024 年 3 月に Sentinel-2C が打ち上げられ、現在は 3 機体制**。赤道付近でも約 3 日ごとに同一地点の画像が得られる。ただし取得プランや雲の影響で実際の有効画像間隔はこれより長くなることがある。

出典: [Sentinel-2 Revisit and Coverage](https://sentiwiki.copernicus.eu/web/s2-mission#S2Mission-Revisit-and-Coverage)

---

## 4. Near Real Time (NRT) 提供について

Sentinel-2 のデータは撮像後 **3〜4 時間以内** に Copernicus Data Space で公開される (NRT 配信)。

ただし実際のアプリケーションで「NRT」と呼ぶ際の制約を理解しておくこと:

| 制約 | 内容 |
|---|---|
| **再訪問周期** | 同じ地点の次の画像は最短 1〜5 日後（緯度・衛星数による） |
| **雲被覆** | 雲があれば地表が見えない。雨季・高緯度冬季は月単位で雲なし画像が取れないこともある |
| **取得プラン** | 衛星が通過しても必ずしも撮像モードになっているとは限らない（特に洋上） |

> つまり「撮像されてから公開までは NRT (数時間)」だが、「前回の有効な画像からの経過時間は 1〜5 日」になる。  
> 野火検知・洪水検知など緊急性が高いユースケースでは、この再訪問遅延が最大のボトルネックになる。

出典: [Copernicus Data Space — Access](https://dataspace.copernicus.eu/) | [S2 Acquisition Plan](https://sentiwiki.copernicus.eu/web/acquisition-plans)

---

## 5. 処理レベル: L1C と L2A の違い

Sentinel-2 はデータを複数の処理レベルで配布している。

| レベル | 名称 | 内容 | 単位 |
|---|---|---|---|
| **L1C** | Top-of-Atmosphere (TOA) | センサーが測定した値そのまま。大気の散乱・吸収の影響が含まれる | 反射率 (TOA) |
| **L2A** | Surface Reflectance (SR) | ESA の Sen2Cor アルゴリズムで大気補正済み。地表面が実際に反射している割合 | 反射率 (SR) |

**SimSat は L2A を使用している** (`sentinel_provider.py` が AWS Element84 STAC の `sentinel-2-l2a` コレクションにアクセスしている)。  
L2A を使う理由: 大気補正済みなので、異なる日時・場所・季節の画像を定量的に比較できる。スペクトル指標 (NDVI, NBR など) の計算も L2A ベースが標準。

出典: [Sen2Cor Processor](https://step.esa.int/main/snap-supported-plugins/sen2cor/) | Müller-Wilm et al. (2018), doi:[10.3390/rs10030462](https://doi.org/10.3390/rs10030462)

---

## 6. SimSat と Sentinel-2 の関係

**SimSat が返すデータは実際の Sentinel-2 L2A データと同一**。SimSat は「衛星搭載 AI」シナリオをシミュレートするフレームワークであり、内部では AWS が公開する Element84 STAC API ([earth-search.aws.element84.com/v1](https://earth-search.aws.element84.com/v1)) から実データを取得している。

| 項目 | Sentinel-2 直接アクセス | SimSat 経由 |
|---|---|---|
| データの中身 | Sentinel-2 L2A 実データ | **同じ** Sentinel-2 L2A 実データ |
| アクセス方法 | STAC API / Copernicus Data Space | `GET /data/image/sentinel?lon=...&lat=...` |
| 解像度 | バンドごとに 10 / 20 / 60 m | **全バンド 10 m に統一**（粗いバンドはアップサンプル） |
| 付加情報 | — | 軌道シミュレーションと連動した衛星位置・可視判定 |

SimSat 固有の制約として、`size_km=20` 以上のクエリはタイル境界で大量の nodata が発生することがある。実用上は `size_km=5` が安定している（[`sentinel_provider.py`](../../../SimSat/src/sim/ImagingProviders/sentinel_provider.py) にてタイルを単一 item で取得しているため）。

### `window_seconds` の動作（後方サーチ・最新シーン選択）

出典: [`sentinel_provider.py`](../simsat/src/sim/ImagingProviders/sentinel_provider.py)（SimSat ソースコード）

`/data/image/sentinel` の `window_seconds` は、`timestamp` を終端とする **後方向のみ** の検索ウィンドウを構成する。

```
検索範囲: [timestamp - window_seconds, timestamp]
```

ウィンドウ内で `eo:cloud_cover < 100`（完全曇り以外）の候補を収集し、その中から **最新（timestamp に最も近い）シーン**を返す。

```
timestamp=2025-03-06T12:00:00Z, window_seconds=1036800 (12日)
→ 検索範囲: 2025-02-22 〜 2025-03-06
→ 返却された capture: 2025-03-05T01:14:35Z
```

`timestamp` の直前に撮像されたシーンが返るため、**capture が timestamp より古くなることは仕様通り**。  
「特定日時以降のシーンのみ取得したい」場合はアプリ側で capture datetime を確認して後処理する必要がある。

---

## 7. スペクトルバンド一覧

Sentinel-2 は 13 バンドのマルチスペクトル画像を提供する。

**SimSat で指定できるバンド名**:  
`aot, blue, coastal, green, nir, nir08, nir09, red, rededge1, rededge2, rededge3, scl, swir16, swir22, visual, wvp`

| SimSat 名 | バンド | 波長名称 | 中心波長 | ネイティブ解像度 | 主な観測内容 |
|---|---|---|---|---|---|
| `coastal` | B1 | Coastal Aerosol | 443 nm | 60 m | 大気エアロゾル・沿岸浅水域 |
| `blue` | B2 | Blue (青) | 490 nm | **10 m** | 可視 RGB の青。水体識別 |
| `green` | B3 | Green (緑) | 560 nm | **10 m** | 植生反射ピーク・クロロフィル |
| `red` | B4 | Red (赤) | 665 nm | **10 m** | 葉緑素吸収・NDVI の分母 |
| `rededge1` | B5 | Red Edge 1 | 705 nm | 20 m | 植物ストレス・クロロフィル量 |
| `rededge2` | B6 | Red Edge 2 | 740 nm | 20 m | 葉面積指数 (LAI) 推定 |
| `rededge3` | B7 | Red Edge 3 | 783 nm | 20 m | 植生構造・密度 |
| `nir` | B8 | NIR (近赤外) | 842 nm | **10 m** | 植生強反射・水体吸収 |
| `nir08` | B8A | Narrow NIR | 865 nm | 20 m | 帯域幅が狭く水蒸気の影響が少ない精密 NIR |
| `nir09` | B9 | Water Vapour NIR | 945 nm | 60 m | 大気水蒸気量（地表観測には使わない） |
| *(非対応)* | B10 | Cirrus | 1375 nm | 60 m | **L2A には含まれない**（大気補正で除去済み） |
| `swir16` | B11 | SWIR 1 | 1610 nm | 20 m | 土壌水分・植生含水量・雪氷識別 |
| `swir22` | B12 | SWIR 2 | 2190 nm | 20 m | 活火災熱放射・地質鉱物マッピング |

> B10 は L1C にのみ存在し、L2A の大気補正（巻雲除去）に使われたあと除外される。SimSat (L2A) では取得不可。

---

## 8. 派生プロダクト

「派生プロダクト」とは、生の分光データではなく、**ESA の Sen2Cor 大気補正アルゴリズムが L2A 処理の過程で計算・付属させるデータ**。  
Sentinel-2 L2A の一部として配布されており、SimSat でも同じバンド名で取得できる。

| SimSat 名 | 正式名称 | 内容 | 何者が生成するか |
|---|---|---|---|
| `scl` | Scene Classification Layer | ピクセルごとに「雲 / 雲影 / 水 / 植生 / 裸地 / 雪 / 欠損」を分類したマスク (11 クラス) | ESA Sen2Cor |
| `aot` | Aerosol Optical Thickness | 大気中のエアロゾル光学厚さ（無次元）。大気透過率の指標 | ESA Sen2Cor |
| `wvp` | Water Vapour Product | 大気中の水蒸気カラム量 (cm) | ESA Sen2Cor |
| `visual` | True Color Image (TCI) | B2 / B3 / B4 を RGB 合成した標準可視画像。L2A に同梱される | ESA |

これらは生バンドと同じように SimSat の `spectral_bands` パラメータで指定して取得できる。  
例: `SCL` は前処理での雲除外フィルタとして特に有用。

---

## 9. バンド別の詳細と応用例

### B1 — Coastal Aerosol (443 nm, 60 m)

**観測原理**: 短い波長の青色光は大気中で強く散乱する (Rayleigh 散乱)。この散乱が逆に大気エアロゾルの指標になる。また水中での減衰が最も少ない波長帯なので沿岸の底質・水深が透けて見える。

**応用例**:
- 大気補正の品質確認 (Dark Dense Vegetation 法の補助)
- 沿岸・浅瀬の底質・藻類マッピング
- 水質（濁度・クロロフィル）の広域モニタリング

---

### B2 / B3 / B4 — Blue / Green / Red (490 / 560 / 665 nm, 10 m)

**観測原理**: 可視光 3 バンド。RGB 合成で人間の目に近い「真色」画像が得られる。植生は葉緑素が赤 (665 nm) を吸収するため、Red が低く Green が高くなる（緑色に見える理由）。

**代表指標**:
```
NDVI  = (NIR − Red) / (NIR + Red)        # 植生密度 (Tucker 1979)
NDWI  = (Green − NIR) / (Green + NIR)    # 水体抽出 (McFeeters 1996)
```

**応用例**:
- 土地被覆・都市域マッピング
- 洪水・河川増水域の抽出 (NDWI)
- 農地の作付け状況・植生密度

出典: Tucker (1979), doi:[10.1016/0034-4257(79)90013-0](https://doi.org/10.1016/0034-4257(79)90013-0) | McFeeters (1996), doi:[10.1080/01431169608948714](https://doi.org/10.1080/01431169608948714)

---

### B5 / B6 / B7 — Red Edge 1/2/3 (705 / 740 / 783 nm, 20 m)

**観測原理**: 植生の分光反射率は 680〜750 nm で急激に上昇する（Red Edge と呼ばれる）。この遷移の形状・位置が葉緑素含量・LAI・植物ストレスと強く相関する。Landsat や MODIS にはないため Sentinel-2 の最大の差別化バンド群。

**代表指標**:
```
CIre  = (NIR / RedEdge1) − 1             # 葉緑素インデックス (Gitelson 2003)
NDre  = (B7 − B5) / (B7 + B5)
```

**応用例**:
- 精密農業：作物ストレス・窒素含量の可変施肥
- 森林健全性モニタリング（病虫害の早期検知）
- 植生バイオマス・LAI の高精度推定

出典: Frampton et al. (2013), doi:[10.1016/j.rse.2012.06.018](https://doi.org/10.1016/j.rse.2012.06.018) | Delegido et al. (2011), doi:[10.3390/rs3091829](https://doi.org/10.3390/rs3091829)

---

### B8 — NIR (842 nm, 10 m)

**観測原理**: 植物の葉肉細胞（柵状組織・海綿状組織）の内部構造が NIR を強く散乱反射する。水体は NIR をほぼ完全に吸収するため、水陸境界・湿地の識別に非常に有効。10 m 解像度を持つ主力バンド。

**代表指標**:
```
NDVI  = (B8 − B4) / (B8 + B4)
SAVI  = [(B8 − B4) / (B8 + B4 + L)] × (1 + L)    # 土壌補正植生指数 (L=0.5)
EVI   = 2.5 × (B8 − B4) / (B8 + 6×B4 − 7.5×B2 + 1)
```

**応用例**:
- 農作物の生育モニタリング・収量予測
- 砂漠化・緑化の長期トレンド監視
- 河川・湖沼・洪水域の水面抽出

出典: Tucker (1979), doi:[10.1016/0034-4257(79)90013-0](https://doi.org/10.1016/0034-4257(79)90013-0)

---

### B8A — Narrow NIR (865 nm, 20 m)

**観測原理**: B8 に隣接するが帯域幅が狭い (20 nm vs B8 の 115 nm)。大気水蒸気の吸収帯 (~940 nm) から離れているため、水蒸気量に依存しない安定した NIR 値が得られる。B12 との組み合わせで活火災の熱異常を検出するのに有効。

**代表指標**:
```
TAI   = (B12 − B11) / B8A               # Thermal Anomaly Index — 活火災
NBR   = (B8A − B12) / (B8A + B12)       # Normalized Burn Ratio
dNBR  = NBR_pre − NBR_post              # 焼跡深刻度 (≥0.1 で軽度焼失)
```

**応用例**:
- 活火災の熱異常検知 (TAI)
- 焼跡の深刻度マッピング (dNBR)
- 水蒸気の影響を除いた精密な植生指数

出典: Filipponi (2019), doi:[10.3390/rs11010053](https://doi.org/10.3390/rs11010053) | Key & Benson (2006) *FIREMON* (NBR 提唱)

---

### B9 — Water Vapour NIR (945 nm, 60 m)

**観測原理**: 大気水蒸気の強い吸収帯に位置し、地表面からの反射はほぼゼロ。水蒸気カラム量の推定と大気補正 (Sen2Cor) に専用。地表面観測には直接使用しない。

**応用例**:
- ESA Sen2Cor における大気水蒸気量の推定（L2A 生成の内部処理）
- (地表向けアプリでは通常不使用)

---

### B11 — SWIR 1 (1610 nm, 20 m)

**観測原理**: 液体水は 1.4〜1.8μm に強い吸収帯を持つため、土壌水分・植生の含水量がこのバンドで低い反射率として現れる。雪は 1.6μm で吸収するのに対し雲は反射するため、**雪と雲の識別**が可能（可視光では両者が白く区別できない）。

**代表指標**:
```
NDSI  = (Green − SWIR16) / (Green + SWIR16)     # 雪氷指数 (Dozier 1989)
NDWI2 = (B8A − B11) / (B8A + B11)              # 植生含水量指数 (Gao 1996)
NBR2  = (B11 − B12) / (B11 + B12)              # 焼跡・活火災シグナル
```

**応用例**:
- 山岳域の積雪範囲・雪水当量の推定 (NDSI)
- 乾燥ストレス・灌漑農業の水管理 (NDWI2)
- 焼跡検知 (NBR2)
- 山腹の雪崩危険度評価

出典: Dozier (1989), doi:[10.1016/0034-4257(89)90026-2](https://doi.org/10.1016/0034-4257(89)90026-2) | Gao (1996), doi:[10.1016/S0034-4257(96)00067-3](https://doi.org/10.1016/S0034-4257(96)00067-3)

---

### B12 — SWIR 2 (2190 nm, 20 m)

**観測原理**: 活火災の炎温度 (600〜1200 K) が 2.2μm 付近にプランク曲線のピークを持つため、センサーが飽和するほど強い熱放射を検出できる。また粘土鉱物・炭酸塩鉱物が 2.0〜2.5μm に特徴的な吸収帯を持つため地質マッピングにも有効。

**代表指標**:
```
TAI   = (B12 − B11) / B8A               # Thermal Anomaly Index — 活火災
NBR2  = (B11 − B12) / (B11 + B12)       # 焼跡・活火災 (負が深いほど顕著)
```

**応用例**:
- 活火災のリアルタイム検知（SWIR22 の熱放射飽和を検出）
- 焼跡範囲・深刻度マッピング
- 乾燥地帯の地質・鉱物資源調査
- 土壌有機炭素含量の推定

出典: Schroeder et al. (2014), doi:[10.1016/j.rse.2014.01.028](https://doi.org/10.1016/j.rse.2014.01.028) | Filipponi (2019), doi:[10.3390/rs11010053](https://doi.org/10.3390/rs11010053) | Murphy et al. (2021), doi:[10.3390/rs13091798](https://doi.org/10.3390/rs13091798)

---

## 10. 用途別バンド組み合わせ早見表

| 用途 | 使用バンド (SimSat 名) | 指標・合成方法 |
|---|---|---|
| 真色合成 (RGB) | `red, green, blue` | そのまま R/G/B |
| 植生モニタリング | `nir, red` | NDVI = (NIR−Red)/(NIR+Red) |
| 水体抽出 | `green, nir` | NDWI = (Green−NIR)/(Green+NIR) |
| 焼跡検知 | `nir08, swir22` | NBR = (B8A−B12)/(B8A+B12); dNBR で深刻度 |
| 活火災検知 | `swir22, swir16, nir08` | NBR2; TAI = (B12−B11)/B8A |
| 疑似カラー (SWIR) | `swir22, swir16, nir` | R=B12, G=B11, B=B8 → 火災が赤〜橙に見える |
| 雪氷マッピング | `green, swir16` | NDSI = (Green−SWIR16)/(Green+SWIR16) |
| 植生含水量 | `nir08, swir16` | NDWI2 = (B8A−B11)/(B8A+B11) |
| 植生ストレス | `rededge1, rededge3` | NDre = (B7−B5)/(B7+B5) |
| 土地被覆分類 | `swir22, nir, red` | 教師あり分類の基本セット |
| 雲マスク | `scl` | SCL クラス 8/9/10 (雲) を除外 |
| 土壌含水量 | `swir16, swir22` | SWIR1 > SWIR2 なら高含水。比が乾燥の指標 |
| 沿岸水質 | `coastal, blue, green` | 水色 (水体の光学特性) の解析 |
| 地質マッピング | `swir22, swir16` | 粘土鉱物・炭酸塩の吸収特性の差 |
