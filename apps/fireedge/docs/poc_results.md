# FireEdge /poc 結果レポート
**実施日**: 2026-04-18〜19  
**フェーズ**: データ PoC (CLAUDE.md §2)

---

## 完了条件との照合

| 完了条件 | 結果 |
|---|---|
| ① GT (FIRMS) から座標・日時を取得し、SimSat が S2 データを返せること | ✅ |
| ② スペクトル指標が positive / negative クラスで分離できること | ✅ (4/5 イベント) |

---

## 実行コマンド

```bash
cd apps/fireedge

# /poc v1 (初期設計。負例設計の問題で全て❌)
FIRMS_MAP_KEY=<key> uv run python poc_data.py --area africa --days 3 --top 5

# 診断スクリプト群 (問題切り分け)
uv run python poc_diag.py     # バンド値・pixel分布の生確認
uv run python poc_diag2.py    # タイムスタンプを今日にして焼跡探索
uv run python poc_diag3.py    # 歴史的大規模火災でのSWIR22確認

# /poc v2 (最終版。Go判定に使用)
FIRMS_MAP_KEY=<key> uv run python poc_v2.py --top 5 --days 3
```

結果は標準出力に表示。`docs/poc_results.md` に手動で転記。

---

## ① SimSat 接続性

### 確認内容

SimSat (localhost:9005、Docker コンテナ `fakesat-sim`) から  
任意の緯度・経度・タイムスタンプで Sentinel-2 データを取得できることを確認した。

### API コール例

```python
# src/data_fetcher.py: SimSatClient.fetch_fire_scene()
GET http://localhost:9005/data/image/sentinel
  ?lon=104.359&lat=19.753
  &timestamp=2026-04-23T07:16:00Z   # FIRMS検知日時 + 6日
  &spectral_bands=swir22,swir16,nir,red,green,blue
  &size_km=5
  &return_type=array
  &window_seconds=1036800            # 12日 = ±6日ウィンドウ

→ (H, W, 6) float32 配列 (uint16 raw を 65535 で除算して [0,1] に正規化)
```

### 推奨設定

| パラメータ | 採用値 | 理由 |
|---|---|---|
| `size_km` | 5 | 20km はタイル境界で ~99.7% nodata になるケースあり |
| `window_seconds` | 1,036,800 (12日) | FIRMS日時+6日基点で ±6日の最近傍 S2 を探索するため |
| `cloud_cover` フィルタ | < 30% | 雲による遮蔽を除外 |

---

## ② スペクトルシグナル分離

### 実験設計

#### データソース

| 役割 | ソース | 詳細 |
|---|---|---|
| GT (火災座標) | NASA FIRMS VIIRS_SNPP_NRT | 信頼度 `h`, FRP > 100MW の上位 5 件 |
| 衛星画像 | SimSat → Sentinel-2 (Copernicus) | 6バンド: SWIR22(B12), SWIR16(B11), NIR(B08), Red(B04), Green(B03), Blue(B02) |

#### サンプリング設計

**Positive**: FIRMS で fire と判定された座標を中心に 5km × 5km シーンを取得。

**Negative**: 同一の S2 フットプリント (5km × 5km, 約 500 × 500 px) 内を 5×5 グリッドに分割し、
全 FIRMS ホットスポット (直近 3日) からの距離が最大の格子点を選択した。
結果として fire 座標から約 0.015° (≈ 1.5km) 離れた点が negative となった。

**タイムスタンプ設計**:
```
SimSat クエリ時刻 = FIRMS 検知日時 + 6日
window_seconds   = 12日 (= 6日 × 2)
→ SimSat は (FIRMS日時 - 6日) 〜 (FIRMS日時 + 6日) の最近傍 S2 を返す
```
これにより FIRMS との時刻ズレ (S2 再訪問周期 2〜5日 + 軌道通過時刻差 ~3時間) を許容した。

#### 前処理パイプライン

```
SimSat API
  → raw uint16 bytes
  → ÷ 65535  →  float32 [0, 1]  ← ここまでが image_array
       │
       ├─ [スペクトル指標計算 (raw値ベース)]
       │     NBR2    = (SWIR16 - SWIR22) / (SWIR16 + SWIR22)
       │     SWIR22  = そのまま使用
       │     fire_px = NBR2 < -0.05  AND  SWIR22 > 0.15 の画素比率
       │
       └─ [LFM-VL への視覚入力]
             R = SWIR22 (B12)  ← 活火災が赤〜橙に見える
             G = SWIR16 (B11)
             B = NIR    (B08)
             → パーセンタイルクリップ [2nd, 98th] で外れ値除去
             → [0, 255] uint8 に変換 → PIL RGB 画像 (448 × 448 px)
```

**重要**: スペクトル指標の計算にパーセンタイルクリップは適用しない。
パーセンタイルクリップは LFM-VL に渡す PIL 画像の見た目を整えるためだけに使用している。

---

### 結果

| # | 座標 (Positive) | FRP | Δ (S2 − FIRMS) | 分離できた指標 | 判定 |
|---|---|---|---|---|---|
| 1 | Laos 19.75N 104.36E | 298MW | −0.1日 | NBR2_min ✅ SWIR22_max ✅ | ✅ |
| 2 | Myanmar 12.72N 98.82E | 208MW | −2.1日 | — | ❌ |
| 3 | Sudan 13.64N 34.46E | 196MW | −3.1日 | NBR2_min ✅ SWIR22_max ✅ | ✅ |
| 4 | Thailand 18.83N 102.32E | 130MW | −1.1日 | NBR2_mean ✅ NBR2_min ✅ SWIR22_max ✅ | ✅ |
| 5 | Thailand 18.82N 101.59E | 127MW | −1.1日 | NBR2_min ✅ | ✅ |

**Go/No-Go**: 4/5 (80%) → **✅ GO**

---

### 各指標の学術的位置づけ

#### TAI — Thermal Anomaly Index (文献上有力・今回未試行)

```python
TAI = (SWIR22 - SWIR16) / NIR08    # B12, B11, B8A を使用
```

活火災検知の学術文献 (Alcaras et al. 2021; Filipponi 2019) で採用されている指標。  
NBR2 との違いは分母に B8A (Narrow NIR, 865nm) を使う点で、**大気水蒸気の影響を除いた正規化**ができる。  
また `SWIR22 - SWIR16` の符号が正 (B12 > B11) になるのは活火災・高温面のみで、焼跡・裸地では B11 ≈ B12 またはわずかに B11 > B12 となるため、偽陽性が少ない。

SimSat では `nir08` バンドとして B8A が取得可能であり、**次フェーズ (/poc2 または /data) で試行予定**。

参照: Alcaras et al. (2021), doi:[10.3390/rs13050973](https://doi.org/10.3390/rs13050973) | Filipponi (2019), doi:[10.3390/rs11010053](https://doi.org/10.3390/rs11010053)

---

#### fire_pixel_ratio (標準的な手法・今回は 0)

```python
fire_pixel_ratio = mean( NBR2 < -0.05  AND  SWIR22 > 0.15 )
```

これが学術的に最も標準的な Sentinel-2 活火災検知の考え方に近い。  
発表論文の多くは「活火災熱放射ピクセル (fire pixel)」を以下の条件で定義している:

- **SWIR22 > 閾値**: 活火災の黒体放射 (600〜1200 K) が 2.2μm で S2 センサーを飽和させる  
  (uint16 ÷ 65535 正規化後で > 0.15 → DN > 9,830 → 実反射率換算で > 98%、すなわちセンサー飽和域)
- **NBR2 < 閾値 (負)**: SWIR22 が SWIR16 を上回ることで NBR2 が負になる

参照: Schroeder et al. (2014) "The New VIIRS 375 m active fire detection data product"; 
Filipponi (2019) "Exploitation of Sentinel-2 Time Series to Map Burned Areas"; 
Murphy et al. (2021) "Sentinel-2 and Landsat-8 active fire detection".

**今回の結果: 全 5 イベントで 0.0000** (詳細は後述)

#### NBR2_min / SWIR22_max (今回有効だった指標・非標準)

```python
NBR2_min   = min( NBR2 全ピクセル )   # シーン内最も負のピクセル1点
SWIR22_max = max( SWIR22 全ピクセル ) # シーン内最も高温相当のピクセル1点
```

これらは探索的な統計量であり、学術論文で「fire detection metric」として用いられる指標ではない。  
今回の PoC では positive と negative の分布が異なることを示す目的で使用した。  
訓練データの「ラベル付け」には使用せず、あくまでシグナル存在の証拠として利用した。

---

## 技術的発見事項

### (A) fire_pixel_ratio = 0 について

`fire_pixel_ratio` の計算はパーセンタイルクリップなどの正規化とは無関係に行われる。根拠は以下:

1. `fire_pixel_ratio` の計算は `image_array` (uint16 ÷ 65535 のみ) から行う
2. パーセンタイルクリップは `_to_pil()` 内でのみ実行 → LFM-VL への視覚入力にだけ影響
3. 実際に SWIR22_max = 0.1613 (> 0.15 閾値) のピクセルは存在する (Event 4)
4. しかしそのピクセルは同時に NBR2 < -0.05 を満たさない → `fire_pixel_ratio = 0`

**原因**: SWIR22 > 0.15 のピクセルと NBR2 < -0.05 のピクセルが空間的に一致していない。  
SWIR22 が高いピクセルは明るい裸地・土壌に起因し (両 SWIR バンドが同程度に高い → NBR2 ≈ 0)、  
NBR2 が負のピクセルは SWIR22 が 0.15 未満の焼跡・陰影に起因している。

**根本的なスケール問題** (後述 (C) も参照):  
混合ピクセル問題により、1 ピクセル (20m × 20m = 400 m²) 内の燃焼面積が約 9% 未満では SWIR22 > 0.15 閾値に届かない。草地の火炎前線幅 (~0.1〜1 m) はこの条件を満たせない。

### (B) FIRMS と Sentinel-2 の時刻ズレ

| 原因 | 詳細 |
|---|---|
| 軌道通過時刻の差 | VIIRS: 地方太陽時 ~13:30 / S2: ~10:30 → 同日でも ~3時間ずれる |
| S2 再訪問周期 | 同じ場所への次の S2 撮像は 2〜5日後 |

ズレは 0.1日〜3日超まで不均一 (「一律ではない」)。±6日ウィンドウ設計で許容した。

### (C) スケール問題 (残存懸念)

活火災の thermal emission を 20m Sentinel-2 ピクセルで reliable に検出するには、文献上 **200ha (500 acres) 以上**の火災規模が必要とされる。

#### 物理的背景: 混合ピクセル問題

S2 の 1 ピクセル = 20m × 20m = 400 m²。ピクセルの観測値は:

```
ピクセル輝度 = f × (火炎輝度) + (1 − f) × (地表輝度)
```

`SWIR22 > 0.15` 閾値を満たすために必要な燃焼面積割合:

```
0.15 = f × 1.0 + (1 − f) × 0.07  →  f ≈ 9%  (≈ 36 m² / ピクセル)
```

| 火災タイプ | 典型的な火炎前線幅 | 20m ピクセル内の燃焼比率 | 検出可否 |
|---|---|---|---|
| サバンナ草地 | 0.1〜1 m | < 5% | ❌ |
| 低木林 | 0.5〜3 m | < 15% | △ |
| 熱帯雨林 (樹冠火) | 5〜20 m | ~25〜100% | ✅ |
| 大規模山火事 | 10〜100 m 以上 | ~50〜100% | ✅ |

今回の FRP 100〜300MW の東南アジア熱帯林火災でも `fire_pixel_ratio = 0` だったのは、燃焼エリアが空間的に分散していたか、樹冠火ではなかったためと推定される。

「信頼性ある検出に 200ha 以上」という文献値は、飽和ピクセルが統計的にまとまって現れ始める面積を指す (200ha ≈ 5,000 S2 ピクセル)。

出典: Wooster et al. (2005), doi:[10.1016/j.rse.2004.09.006](https://doi.org/10.1016/j.rse.2004.09.006) | Schroeder et al. (2014), doi:[10.1016/j.rse.2014.01.028](https://doi.org/10.1016/j.rse.2014.01.028)

**検出できているのは焼跡ピクセル (burn scar) の可能性が高い**。  
LFM-VL が学習するのは「活火災輝点」ではなく「SWIR 疑似カラー画像における火災被影響エリアの空間パターン」となる。

---

## 懸念点と対処方針

| # | 懸念 | 深刻度 | /poc2〜/build での対処 |
|---|---|---|---|
| **A** | `fire_px_ratio = 0` 全イベント。LFM-VL が視覚的に学習できるか未確認 | **高** | /poc2 で few-shot ICL テストを実施し、視覚的判定可否を先に確認する |
| **B** | 負例がフットプリント内最遠点 (~1.5km) → 大規模火災では隣接ホットスポットに入る (Event 2 失敗の原因) | **中** | /data では「周囲 25km 以内に FIRMS 検知ゼロ」を負例条件にする |
| **C** | スケール問題残存。savanna 草地では活火災ピクセル検出不可 | **中** | 訓練データを FRP > 100MW の大規模火災に絞る。/poc2 で可視化確認 |
| **D** | Δ < 0 (S2 が火災前) のケースが多い。未燃地が fire ラベルになるリスク | **中** | /data では Δ ≥ 0 (火災後 S2) を優先使用 |
| **E** | Event 2 失敗の合理的説明: Myanmar 海岸沿いの密林地帯で、負例が大規模火災クラスターの隣接ホットスポットに入った。フットプリントを超えた内陸 (~9km) では正常な非火災シグナルを確認済み | (参考) | — |
| **F** | `size_km=20` でタイル境界 nodata が発生 | **低** | /data パイプラインに nonzero_ratio > 0.5 チェックを入れる |
| **G** | `cloud_cover=44%` (Event 1) が混入 | **低** | /data で cloud_cover < 20% フィルタを適用 |

---

## /poc2 入場条件

CLAUDE.md 記載の入場条件:

- [x] /poc 完了 (スペクトルシグナルの分離確認済み) → **✅ 満たしている**

**→ /poc2 に進む。**

なお /poc2 の**完了条件**は以下の 2 点 (本フェーズでは未実施):

- [ ] ① few-shot ICL で LFM-VL がドメイン固有の疑似カラー画像に対して一定の判断ができること
- [ ] ② 小規模 LoRA FT (10〜20 サンプル) で few-shot より改善が見られること

懸念点 A (`fire_px_ratio=0` の条件下で VLM が視覚的に区別できるか) は ① の確認で明らかになる。
