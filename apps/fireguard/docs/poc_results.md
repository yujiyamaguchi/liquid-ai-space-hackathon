# FireGuard /poc 結果レポート (v2)
**方法論改訂**: 2026-04-26  
**スクリプト**: `apps/fireguard/poc.py`  
**ステータス**: ⏳ **再実行待ち** — `uv run python poc.py` で更新

---

## v1 → v2 の変更点

| 項目 | v1 (旧) | v2 (新) |
|---|---|---|
| **主指標** | NBR = (B8A-B12)/(B8A+B12) | **NDMI = (B8A-B11)/(B8A+B11)** |
| **科学的根拠** | Key & Benson 2006 (post-fire 評価用) | Gao 1996 + Myoung et al. 2018 (LFMC推定 R²=0.76) |
| **NEG サンプル** | サクラメントバレー農地 (植生タイプ交絡) | **チャパラル優勢エリア** (南カリフォルニア/Bay Area hills) |
| **バンド** | NDWI に B08 (842nm) を使用 | NDMI に **B8A (865nm)** を使用 (Gao 1996 正規近似) |
| **副指標** | なし | NBR (燃料構造+水分の複合) |

---

## 指標の科学的根拠

### 主指標: NDMI = (B8A - B11) / (B8A + B11)

- **Gao 1996** (Remote Sensing of Environment 58:257-266): NIR と SWIR を用いた葉内液体水分量 (EWT) 推定指標を定義。現代では NDMI と呼ぶ
- **B11 (1610nm)**: 液体水の吸収帯に位置し、植生含水率 (LFMC) 推定に最感度
- **Myoung et al. 2018** (MDPI Remote Sensing): 南カリフォルニアのチャパラル (Chamise) において NDMI ベースの LFMC 推定で R²=0.76, MAE=9.68% を達成
- **LFMC 臨界値**: < 77-80% で南カリフォルニア大規模火災リスクが急増 (Santa Monica Mountains 研究)
- **14-28日リードタイムの生理的根拠**: 乾燥ストレス下でクロロフィル 7-75%・アントシアニン 38-100% 減少 (Mississippi State 2025)

### 副指標: NBR = (B8A - B12) / (B8A + B12)

- Key & Benson 2006 (FIREMON): 本来は post-fire 焼失深刻度 (dNBR) 用途
- 発火前ベースラインとして燃料構造 + 水分の複合把握に有用
- B12 (2190nm) は土壌・枯死燃料への反応が大きいため、純粋な水分指標としては NDMI に劣る

---

## サンプル設計 (v2)

### POS イベント (10件、植生タイプ付き)

| イベント | 植生タイプ | 発火日 |
|---|---|---|
| CZU Lightning Complex | **チャパラル** | 2020-08-16 |
| SCU Lightning Complex | **チャパラル** | 2020-08-18 |
| LNU Lightning Complex | **チャパラル** | 2020-08-17 |
| Thomas Fire | **チャパラル** | 2017-12-04 |
| Kincade Fire | **チャパラル** | 2019-10-23 |
| Creek Fire | 針葉樹混合 | 2020-09-04 |
| Carr Fire | 針葉樹混合 | 2018-07-23 |
| Mendocino Complex | 針葉樹混合 | 2018-07-27 |
| Dixie Fire | 針葉樹混合 | 2021-07-13 |
| Caldor Fire | 針葉樹混合 | 2021-08-14 |

### NEG サンプル (10件、すべてチャパラル優勢エリア)

| 地点 | エリア | 日付 |
|---|---|---|
| Santa Monica Mts Aug2019 | 南カリフォルニア | 2019-08-01 |
| Santa Monica Mts Sep2019 | 南カリフォルニア | 2019-09-01 |
| Angeles NF foothills Jul2019 | 南カリフォルニア | 2019-07-15 |
| San Bernardino foothills Aug2022 | 南カリフォルニア | 2022-08-01 |
| Riverside chaparral Sep2022 | 南カリフォルニア | 2022-09-01 |
| Ventura hills Oct2022 | 南カリフォルニア | 2022-10-15 |
| Diablo Range Aug2019 | Bay Area hills | 2019-08-01 |
| Diablo Range Sep2019 | Bay Area hills | 2019-09-01 |
| Mt Diablo Jul2022 | Bay Area hills | 2022-07-15 |
| Sonoma hills Oct2022 | Bay Area hills | 2022-10-01 |

全 NEG は FIRMS SP fire-free 確認済み (実行時に自動チェック)

---

## 完了条件 (v2)

| 条件 | 評価指標 | 閾値 |
|---|---|---|
| ① SimSat データ取得 | POS/NEG シーン数 | POS > 0, NEG > 0 |
| ② NDMI スペクトル分離 (主) | Mann-Whitney U (alternative="less") | **チャパラル限定** で p < 0.10 (いずれかの lead) |
| ③ NBR 副指標確認 | 同上 | 参考値として記録 |
| ④ seasonal anomaly (任意) | NDMI(現在) − NDMI(1〜2年前同日) | 負値 (発火前の方が例年より低い) |

---

## 定量結果

**(再実行後にここを更新)**

---

## 根本原因分析・Plan B

**(再実行後に必要に応じて更新)**

v1 で確認済みの事項:
- チャパラル/低木イベント (CZU/SCU/LNU/Thomas) では NBR Δ = -0.075 と正しい方向
- 針葉樹混合では逆方向 (常緑で乾燥しても NIR が維持される)
- NEG が農地だったことで絶対値比較が無効だった

v2 で確認すること:
- NEG をチャパラルに揃えた上で NDMI の絶対値比較が有意かどうか
- NDMI が NBR より強いシグナルを示すか
- seasonal anomaly が取得できる場合、絶対値比較より有意差が改善するか

---

## ファイル (再実行後に生成)

- `data/poc/results.csv` — 全シーンの指標値
- `data/poc/figures/ndmi_distributions.png` — NDMI 分布図 (主指標)
- `data/poc/figures/nbr_distributions.png` — NBR 分布図 (副指標)
- `data/poc/figures/ndmi_by_vegtype.png` — 植生タイプ別 scatter
- `data/poc/images/` — 疑似カラー画像
