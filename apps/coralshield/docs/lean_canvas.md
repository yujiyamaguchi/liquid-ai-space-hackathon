# CoralShield — Lean Canvas

---

## UVP

> **「NOAAの5km解像度では見えない礁スケールの白化を、Sentinel-2可視光バンドの色変化パターンをLFMが解析することで10m解像度・自動でリアルタイム検出する」**

---

## 1. Problem

サンゴ礁の大規模白化・死滅の早期検知ができない。

| 指標 | 数値 |
|---|---|
| サンゴ礁が支える経済価値 | 年間 $375B（観光・漁業・沿岸保護） |
| 過去40年の消滅率 | 50%（Global Coral Reef Monitoring Network） |
| 現在の目視調査 | 1サイト確認に 3〜7日、広域は事実上不可能 |
| NOAA CoralWatch の限界 | 空間解像度 1km（礁スケール 10m の被害を見逃す） |
| 大規模白化頻度 | 以前は10年に1回 → 現在は 2〜3年に1回 |

---

## 2. Customer / Early Adopter

| 層 | 組織 | 購買動機 |
|---|---|---|
| **Early Adopter** | **グレートバリアリーフ海洋公園局（GBRMPA）** | 年間 $1B AUD 予算、連続白化で政治的圧力最大 |
| Customer Segment 2 | 海洋保護区管理機関（UNESCO 世界遺産管理局） | 広域リアルタイム監視 |
| Customer Segment 3 | 海洋生態系保険会社 | パラメトリック保険トリガーの自動化 |

---

## 3. 市場規模

| | 規模 |
|---|---|
| TAM | $4.2B/年（海洋環境モニタリング市場全体） |
| SAM | $620M/年（サンゴ礁・沿岸生態系特化） |
| SOM | $28M/年（衛星ベース白化アラート SaaS、5年以内） |

---

## 4. Unfair Advantage

| ソリューション | 空間解像度 | 更新頻度 | 白化判定 | コスト/年 |
|---|---|---|---|---|
| **CoralShield** | **10m** | **2〜5日** | **自動・面的・定量** | **$15K〜$80K** |
| NOAA CoralWatch (SST) | 1km | 毎日 | 間接的（温度リスク） | 無償（精度低） |
| PlanetScope | 3m | 毎日 | 手動目視のみ | $50K〜$500K/年 |
| ダイバー目視調査 | cm | 月〜四半期 | 正確だが遅い | $200K〜$2M/年 |
| Allen Coral Atlas | 10m | 年1回 | 静的ベースマップのみ | 無償（リアルタイム不可） |

---

## 5. Key Metrics

| 指標 | 目標値 |
|---|---|
| Recall | ≥ 0.85 |
| Precision | ≥ 0.75 |
| F1 | ≥ 0.80 |
| False Positive Rate | ≤ 0.15 |
| 帯域幅削減率 | ≥ 80% |

---

## 6. Revenue Streams

- SaaS 年間ライセンス: $15K〜$80K/年
- パラメトリック保険トリガー API: $30K〜$150K/年
- 政府調達: $200K〜$1M/プロジェクト
- Blue Carbon クレジット検証補助: 従量課金

---

## 7. Channels

- API 配信: GeoJSON 形式の白化マスク + 重症度スコア
- Email/SMS アラート: 閾値超えで管理者に自動通知
- NGO パートナー経由: Reef Check・GBRMPA との共同研究

---

## 8. Assumptions

1. Sentinel-2 可視光（B2/B3/B4）+ B8A で白化サンゴの色変化（青緑化）が検知できる
2. 水深 15m 以浅の礁に限定すれば底部反射が取得可能
3. 熱帯海域で十分な数のクリーンシーン（雲被覆 < 20%）が取得できる
4. 2024年世界規模白化イベントの GBRMPA 公開データが FT 用 GT として使える
5. パラメトリック保険会社が AI 判定をトリガーとして法的に認める（メキシコ MARI 保険が先例）

---

## 技術スタック

```
SimSat API (Sentinel-2: B2/B3/B4/B8A)
  → 水面反射補正（Sunglint 除去）
  → 白化指標: B3/B4 比（白化サンゴは青緑化 → B3↑, B4↓）
  → Water Column Correction → 底部反射率推定
  → 疑似カラー合成 [B4, B3, B8A] → 224×224px チップ
  → LFM 2.5-VL-450M (LoRA FT)
  → {"status": "Bleached", "coverage_pct": 42, "confidence": 0.81}
  → GeoJSON（白化マスク）→ アラート配信
```

GT: 2024年白化ピーク期間（GBRMPA 公開座標）vs 2023年同月（白化なし）の対比
