# **Sentinel-2衛星を用いたチャパラルにおける発火前植生乾燥ストレス検出の科学的根拠：生体燃料水分量（LFMC）推定とスペクトル指標の統合的分析**

## **序論：地中海性低木林における火災リスク管理の現代的課題**

南カリフォルニアのチャパラルや地中海沿岸に広がる低木林（Mediterranean Shrublands）は、世界で最も激しい火災が発生しやすい生態系の一つである。これらの地域は、湿潤な冬と極端に乾燥した夏を特徴とする地中海性気候に属しており、夏季の長期的な水分欠乏は植生の燃料としての可燃性を極限まで高める原因となる 1。特に、生体燃料水分量（Live Fuel Moisture Content: LFMC）は、火災の着火性、延焼速度、および燃焼強度を決定する決定的な要因であり、火災行動モデルにおける最も重要な入力変数の一つである 3。  
LFMCは、植物の生重量（$W\_f$）と絶乾重量（$W\_d$）の比率として定義され、以下の式で表される：

$$LFMC (\\%) \= \\frac{W\_f \- W\_d}{W\_d} \\times 100$$  
チャパラル種において、この値が臨界値（通常70-80%程度）を下回ると、火災の拡大リスクが急激に増大することが知られている 2。しかし、広大な山岳地帯において現場でのサンプリングによるLFMC測定を行うことは、多大なコストと時間を要し、リアルタイムのモニタリングには適していない 6。  
欧州宇宙機関（ESA）のSentinel-2ミッションの登場は、この課題に対して革新的な解決策をもたらした。10〜20mという高い空間解像度、5日という短い再訪周期、そして植生の水分状態を直接的に捉える短波長赤外（SWIR）バンドの搭載により、発火前の段階で植生の乾燥ストレスを詳細に追跡することが可能となった 9。本報告書では、Sentinel-2のデータを活用した発火前（pre-fire）の乾燥ストレス検出に関する科学的根拠を、14〜28日前という時間窓におけるスペクトル変化、LFMCの推定精度、および最適な指標の選択という観点から、既存の学術的証拠に基づいて詳述する。

## **第1章：発火14〜28日前におけるスペクトル指標の変遷と定量的実証**

火災発生の数週間前、植生は生理学的な乾燥プロセスを経て、そのスペクトル特性を変化させる。この段階での変化を捉えることは、早期警戒システムの構築において極めて重要である。

### **1.1 生理的変化とスペクトル相関のメカニズム**

乾燥ストレス下にあるチャパラル植生は、蒸散を抑制するために気孔を閉鎖し、葉の内部水分量を減少させる。研究によれば、極端な熱ストレスや乾燥が継続する場合、発火の14〜28日前の期間において、植物の色素組成に顕著な変化が生じることが実証されている 11。

| 測定項目 | 14〜28日間の変化率（減少） | 生理的含意 | 根拠 |
| :---- | :---- | :---- | :---- |
| **クロロフィル** | 7% 〜 75% | 光合成能力の低下と老化の加速 | 11 |
| **アントシアニン** | 38% 〜 100% | 抗酸化能力の喪失 | 11 |
| **フラボノイド** | 34% 〜 88% | 紫外線および環境ストレス保護機能の低下 | 11 |
| **窒素バランス指数 (NBI)** | 7% 〜 85% | 植物の栄養状態と全体的な活力の低下 | 11 |

これらの生理的変化は、衛星から観測される近赤外（NIR）および短波長赤外（SWIR）の反射率に直接的な影響を与える。水分を失った葉はSWIR域での反射率が増大し、同時に葉の内部構造の崩壊やクロロフィルの減少によってNIR域の反射率が低下する 12。

### **1.2 チャパラルにおける定量的実証結果**

南カリフォルニアのチャパラル（Chamise: アデノストマ等）を対象とした研究では、発火前の乾燥プロセスを追跡するために膨大なデータセットが活用されている。ある研究では、10,000件以上のLFMC地上観測データとLandsat/Sentinel-2の時系列データを結合し、ランダムフォレスト（Random Forest）モデルを用いて推定を行っている 3。

* **Chamise（アデノストマ）**: 最も頑健なモデル出力が得られており、決定係数 $R^2 \= 0.76$、平均絶対誤差 MAE \= 9.68% を記録した 3。  
* **サンプルサイズと統計的有意性**: スペインの地中海性低木林（Cistus ladanifer）の研究では、n \= 335 の現場スペクトルマッチングデータに基づき、発火前のLFMC低下を予測している。このモデルでは、LFMCが100%以下の「高リスク状態」において、推定誤差（MAE）が10%まで低下することが示されており、火災リスクが高いほどモデルの精度が向上する傾向にある 7。

### **1.3 14〜28日前というリードタイムの意義**

歴史的に、14〜28日という期間はLandsat衛星の再訪周期（16日）に伴う観測の制約として語られることが多かった 9。しかし、Sentinel-2の5日周期観測（Sentinel-2A/B併用）により、この期間内に複数の高品質な画像を取得することが可能となった 10。  
スペインのバレンシア地域における研究では、発火の約1週間（8〜9日前）に予測されたLFMCが、実際の火災発生直前の極端な乾燥状態を正確に捉えていた 8。この事実は、14〜28日前から始まる緩やかな乾燥シグナルが、発火直前の1週間で臨界値に達するまでの連続的な軌道を衛星データで描けることを示唆している 8。

## **第2章：Sentinel-2バンドによる生体燃料水分量（LFMC）の推定精度**

Sentinel-2は、従来の光学衛星と比較して、植生の水分と構造に特化したバンド構成（特にRed-edgeバンドと2つのSWIRバンド）を有している。

### **2.1 主要バンドの役割と感度**

LFMC推定において、B8A（ナローNIR: 865nm）、B11（SWIR1: 1610nm）、B12（SWIR2: 2190nm）の3バンドは中核をなす。

* **B8A (865nm)**: 従来のB8（広帯域NIR）に比べ水蒸気の影響を受けにくく、葉の内部構造（スポンジ状組織）の変化を反映する 7。  
* **B11 (1610nm)**: 葉の水分含有量（EWT）に最も敏感な波長域であり、LFMC推定モデルの多くで主要な変数として選択される 12。  
* **B12 (2190nm)**: 水分に加えて、枯死燃料や土壌の露出、さらには火災後の「炭」のシグナルにも反応するため、乾燥が極限まで進んだ段階での識別に有効である 13。

### **2.2 推定精度の定量的比較**

複数の先行研究から得られたLFMC推定の精度指標を以下に統合する。

| 対象地域・植生 | モデル手法 | 使用データ | R2 (adj) | RMSE / MAE | 根拠 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **南カリフォルニア (Chamise)** | Random Forest | Landsat/S2時系列 | 0.76 | MAE: 9.68% | 3 |
| **スペイン (Cistus ladanifer)** | 重回帰 (MLR) | S2 (VARI, EVI等) | 0.76 | MAE: 12% | 6 |
| **スペイン (低木混合)** | GAM (一般化加法モデル) | S2 \+ 気象データ | 0.65 〜 0.74 | RMSE: 7.4 〜 27.7 | 8 |
| **地中海性低木 (広域)** | 線形回帰 | S2 (20m解像度) | 0.55 〜 0.63 | MAE: 13.4% 〜 15.1% | 4 |
| **高リスク状態 (LFMC \< 120%)** | 閾値限定解析 | S2 | \- | MAE: 9.1% | 5 |

### **2.3 モデルの特性：経験モデル vs 物理モデル**

研究によれば、特定のサイトに特化した経験的モデル（回帰、機械学習）は、物理的な放射伝達モデル（RTM）の逆転解析よりも局所的な精度において優れる場合が多い 19。一方で、RTMは場所を問わない汎用性を持つが、パラメータ設定が複雑であるという欠点がある 8。Sentinel-2のB8A, B11, B12を組み合わせたスペクトル指標（NDMI等）は、これらのモデルにおいて最も寄与率の高い変数となっている 8。

## **第3章：発火前シグナルの適切なリードタイムと環境因子による変動**

発火前の乾燥シグナルが「何日前まで有効か」という問いに対しては、植生タイプと気候条件が決定的な回答を与える。

### **3.1 植生タイプ別の反応速度**

チャパラル生態系内でも、種によって乾燥への適応戦略が異なるため、衛星データのシグナル強度も変化する。

* **低木（Shrubs）**: チャパラルの主役である常緑低木（Chamise, Rosmarinus等）は、顕著な季節変動を示す。発火の数ヶ月前から徐々に低下し始め、28日前から14日前にかけて乾燥が加速する 8。  
* **樹木（Trees）**: 地中海性の高木はLFMCの挙動が比較的平坦であり、季節的な変化が少ない。そのため、樹木主体のエリアでは短期的な発火前シグナルの検出は低木地帯よりも困難である 8。  
* **草本（Grasses）**: 乾季（6月〜9月）に急速に乾燥し、燃料水分指標は急落する。乾燥期のCSC（Canopy Storage Capacity）の推定精度は $R^2 \= 0.93$ と非常に高く、極めて短いスパンでのモニタリングが可能である 15。

### **3.2 気候帯と干ばつの影響**

南カリフォルニアのサンタモニカ山脈における解析では、LFMCが77%を下回ることが大規模火災発生の閾値（Critical threshold）として特定されている 2。また、2012〜2016年のカリフォルニアにおける前例のないメガ・ドライト（巨大干ばつ）の影響下では、発火前の植生回復力が著しく低下しており、これが発火前のスペクトル指標にベースラインの低下として現れることが確認されている 1。

## **第4章：Key & Benson (2006) のNBRは発火前検出に使えるか？**

本来、正規化燃焼比（Normalized Burn Ratio: NBR）は、火災前後の画像を用いた差分（dNBR）によって燃焼過酷度（Burn Severity）を測定するために開発された 25。しかし、近年の研究ではこれを「発火前」の状態把握に利用する例が増えている。

### **4.1 発火前ベースラインとしてのNBR**

dNBRの算出公式は以下の通りである：

$$dNBR \= NBR\_{prefire} \- NBR\_{postfire}$$  
ここで、$NBR\_{prefire}$ は単なる計算上の参照値ではなく、その時点での「燃料の状態」を示す指標として機能する 17。

* **燃料ポテンシャルの把握**: $NBR\_{prefire}$ が高い値を示す場合、それは植生密度が高く、水分が豊富であることを示唆する。逆に、発火前の段階でNBRが低下している地点は、燃料が既にストレスを受けているか、あるいは疎な状態であることを示す 25。  
* **乾燥ストレスの複合指標**: 一部の研究では、NBRを単独の水分指標としてではなく、MSI（Moisture Stress Index）やNDDI（Normalized Difference Drought Index）と組み合わせて「複合的乾燥過酷度」を評価する変数として利用している。この手法を用いたランダムフォレストモデルは、86.53%という高い精度で乾燥ストレスの分類に成功している 32。

### **4.2 制約事項**

NBRはSWIR2（B12）を使用するため、水分含有量だけでなく「炭」や「土壌の露出」に対して極めて敏感である 13。したがって、純粋な生理的乾燥ストレスの検出という点では、後述するNDMIに一歩譲る。しかし、火災管理において「燃えやすい構造（燃料負荷）」と「水分状態」を同時に把握したい場合には、NBRは依然として有用な発火前シグナルとなり得る 25。

## **第5章：推奨指標の比較：NDMI vs NDWI vs NBR**

発火前の乾燥ストレス検出において、どのスペクトル指標が最も信頼できるかについては、学術的なコンセンサスが得られつつある。

### **5.1 指標別の特性比較**

| 指標 | 波長構成 (S2) | 主な適用範囲 | 推奨度 (発火前) | 理由・根拠 |
| :---- | :---- | :---- | :---- | :---- |
| **NDMI** | (B8 \- B11) / (B8 \+ B11) | 植生内部の水分ストレス、LFMC推定 | **最高** | SWIR1 (B11) は水の吸収に特化しており、葉の構造ノイズをNIRで相殺できる。 12 |
| **NDWI** | (B3 \- B8) / (B3 \+ B8) | 水体、湖沼、土壌表面水分 | **低い** | 元来、水面検出用に開発されたもので、植生内部の微量な水分変化への感度は低い。 33 |
| **NBR** | (B8 \- B12) / (B8 \+ B12) | 燃焼過酷度、火災痕跡、燃料構造 | **中程度** | 水分にも反応するが、SWIR2 (B12) は土壌や炭の影響が大きく、水分単独の指標としてはノイズが多い。 13 |

### **5.2 NDMIの優位性**

Gao (1996) によって提案されたNDMI（論文によっては当初NDWIと呼ばれていた）は、NIRとSWIRを組み合わせることで、葉の内部水分含有量（EWT）を最も正確に抽出する 12。

1. **分光学的理由**: 1610nm付近（B11）は液体水の吸収の谷に位置し、NIRは葉の多重散乱を捉える。この比率をとることで、葉の量（LAI）による変動を抑え、水分量そのものの変化を強調できる 12。  
2. **実証データ**: 地中海地域の森林火災において、他の指標（NDVI等）よりも火災リスクの空間分布を正確に反映することが示されている 36。  
3. **火災管理への統合**: EOSDA等の最新プラットフォームでも、NDMIは「点火リスクの検出」におけるデファクトスタンダードとして推奨されている 34。

## **結論：Sentinel-2を活用した早期警戒システムの構築に向けて**

本報告書での分析を通じて、Sentinel-2衛星のB8A, B11, B12バンドを活用したチャパラルの発火前乾燥ストレス検出は、高い科学的妥当性と実用性を持つことが明らかになった。

* **14〜28日という時間窓**: この期間は、植物が色素や水分を失い、生理学的な限界点に達する重要な移行期である。Sentinel-2の5日周期観測は、この変化を捉えるのに十分な時間解像度を提供する 10。  
* **LFMC推定の精度**: チャパラル植生において、MAE 10% 未満という高い精度でのLFMC推定が可能であり、これは現地の火災管理者が「危険域（LFMC \< 80%）」を判断する上で極めて信頼性の高い基準となる 3。  
* **NBRの再定義**: Key & BensonのNBRは、単なる事後評価ツールではなく、発火前の「燃料の状態」を定量化するベースラインとして極めて有用である 25。  
* **NDMIの推奨**: 植生の乾燥ストレスを純粋に追跡する目的においては、NDMI（B8/B11）が最も推奨される。これは、その分光学的設計が植物生理に忠実であるためである 12。

今後の展望として、Sentinel-2の光学データにSentinel-1のレーダー（SAR）データを統合するアプローチが注目されている。一部の研究では、SARデータ単独では光学データほどの精度は得られない（$R^2=0.28$）ものの、両者を組み合わせることで、厚い雲に覆われた場合でも乾燥トレンドを継続的に追跡できる可能性がある 4。気候変動に伴い火災シーズンが長期化する中で、これらの衛星指標を活用した精緻な発火前モニタリングは、被害を最小限に抑えるための不可欠な技術基盤となるであろう。

#### **引用文献**

1. Evaluating Drought Impact on Postfire Recovery of Chaparral Across Southern California \- PMC, 4月 26, 2026にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC7720657/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7720657/)  
2. VOLUME 2B: GOALS AND OBJECTIVES FOR THREATS/STRESSORS 1.0 ALTERED FIRE REGIME \- SDMMP.com, 4月 26, 2026にアクセス、 [https://sdmmp.com/upload/SDMMP\_Repository/0/f9j328v4sc0yrpn5kzxbh6wm71qgdt.pdf](https://sdmmp.com/upload/SDMMP_Repository/0/f9j328v4sc0yrpn5kzxbh6wm71qgdt.pdf)  
3. A 32-year species-specific live fuel moisture content dataset for ..., 4月 26, 2026にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC13009464/](https://pmc.ncbi.nlm.nih.gov/articles/PMC13009464/)  
4. Characterizing Live Fuel Moisture Content from Active and Passive Sensors in a Mediterranean Environment \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/1999-4907/13/11/1846](https://www.mdpi.com/1999-4907/13/11/1846)  
5. Characterizing Live Fuel Moisture Content from Active and Passive Sensors in a Mediterranean Environment \- ResearchGate, 4月 26, 2026にアクセス、 [https://www.researchgate.net/publication/365238687\_Characterizing\_Live\_Fuel\_Moisture\_Content\_from\_Active\_and\_Passive\_Sensors\_in\_a\_Mediterranean\_Environment](https://www.researchgate.net/publication/365238687_Characterizing_Live_Fuel_Moisture_Content_from_Active_and_Passive_Sensors_in_a_Mediterranean_Environment)  
6. Estimation of live fuel moisture content of shrubland using MODIS and Sentinel-2 images \- SciSpace, 4月 26, 2026にアクセス、 [https://scispace.com/pdf/estimation-of-live-fuel-moisture-content-of-shrubland-using-4zj850grtv.pdf](https://scispace.com/pdf/estimation-of-live-fuel-moisture-content-of-shrubland-using-4zj850grtv.pdf)  
7. Transferability of Empirical Models Derived from Satellite Imagery for Live Fuel Moisture Content Estimation and Fire Risk Prediction \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/2571-6255/7/8/276](https://www.mdpi.com/2571-6255/7/8/276)  
8. Analyzing Independent LFMC Empirical Models in the Mid-Mediterranean Region of Spain Attending to Vegetation Types and Bioclimatic Zones \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/1999-4907/14/7/1299](https://www.mdpi.com/1999-4907/14/7/1299)  
9. Current and Near-Term Earth-Observing Environmental Satellites, Their Missions, Characteristics, Instruments, and Applications \- PMC, 4月 26, 2026にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC11175343/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175343/)  
10. Current and Near-Term Earth-Observing Environmental Satellites, Their Missions, Characteristics, Instruments, and Applications \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/1424-8220/24/11/3488](https://www.mdpi.com/1424-8220/24/11/3488)  
11. October 18, 2025 \- The Graduate School \- Mississippi State University, 4月 26, 2026にアクセス、 [https://www.grad.msstate.edu/sites/www.grad.msstate.edu/files/2025-10/Fall%202025%20Grad%20Symposium%20Program.pdf](https://www.grad.msstate.edu/sites/www.grad.msstate.edu/files/2025-10/Fall%202025%20Grad%20Symposium%20Program.pdf)  
12. Normalized Difference Moisture Index (NDMI) \- Sentinel Hub custom scripts, 4月 26, 2026にアクセス、 [https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndmi/](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndmi/)  
13. Time series of spectral signatures of burned areas in different types... \- ResearchGate, 4月 26, 2026にアクセス、 [https://www.researchgate.net/figure/Time-series-of-spectral-signatures-of-burned-areas-in-different-types-of-land-cover-in\_fig6\_236694480](https://www.researchgate.net/figure/Time-series-of-spectral-signatures-of-burned-areas-in-different-types-of-land-cover-in_fig6_236694480)  
14. The temporal dimension of differenced Normalized Burn Ratio (dNBR) fire/burn severity studies \- EarthMapps.io, 4月 26, 2026にアクセス、 [https://earthmapps.io/pubs/2010\_Veraverbeke\_RSE\_The%20temporal%20dimension%20of%20differenced.pdf](https://earthmapps.io/pubs/2010_Veraverbeke_RSE_The%20temporal%20dimension%20of%20differenced.pdf)  
15. Inter-Seasonal Estimation of Grass Water Content Indicators Using Multisource Remotely Sensed Data Metrics and the Cloud-Computing Google Earth Engine Platform \- ResearchGate, 4月 26, 2026にアクセス、 [https://www.researchgate.net/publication/368914237\_Inter-Seasonal\_Estimation\_of\_Grass\_Water\_Content\_Indicators\_Using\_Multisource\_Remotely\_Sensed\_Data\_Metrics\_and\_the\_Cloud-Computing\_Google\_Earth\_Engine\_Platform](https://www.researchgate.net/publication/368914237_Inter-Seasonal_Estimation_of_Grass_Water_Content_Indicators_Using_Multisource_Remotely_Sensed_Data_Metrics_and_the_Cloud-Computing_Google_Earth_Engine_Platform)  
16. Introduction to Vegetation Indices: What are Vegetation Indices and Why are They Important? \- Geohub, 4月 26, 2026にアクセス、 [https://geohubkenya.wordpress.com/2025/02/09/introduction-to-vegetation-indices-what-are-vegetation-indices-and-why-are-they-important/](https://geohubkenya.wordpress.com/2025/02/09/introduction-to-vegetation-indices-what-are-vegetation-indices-and-why-are-they-important/)  
17. Evaluation of thresholds for burn severity classes \- ArcGIS StoryMaps, 4月 26, 2026にアクセス、 [https://storymaps.arcgis.com/stories/0083e8915b9d4d0ca2f77e562257651a](https://storymaps.arcgis.com/stories/0083e8915b9d4d0ca2f77e562257651a)  
18. Assessment of Burned Forest Area Severity and Postfire Regrowth in Chapada Diamantina National Park (Bahia, Brazil) Using dNBR and RdNBR Spectral Indices \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/2076-3263/10/3/106](https://www.mdpi.com/2076-3263/10/3/106)  
19. Analyzing Independent LFMC Empirical Models in the Mid-Mediterranean Region of Spain Attending to Vegetation Types and Bioclimatic Zones \- ResearchGate, 4月 26, 2026にアクセス、 [https://www.researchgate.net/publication/371899584\_Analyzing\_Independent\_LFMC\_Empirical\_Models\_in\_the\_Mid-Mediterranean\_Region\_of\_Spain\_Attending\_to\_Vegetation\_Types\_and\_Bioclimatic\_Zones](https://www.researchgate.net/publication/371899584_Analyzing_Independent_LFMC_Empirical_Models_in_the_Mid-Mediterranean_Region_of_Spain_Attending_to_Vegetation_Types_and_Bioclimatic_Zones)  
20. Characterizing Live Fuel Moisture Content from Active and Passive Sensors in a Mediterranean Environment \- Semantic Scholar, 4月 26, 2026にアクセス、 [https://pdfs.semanticscholar.org/ef76/4a41009ada26eeb5a3d24a176dd9f2c1d77a.pdf](https://pdfs.semanticscholar.org/ef76/4a41009ada26eeb5a3d24a176dd9f2c1d77a.pdf)  
21. Empirical Models for Spatio-Temporal Live Fuel Moisture Content Estimation in Mixed Mediterranean Vegetation Areas Using Sentinel-2 Indices and Meteorological Data \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/2072-4292/13/18/3726](https://www.mdpi.com/2072-4292/13/18/3726)  
22. International Journal of Wildland Fire 20-Year Author Index Vol. 1(1) 1991 – vol. 20(8) 2011 \- CSIRO Publishing, 4月 26, 2026にアクセス、 [https://www.publishing.csiro.au/media/client/WFauths20yrs.pdf](https://www.publishing.csiro.au/media/client/WFauths20yrs.pdf)  
23. Monitoring Post-Fire Recovery of Chaparral and Conifer Species Using Field Surveys and Landsat Time Series \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/2072-4292/11/24/2963](https://www.mdpi.com/2072-4292/11/24/2963)  
24. (PDF) Monitoring Post-Fire Recovery of Chaparral and Conifer Species Using Field Surveys and Landsat Time Series \- ResearchGate, 4月 26, 2026にアクセス、 [https://www.researchgate.net/publication/337886118\_Monitoring\_Post-Fire\_Recovery\_of\_Chaparral\_and\_Conifer\_Species\_Using\_Field\_Surveys\_and\_Landsat\_Time\_Series](https://www.researchgate.net/publication/337886118_Monitoring_Post-Fire_Recovery_of_Chaparral_and_Conifer_Species_Using_Field_Surveys_and_Landsat_Time_Series)  
25. Landscape Assessment (LA) \- Sampling and Analysis Methods \- FIREMON: Fire effects monitoring and inventory system, 4月 26, 2026にアクセス、 [https://www.fs.usda.gov/rm/pubs\_series/rmrs/gtr/rmrs\_gtr164/rmrs\_gtr164\_13\_land\_assess.pdf](https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr164/rmrs_gtr164_13_land_assess.pdf)  
26. Landscape Assessment: Ground measure of severity, the Composite Burn Index; and Remote sensing of severity, the Normalized Burn Ratio \- USGS Publications Warehouse, 4月 26, 2026にアクセス、 [https://pubs.usgs.gov/publication/2002085](https://pubs.usgs.gov/publication/2002085)  
27. (PDF) Landscape Assessment: Ground measure of severity, the Composite Burn Index; and Remote sensing of severity, the Normalized Burn Ratio. \- ResearchGate, 4月 26, 2026にアクセス、 [https://www.researchgate.net/publication/241687027\_Landscape\_Assessment\_Ground\_measure\_of\_severity\_the\_Composite\_Burn\_Index\_and\_Remote\_sensing\_of\_severity\_the\_Normalized\_Burn\_Ratio](https://www.researchgate.net/publication/241687027_Landscape_Assessment_Ground_measure_of_severity_the_Composite_Burn_Index_and_Remote_sensing_of_severity_the_Normalized_Burn_Ratio)  
28. Multidecadal satellite-derived Portuguese Burn Severity Atlas (1984–2022) \- ESSD Copernicus, 4月 26, 2026にアクセス、 [https://essd.copernicus.org/preprints/essd-2024-305/essd-2024-305-manuscript-version4.pdf](https://essd.copernicus.org/preprints/essd-2024-305/essd-2024-305-manuscript-version4.pdf)  
29. A New Metric for Quantifying Burn Severity: The Relativized Burn Ratio \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/2072-4292/6/3/1827](https://www.mdpi.com/2072-4292/6/3/1827)  
30. Full article: High spatial resolution burn severity mapping of the New Jersey Pine Barrens with WorldView-3 near-infrared and shortwave infrared imagery \- Taylor & Francis, 4月 26, 2026にアクセス、 [https://www.tandfonline.com/doi/full/10.1080/01431161.2016.1268739](https://www.tandfonline.com/doi/full/10.1080/01431161.2016.1268739)  
31. Predicting Potential Fire Severity Using Vegetation, Topography and Surface Moisture Availability in a Eurasian Boreal Forest Landscape \- MDPI, 4月 26, 2026にアクセス、 [https://www.mdpi.com/1999-4907/9/3/130](https://www.mdpi.com/1999-4907/9/3/130)  
32. Assessment of Spatio-Temporal Dynamics of Drought Stress Anomalies Using Hyperspectral Imagery Fusion \- ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 4月 26, 2026にアクセス、 [https://isprs-annals.copernicus.org/articles/X-5-W2-2025/115/2025/isprs-annals-X-5-W2-2025-115-2025.pdf](https://isprs-annals.copernicus.org/articles/X-5-W2-2025/115/2025/isprs-annals-X-5-W2-2025-115-2025.pdf)  
33. Harmonized Landsat Sentinel-2 (HLS) Vegetation Indices (VI) Product User Guide \- LP DAAC, 4月 26, 2026にアクセス、 [https://lpdaac.usgs.gov/documents/2088/HLS\_VI\_User\_Guide\_V2.pdf](https://lpdaac.usgs.gov/documents/2088/HLS_VI_User_Guide_V2.pdf)  
34. Wildfire Detection And Forest Fire Monitoring With Satellite Data \- EOS Data Analytics, 4月 26, 2026にアクセス、 [https://eos.com/wildfire-detection-and-monitoring/](https://eos.com/wildfire-detection-and-monitoring/)  
35. NDMI Index: Monitor Crop Moisture And Manage Water Stress \- EOS Data Analytics, 4月 26, 2026にアクセス、 [https://eos.com/make-an-analysis/ndmi/](https://eos.com/make-an-analysis/ndmi/)  
36. Short-term temporal and spatial analysis for post-fire vegetation regrowth characterization and mapping in a Mediterranean ecosy \- Taylor & Francis, 4月 26, 2026にアクセス、 [https://www.tandfonline.com/doi/pdf/10.1080/10106049.2022.2097482](https://www.tandfonline.com/doi/pdf/10.1080/10106049.2022.2097482)  
37. Full article: Short-term temporal and spatial analysis for post-fire vegetation regrowth characterization and mapping in a Mediterranean ecosystem using optical and SAR image time-series \- Taylor & Francis, 4月 26, 2026にアクセス、 [https://www.tandfonline.com/doi/full/10.1080/10106049.2022.2097482](https://www.tandfonline.com/doi/full/10.1080/10106049.2022.2097482)