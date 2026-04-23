"""
FireEdge: Smoke-Penetrating Fire Detection Pipeline
====================================================
Specification-Driven Development (SDD) — すべての入出力インターフェース定義

このファイルはコードの「憲法」です。実装前にここで合意を取り、
実装中もここを参照して型・スキーマのズレを防いでください。

Author: FireEdge Team
Date: 2026-04-15
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from PIL.Image import Image  # type: ignore


# ===========================================================================
# 1. 定数 & Enum
# ===========================================================================

class SpectralBand(str, Enum):
    """SimSat API が受け付けるバンド名 (spectral_bands パラメータ)"""
    RED     = "red"       # B04: 665nm  — 可視赤
    GREEN   = "green"     # B03: 560nm  — 可視緑
    BLUE    = "blue"      # B02: 490nm  — 可視青
    NIR     = "nir"       # B08: 842nm  — 近赤外
    SWIR16  = "swir16"    # B11: 1610nm — 短波赤外1 (煙透過・熱異常)
    SWIR22  = "swir22"    # B12: 2190nm — 短波赤外2 (活火検出の主役)
    REDEDGE1 = "rededge1" # B05: 705nm
    REDEDGE2 = "rededge2" # B06: 740nm
    REDEDGE3 = "rededge3" # B07: 783nm


class FireSeverity(str, Enum):
    NONE     = "NONE"
    LOW      = "LOW"      # 小規模・延焼リスク低
    MEDIUM   = "MEDIUM"   # 中規模・要監視
    HIGH     = "HIGH"     # 大規模・要警戒
    CRITICAL = "CRITICAL" # 緊急・即時通報


class SpreadDirection(str, Enum):
    N  = "N"
    NE = "NE"
    E  = "E"
    SE = "SE"
    S  = "S"
    SW = "SW"
    W  = "W"
    NW = "NW"


# ===========================================================================
# 2. SimSat API レスポンス型
# ===========================================================================

@dataclass
class SatellitePosition:
    """GET /data/current/position レスポンス"""
    lon: float         # 経度 [-180, 180]
    lat: float         # 緯度 [-90, 90]
    alt_km: float      # 高度 [km]
    timestamp: str     # ISO-8601 UTC


@dataclass
class SentinelImageResponse:
    """GET /data/current/image/sentinel または /data/image/sentinel レスポンス

    return_type="array" を指定した場合の追加フィールド:
    - image_array: shape (H, W, C) の numpy ndarray、dtype=float32、値域 [0.0, 1.0]
      H=W は size_km と解像度(10m/px)から決まる (例: size_km=10 → 1000px)
      C はリクエストした spectral_bands の数

    注意: bands の順序は リクエスト時の spectral_bands リストと一致する
    """
    image_available: bool
    source: str                    # "sentinel-2a" | "sentinel-2b" | "sentinel-2c"
    spectral_bands: list[str]      # 取得バンド名リスト
    footprint: tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    size_km: float
    cloud_cover: float             # [0.0, 100.0] %
    datetime: str                  # 撮影日時 ISO-8601 UTC
    satellite_position: SatellitePosition
    timestamp: str                 # API 応答日時 ISO-8601 UTC
    image_array: Optional[object] = None  # numpy ndarray (H, W, C) dtype=float32


# ===========================================================================
# 3. スペクトル処理 入出力
# ===========================================================================

# --- 3a. バンドリクエスト仕様 ---

FIRE_DETECTION_BANDS = [
    SpectralBand.SWIR22,   # ch0: R → 活火の主役
    SpectralBand.SWIR16,   # ch1: G → 熱異常・煙透過
    SpectralBand.NIR,      # ch2: B → 植生背景
]
"""疑似カラー合成に使うバンドセット (return_type="array" で取得)
活火 → 赤, 焼跡 → 暗赤, 植生 → 緑, 煙 → 白/灰, 水 → 黒 に見える"""

SMOKE_DETECTION_BANDS = [
    SpectralBand.RED,
    SpectralBand.GREEN,
    SpectralBand.BLUE,
]
"""煙の可視化用 RGB (smoke_mask 生成に使用)"""

RECOMMENDED_SIZE_KM = 20.0
"""1シーンのカバレッジ [km]: 10m解像度 → 2000×2000 px の配列"""


# --- 3b. 前処理パラメータ ---

@dataclass
class SpectralPreprocessConfig:
    """スペクトル前処理の設定値"""
    target_size_px: int = 448
    """LFM 2.5-VL への入力サイズ [px]。正方形にリサイズする。"""

    percentile_clip: tuple[float, float] = (2.0, 98.0)
    """外れ値クリップのパーセンタイル範囲 (下限, 上限)"""

    normalize_per_band: bool = True
    """バンドごとに独立して [0, 1] 正規化するか。
    False の場合は全チャネルを一括正規化 (コントラスト重視時)"""


# --- 3c. スペクトル指標 ---

@dataclass
class SpectralIndices:
    """バンド演算で導出するスペクトル指標

    各指標の値域と物理的意味:

    NBR2 = (swir16 - swir22) / (swir16 + swir22)
      値域: [-1.0, 1.0]
      > 0.3  : 健全な植生
      0 〜 0.3: 裸地・乾燥植生
      < 0    : 活火または焼跡 ← 火災検知の主要閾値

    NBR2_min: シーン内ピクセルの NBR2 最小値 (最も活火に近いピクセル)
      poc2 と同じ計算。小規模火災でも敏感に反応する。
      シーン平均 (nbr2) は植生に希釈されるが、nbr2_min は火炎ピクセルを直接拾う。

    NDVI = (nir - red) / (nir + red)
      値域: [-1.0, 1.0]
      > 0.4  : 濃い植生 (森林)
      0.1〜0.4: 疎な植生・農地
      < 0    : 水・雲・裸地

    BAI (Burned Area Index) = 1 / ((0.1 - red)^2 + (0.06 - nir)^2)
      値域: [0, ~1000]
      高値  : 焼跡 (charcoal) の特徴スペクトルに近い
    """
    nbr2: float        # シーン平均 NBR2 (植生優勢シーンでは正になる)
    nbr2_min: float    # シーン内 NBR2 最小値 (火炎ピクセルに感応、poc2 と同じ)
    ndvi: float
    bai: float
    mean_swir22: float    # シーン平均 SWIR22 輝度 [0.0, 1.0]
    swir22_max: float     # シーン最大 SWIR22 輝度 (poc2 と同じ、熱異常ピクセルに感応)
    fire_pixel_ratio: float  # 閾値以下の NBR2 画素の割合 [0.0, 1.0]


# --- 3d. 前処理出力 ---

@dataclass
class ProcessedScene:
    """スペクトル前処理の出力。LFM 2.5-VL への入力に直接使う。"""

    fire_composite: Image
    """疑似カラー合成画像 (SWIR22, SWIR16, NIR → RGB)
    mode="RGB", size=(448, 448), dtype相当は uint8"""

    rgb_image: Image
    """通常RGB画像 (smoke 可視化用)
    mode="RGB", size=(448, 448)"""

    indices: SpectralIndices
    """シーン全体のスペクトル指標サマリ"""

    footprint: tuple[float, float, float, float]
    """地理座標 (lon_min, lat_min, lon_max, lat_max)"""

    capture_datetime: str
    """Sentinel-2 の撮影日時 ISO-8601 UTC"""

    cloud_cover: float
    """雲量 [%]。50%超の場合は検出信頼度が低下する。"""


# ===========================================================================
# 4. LFM 2.5-VL 推論 入出力
# ===========================================================================

# --- 4a. プロンプトテンプレート ---

FIRE_DETECTION_SYSTEM_PROMPT = """\
You are an expert satellite image analyst specializing in wildfire detection \
using multispectral remote sensing data.

You are analyzing a false-color composite image where:
- RED channel = SWIR 2.2μm (B12): Active fire appears BRIGHT RED/ORANGE; \
burn scars appear DARK RED/BROWN
- GREEN channel = SWIR 1.6μm (B11): Thermal anomalies appear GREEN
- BLUE channel = NIR 842nm (B08): Healthy vegetation appears BLUE/GREEN

This composite penetrates smoke, so fires hidden under smoke clouds \
are visible as bright red/orange areas.

Respond ONLY with a valid JSON object matching the specified schema. \
Do not include any explanation or markdown formatting.\
"""

FIRE_DETECTION_USER_PROMPT = """\
Analyze this satellite false-color composite image for active wildfires.

Image channel encoding:
- RED channel = SWIR 2.2μm: Active fire appears BRIGHT RED/ORANGE
- GREEN channel = SWIR 1.6μm: Thermal anomalies appear GREEN
- BLUE channel = NIR 865nm: Healthy vegetation appears BLUE

IMPORTANT: All confidence values must be realistic probabilities (0.0 to 1.0).
- 0.9-1.0: very certain | 0.7-0.9: likely | 0.5-0.7: possible | 0.3-0.5: unlikely | 0.0-0.3: very uncertain
- Do NOT output 0.0 if fire_detected=true; estimate your actual visual certainty.

Provide your analysis as JSON with this exact schema (all fields required, no nulls for numbers):
{{
  "smoke_detected": <boolean>,
  "smoke_confidence": <float 0.0-1.0, your visual certainty>,
  "smoke_area_fraction": <float 0.0-1.0, fraction of image covered by smoke>,
  "fire_detected": <boolean>,
  "fire_confidence": <float 0.0-1.0, your visual certainty>,
  "fire_area_ha": <float, estimated fire area in hectares, 0.0 if none>,
  "fire_front_bbox": <[x1, y1, x2, y2] in image pixels 0-448, null if no fire>,
  "spread_direction": <"N"|"NE"|"E"|"SE"|"S"|"SW"|"W"|"NW"|null>,
  "severity": <"NONE"|"LOW"|"MEDIUM"|"HIGH"|"CRITICAL">,
  "alert_recommended": <boolean, true if fire_confidence >= 0.6>,
  "description": <string, max 100 chars, concise scene summary>
}}\
"""

FIRE_DETECTION_FT_PROMPT = """\
Examine this satellite false-color composite image (R=SWIR2.2μm, G=SWIR1.6μm, B=NIR).

Does this scene contain active fire or burn scar?
Respond with JSON only: {"fire_detected": true} or {"fire_detected": false}\
"""
"""ファインチューニング・評価用の簡易プロンプト。
学習GTと完全に一致させる。fire_detected の true/false のみを出力させる。"""


# --- 4b. LFM 推論設定 ---

@dataclass
class LFMInferenceConfig:
    """LFM 2.5-VL-450M 推論パラメータ"""
    model_id: str = "LiquidAI/LFM2.5-VL-450M"
    max_new_tokens: int = 512
    temperature: float = 0.1      # 構造化出力のため低温度
    top_p: float = 0.9
    device: str = "cuda"          # RTX 5090
    dtype: str = "bfloat16"       # VRAM効率優先
    use_onnx: bool = False        # エッジ展開時は True に切替


# --- 4c. LFM 生出力 → パース後の型 ---

@dataclass
class FireDetectionResult:
    """LFM 2.5-VL の推論結果 (パース・バリデーション済み)"""

    # --- 煙 ---
    smoke_detected: bool
    smoke_confidence: float        # [0.0, 1.0]
    smoke_area_fraction: float     # [0.0, 1.0]

    # --- 火災 ---
    fire_detected: bool
    fire_confidence: float         # [0.0, 1.0]
    fire_area_ha: float            # [0.0, ∞)
    fire_front_bbox: Optional[tuple[int, int, int, int]]  # (x1,y1,x2,y2) px、なければ None
    spread_direction: Optional[SpreadDirection]
    severity: FireSeverity
    alert_recommended: bool
    description: str               # max 100 chars

    # --- メタ ---
    raw_llm_output: str            # デバッグ用: LLM の生テキスト出力
    inference_time_ms: float       # 推論レイテンシ [ms]


# ===========================================================================
# 5. パイプライン 最終出力
# ===========================================================================

@dataclass
class FireEdgeAlert:
    """パイプラインのエンドツーエンド出力。これがダウンリンクされるペイロード。

    JSON シリアライズ後のバイトサイズ目標: < 2 KB
    (帯域節約: フル画像のダウンリンク (~4MB) の 1/2000)
    """

    # --- シーン識別 ---
    scene_id: str                  # "{satellite}_{datetime}_{lon:.2f}_{lat:.2f}"
    satellite_source: str          # "sentinel-2a" | "sentinel-2b" | "sentinel-2c"
    capture_datetime: str          # 撮影日時 ISO-8601 UTC
    processed_datetime: str        # オンボード処理完了日時 ISO-8601 UTC

    # --- 地理情報 ---
    footprint: tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    satellite_position: SatellitePosition

    # --- 検知結果 ---
    detection: FireDetectionResult

    # --- 品質フラグ ---
    cloud_cover: float             # [0.0, 100.0] %
    data_quality: str              # "GOOD" | "DEGRADED" | "POOR"

    # --- スペクトル指標 (Ground Truth 照合用) ---
    indices: SpectralIndices

    # --- パフォーマンス計測 ---
    total_pipeline_time_ms: float  # データ取得〜出力生成の合計 [ms]
    peak_vram_mb: float            # ピーク VRAM 使用量 [MB]


# ===========================================================================
# 6. NASA FIRMS 照合 入出力
# ===========================================================================

@dataclass
class FIRMSHotspot:
    """NASA FIRMS VIIRS/MODIS の1検知点"""
    lon: float
    lat: float
    brightness: float              # 輝度温度 [K]
    frp: float                     # Fire Radiative Power [MW]
    confidence: str                # "low" | "nominal" | "high"
    datetime: str                  # ISO-8601 UTC


@dataclass
class ValidationResult:
    """FireEdge 検知 vs NASA FIRMS の照合結果"""
    firms_hotspots_in_footprint: int    # フットプリント内の FIRMS 検知点数
    true_positive: bool                 # FIRMS が存在 & FireEdge が検知
    false_positive: bool                # FIRMS なし & FireEdge が検知
    false_negative: bool                # FIRMS あり & FireEdge が未検知
    spatial_overlap_km: Optional[float] # 検知位置のズレ [km]、照合できた場合のみ


# ===========================================================================
# 7. バンド合成 数式サマリ (実装者向けリファレンス)
# ===========================================================================

SPECTRAL_FORMULAS = """
【疑似カラー合成 (fire_composite)】
  R = swir22  (B12: 2190nm)  → 活火: 明るい赤/橙
  G = swir16  (B11: 1610nm)  → 熱異常: 明るい緑
  B = nir     (B08: 842nm)   → 植生: 明るい青

【スペクトル指標】
  NBR2  = (swir16 - swir22) / (swir16 + swir22)
          < 0.0  → 活火または焼跡 [検知閾値]
          < -0.1 → 高確度の活火

  NDVI  = (nir - red) / (nir + red)
          > 0.4  → 健全植生 (焼跡との区別)

  BAI   = 1 / ((0.1 - red)^2 + (0.06 - nir)^2)
          高値  → 焼跡のチャーコール反射特性

【火災画素マスク (NBR2 閾値法)】
  fire_mask = (nbr2 < -0.05) AND (swir22 > 0.15)
  ※ 低 NBR2 単独では焼跡との混同あり。swir22 輝度条件を AND で付加。

【煙マスク (可視光法)】
  brightness = (red + green + blue) / 3
  smoke_mask = (brightness > 0.6) AND (ndvi < 0.1)
  ※ 明るい白/灰で植生でない領域 = 煙
"""
