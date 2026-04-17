"""
[5] Validator — NASA FIRMS Ground Truth 照合
============================================
FireEdge の検知結果を NASA FIRMS VIIRS/MODIS ホットスポットと照合し、
精度 (TP/FP/FN) を定量評価する。

入力 : FireEdgeAlert + フットプリント (lon_min, lat_min, lon_max, lat_max)
出力 : ValidationResult (interfaces.py)

NASA FIRMS API:
  https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/{area}/{days}
  MAP_KEY は環境変数 FIRMS_MAP_KEY または引数で渡す。
  無料登録: https://firms.modaps.eosdis.nasa.gov/api/

オフライン時:
  CSV を data/firms_cache/ に手動で置けばそちらを使う。
"""

from __future__ import annotations

import csv
import io
import math
import os
from datetime import datetime, timezone
from typing import Optional

import requests

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import (
    FIRMSHotspot,
    FireEdgeAlert,
    ValidationResult,
)

# FIRMS API の設定
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CACHE_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "firms_cache")

# 空間的マッチング許容距離 [km]
SPATIAL_TOLERANCE_KM = 5.0


class FIRMSValidator:
    """NASA FIRMS ホットスポットデータを使った FireEdge 精度評価器。"""

    def __init__(self, map_key: Optional[str] = None) -> None:
        self.map_key = map_key or os.environ.get("FIRMS_MAP_KEY", "")
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # メイン API
    # ------------------------------------------------------------------

    def validate(self, alert: FireEdgeAlert, day_range: int = 1) -> ValidationResult:
        """
        FireEdgeAlert のフットプリントに含まれる FIRMS ホットスポットを取得し、
        FireEdge の検知結果と照合する。

        Args:
            alert     : pipeline.run() の出力
            day_range : FIRMS から取得する日数 (1〜10)

        Returns:
            ValidationResult
        """
        fp = alert.footprint  # (lon_min, lat_min, lon_max, lat_max)
        hotspots = self.fetch_hotspots(fp, alert.capture_datetime, day_range)

        firms_count = len(hotspots)
        fire_detected = alert.detection.fire_detected

        true_positive  = firms_count > 0 and fire_detected
        false_positive = firms_count == 0 and fire_detected
        false_negative = firms_count > 0 and not fire_detected

        # 最近傍 FIRMS 点との距離
        overlap_km: Optional[float] = None
        if alert.detection.fire_front_bbox and hotspots:
            bbox = alert.detection.fire_front_bbox
            # bbox 中心を footprint 座標に変換
            img_size = 448
            bbox_cx = (bbox[0] + bbox[2]) / 2 / img_size  # [0,1]
            bbox_cy = (bbox[1] + bbox[3]) / 2 / img_size  # [0,1]
            pred_lon = fp[0] + bbox_cx * (fp[2] - fp[0])
            pred_lat = fp[1] + (1 - bbox_cy) * (fp[3] - fp[1])  # y軸反転
            min_dist = min(
                _haversine(pred_lon, pred_lat, h.lon, h.lat)
                for h in hotspots
            )
            overlap_km = min_dist

        return ValidationResult(
            firms_hotspots_in_footprint=firms_count,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            spatial_overlap_km=overlap_km,
        )

    def fetch_hotspots(
        self,
        footprint: tuple[float, float, float, float],
        capture_datetime: str,
        day_range: int = 1,
    ) -> list[FIRMSHotspot]:
        """
        FIRMS API または CSV キャッシュから指定フットプリント内のホットスポットを返す。

        footprint: (lon_min, lat_min, lon_max, lat_max)
        """
        lon_min, lat_min, lon_max, lat_max = footprint
        area_str = f"{lon_min:.4f},{lat_min:.4f},{lon_max:.4f},{lat_max:.4f}"

        # キャッシュチェック
        cache_path = self._cache_path(footprint, capture_datetime, day_range)
        if os.path.exists(cache_path):
            print(f"[Validator] キャッシュ読み込み: {cache_path}")
            with open(cache_path, encoding="utf-8") as f:
                return _parse_firms_csv(f.read(), footprint)

        # API 呼び出し
        if not self.map_key:
            print("[Validator] FIRMS_MAP_KEY が未設定。キャッシュなしでスキップ。")
            print("           https://firms.modaps.eosdis.nasa.gov/api/ で無料取得可能。")
            return []

        url = f"{FIRMS_BASE}/{self.map_key}/VIIRS_SNPP_NRT/{area_str}/{day_range}"
        print(f"[Validator] FIRMS API 呼び出し: {url}")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            csv_text = r.text

            # キャッシュ保存
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(csv_text)

            return _parse_firms_csv(csv_text, footprint)
        except requests.RequestException as e:
            print(f"[Validator] FIRMS API エラー: {e}")
            return []

    def summarize(self, result: ValidationResult) -> str:
        """ValidationResult を人間可読な文字列に変換する。"""
        lines = [
            "=== FIRMS Ground Truth 照合結果 ===",
            f"  フットプリント内 FIRMS ホットスポット数: {result.firms_hotspots_in_footprint}",
            f"  True Positive  (FireEdge ✓, FIRMS ✓): {result.true_positive}",
            f"  False Positive (FireEdge ✓, FIRMS ✗): {result.false_positive}",
            f"  False Negative (FireEdge ✗, FIRMS ✓): {result.false_negative}",
        ]
        if result.spatial_overlap_km is not None:
            lines.append(f"  最近傍 FIRMS 点との距離: {result.spatial_overlap_km:.2f} km")
            match = result.spatial_overlap_km <= SPATIAL_TOLERANCE_KM
            lines.append(f"  空間一致 ({SPATIAL_TOLERANCE_KM}km 以内): {'✓' if match else '✗'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def _cache_path(
        self,
        footprint: tuple[float, float, float, float],
        dt: str,
        day_range: int,
    ) -> str:
        date_str = dt[:10].replace("-", "")
        fp_str   = "_".join(f"{v:.2f}" for v in footprint).replace("-", "m")
        return os.path.join(CACHE_DIR, f"firms_{date_str}_{fp_str}_d{day_range}.csv")


# ------------------------------------------------------------------
# CSV パーサ
# ------------------------------------------------------------------

def _parse_firms_csv(
    csv_text: str,
    footprint: tuple[float, float, float, float],
) -> list[FIRMSHotspot]:
    """FIRMS CSV テキスト → FIRMSHotspot リスト (フットプリント内のみ)。"""
    lon_min, lat_min, lon_max, lat_max = footprint
    hotspots = []

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        try:
            lat = float(row.get("latitude", row.get("lat", 0)))
            lon = float(row.get("longitude", row.get("lon", 0)))

            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue

            # 輝度温度: bright_ti4 (VIIRS) or brightness (MODIS)
            brightness = float(
                row.get("bright_ti4") or row.get("brightness") or 0
            )
            frp = float(row.get("frp", 0) or 0)
            confidence = str(row.get("confidence", "nominal")).strip().lower()
            acq_date = row.get("acq_date", "")
            acq_time = row.get("acq_time", "0000")
            dt_str = f"{acq_date}T{acq_time[:2]}:{acq_time[2:]}:00Z" if acq_date else ""

            hotspots.append(FIRMSHotspot(
                lon=lon,
                lat=lat,
                brightness=brightness,
                frp=frp,
                confidence=confidence,
                datetime=dt_str,
            ))
        except (ValueError, KeyError):
            continue

    return hotspots


# ------------------------------------------------------------------
# 距離計算
# ------------------------------------------------------------------

def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """2点間の大圏距離 [km] (Haversine 公式)。"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ------------------------------------------------------------------
# スタンドアロン実行
# ------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from src.pipeline import FireEdgePipeline

    print("FireEdge × FIRMS 精度評価デモ")
    pipeline  = FireEdgePipeline()
    alert     = pipeline.run(lon=142.0, lat=-30.0, timestamp="2026-01-15T05:00:00Z")

    validator = FIRMSValidator()
    result    = validator.validate(alert)
    print(validator.summarize(result))
    print(f"\nFireEdge 検知結果: fire={alert.detection.fire_detected}, "
          f"conf={alert.detection.fire_confidence:.2f}, severity={alert.detection.severity.value}")
