"""
Ước lượng quãng đường còn lại (Range) dựa trên SoC, SoH, và hành vi lái.

Thuật toán gồm 4 chương:

1. EWMA Baseline (Chương 1): Tính mức tiêu thụ năng lượng trung bình
   - Dùng exponential weighted moving average để theo dõi wh_per_km
   - Cập nhật liên tục khi có dữ liệu mới
   - Làm mịn các biến động ngắn hạn

2. Coulomb Counting cho SoH (Chương 3): Ước lượng State of Health
   - Theo dõi độ giảm dung lượng pin theo thời gian
   - Dùng trong công thức range cuối cùng

3. Behavior Features (Chương 3): Trích 3 tính chất lái trong cửa sổ 60s
   - avg_speed_kmh: tốc độ trung bình
   - accel_std_mps2: độ lệch chuẩn gia tốc (biến động tốc độ)
   - stop_ratio: tỷ lệ thời gian tốc độ < 5 km/h

4. Linear Regression Factor (Chương 4): Kết hợp behavior
   - range_factor = α × avg_speed + β × accel_std + γ × stop_ratio + offset
   - Dùng để điều chỉnh mức tiêu thụ dựa trên hành vi lái

Công thức cuối cùng (Chương 4):
    range_km = (soc_pct / 100 × capacity_wh × soh_pct / 100)
             / (wh_per_km_ewma × behavior_factor)

Tham số trong configs/range_estimator.yaml.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Đọc cấu hình từ YAML
_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "range_estimator.yaml"


def _load_range_config() -> dict:
    """Đọc cấu hình từ configs/range_estimator.yaml."""
    try:
        import yaml

        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError):
        logger.warning(
            "Không đọc được configs/range_estimator.yaml, dùng giá trị mặc định"
        )
        return {}


_RANGE_CONFIG = _load_range_config()

# Mặc định nếu config không có
DEFAULT_PACK_CAPACITY_WH = (
    _RANGE_CONFIG.get("battery", {}).get("pack_capacity_wh", 2206.8)
)
DEFAULT_EWMA_ALPHA = _RANGE_CONFIG.get("ewma", {}).get("alpha", 0.3)


# ============================================================================
# Chương 1: EWMA Baseline — Tính mức tiêu thụ năng lượng trung bình
# ============================================================================


def update_ewma_consumption(
    new_wh_per_km: float, alpha: float = DEFAULT_EWMA_ALPHA, prev_ewma: float = None
) -> float:
    """
    Cập nhật EWMA của mức tiêu thụ năng lượng.

    Công thức EWMA:
        ewma_new = α × new_value + (1 - α) × ewma_prev

    Với α = 0.3 (mặc định):
    - Giá trị mới được giáo 30%, lịch sử được giáo 70%
    - Giảm tác động của các outlier (điều kiện lái khác thường)
    - Điểm cân bằng giữa phản ứng nhanh và ổn định

    Args:
        new_wh_per_km: Mức tiêu thụ mới đo được (Wh/km).
        alpha: Tham số smoothing (0.0–1.0), mặc định 0.3.
        prev_ewma: Giá trị EWMA trước đó. Nếu None, khởi tạo bằng new_wh_per_km.

    Returns:
        Giá trị EWMA mới (Wh/km).
    """
    if prev_ewma is None:
        return float(new_wh_per_km)

    ewma_new = alpha * new_wh_per_km + (1.0 - alpha) * prev_ewma
    return float(ewma_new)


# ============================================================================
# Chương 3: Behavior Features — Trích tính chất lái từ cửa sổ 60s
# ============================================================================


def compute_behavior_features(
    speed_window: np.ndarray, current_window: np.ndarray
) -> Dict[str, float]:
    """
    Tính 3 behavior features từ cửa sổ dữ liệu 60 sample (~60 giây).

    Features:
      1. avg_speed_kmh: Tốc độ trung bình trong cửa sổ
      2. accel_std_mps2: Độ lệch chuẩn gia tốc (độ biến động tốc độ)
      3. stop_ratio: Tỷ lệ thời gian tốc độ < 5 km/h (khoảng dừng/chờ)

    Args:
        speed_window: np.ndarray shape (N,), tốc độ (km/h).
        current_window: np.ndarray shape (N,), dòng điện (A).

    Returns:
        Dict với keys: 'avg_speed_kmh', 'accel_std_mps2', 'stop_ratio'.
    """
    # Feature 1: avg_speed_kmh
    avg_speed = float(np.mean(speed_window))

    # Feature 2: accel_std_mps2
    # Chuyển km/h → m/s, rồi tính đạo hàm (gia tốc)
    speed_ms = speed_window / 3.6  # 1 m/s = 3.6 km/h
    accel = np.diff(speed_ms)  # Gia tốc (m/s² giả sử dt=1s)
    accel_std = float(np.std(accel)) if len(accel) > 1 else 0.0

    # Feature 3: stop_ratio
    # Tỷ lệ sample có tốc độ < 5 km/h
    n_stopped = np.sum(speed_window < 5.0)
    stop_ratio = float(n_stopped / len(speed_window))

    return {
        "avg_speed_kmh": avg_speed,
        "accel_std_mps2": accel_std,
        "stop_ratio": stop_ratio,
    }


# ============================================================================
# Chương 4: Linear Regression Factor — Kết hợp behavior
# ============================================================================


def compute_behavior_factor(
    features: Dict[str, float], coefficients: Dict[str, float] = None
) -> float:
    """
    Tính behavior factor từ các features.

    Công thức:
        behavior_factor = coeff_speed × avg_speed
                        + coeff_accel × accel_std
                        + coeff_stop × stop_ratio
                        + offset

    Giải thích vật lý:
    - Tốc độ cao → tiêu thụ nhiều → factor cao → range giảm ✓
    - Gia tốc cao (lái bốc) → tiêu thụ nhiều → factor cao → range giảm ✓
    - Tỷ lệ dừng cao (lái trong thành phố) → tiêu thụ ít → factor thấp → range tăng ✓

    Args:
        features: Dict từ compute_behavior_features().
        coefficients: Dict với keys 'speed', 'accel', 'stop', 'offset'.
                     Nếu None, dùng giá trị mặc định từ config.

    Returns:
        Behavior factor (thường 0.8–1.2).
    """
    if coefficients is None:
        # Lấy từ config hoặc dùng giá trị mặc định hợp lý
        try:
            import yaml

            with open(_CONFIG_PATH, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            behavior_coeffs = cfg.get("behavior_factor", {}).get("coefficients", {})
        except:
            behavior_coeffs = {}

        coefficients = {
            "speed": behavior_coeffs.get("speed", 0.005),  # mỗi km/h +0.5% factor
            "accel": behavior_coeffs.get("accel", 0.02),  # mỗi m/s² +2% factor
            "stop": behavior_coeffs.get("stop", -0.3),  # 100% dừng → -30% factor
            "offset": behavior_coeffs.get("offset", 1.0),  # baseline = 1.0
        }

    factor = (
        coefficients["offset"]
        + coefficients["speed"] * features["avg_speed_kmh"]
        + coefficients["accel"] * features["accel_std_mps2"]
        + coefficients["stop"] * features["stop_ratio"]
    )

    # Clamp vào [0.3, 2.0] để tránh outlier
    factor = np.clip(factor, 0.3, 2.0)

    return float(factor)


# ============================================================================
# Range Estimator — Class chính
# ============================================================================


class RangeEstimator:
    """
    Ước lượng quãng đường còn lại từ SoC, SoH, và hành vi lái.

    Khởi tạo một lần, rồi gọi update_and_estimate() mỗi tick 10Hz.

    Ví dụ sử dụng::

        estimator = RangeEstimator(pack_capacity_wh=2206.8, ewma_alpha=0.3)
        ...
        range_km = estimator.update_and_estimate(
            soc_pct=75.0,
            soh_pct=95.8,
            speed_window=speeds[-60:],
            current_window=currents[-60:]
        )
    """

    def __init__(
        self, pack_capacity_wh: float = DEFAULT_PACK_CAPACITY_WH, ewma_alpha: float = DEFAULT_EWMA_ALPHA
    ):
        """
        Khởi tạo Range Estimator.

        Args:
            pack_capacity_wh: Dung lượng pin tính theo năng lượng (Wh).
                             Default 2206.8 Wh = 72V × 30.65Ah.
            ewma_alpha: Tham số smoothing cho EWMA tiêu thụ (0.0–1.0).
        """
        self.pack_capacity_wh = pack_capacity_wh
        self.ewma_alpha = ewma_alpha

        # EWMA tiêu thụ năng lượng — khởi tạo khi có dữ liệu
        self.wh_per_km_ewma = None

        # Đọc tham số làm mượt từ config (với fallback hardcode)
        _sm = _RANGE_CONFIG.get("smoothing", {})
        self.v_min_kmh        = float(_sm.get("v_min_kmh",        5.0))
        self.wh_per_km_min    = float(_sm.get("wh_per_km_min",   18.0))
        self.wh_per_km_max    = float(_sm.get("wh_per_km_max",   60.0))
        self.range_ewma_alpha = float(_sm.get("range_ewma_alpha",  0.05))
        self.range_km_max     = float(_sm.get("range_km_max",    120.0))
        self.range_ewma       = None   # seed lần đầu từ range_raw, không bò từ 0

        logger.info(
            f"RangeEstimator khởi tạo: capacity={pack_capacity_wh:.1f}Wh, alpha={ewma_alpha}"
        )

    def update_and_estimate(
        self,
        soc_pct: float,
        soh_pct: float,
        speed_window: np.ndarray,
        current_window: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Cập nhật EWMA tiêu thụ và ước lượng quãng đường còn lại.

        Thiết kế làm mượt 3 tầng + freeze:
        (a) Gate v_min: chỉ cập nhật Wh/km khi avg_speed >= v_min_kmh (mặc định 5 km/h).
            Tránh chia P cho tốc độ gần 0 → Wh/km nổ → range về 0 hoặc vài km.
        (b) Clamp input Wh/km vào [wh_per_km_min, wh_per_km_max] trước khi feed EWMA.
            Chặn outlier đầu phiên trước khi EWMA kịp hội tụ.
        (c) EWMA thứ hai trên output range_km (α=range_ewma_alpha, τ≈20s ở 1Hz).
            Cần vì behavior_factor nhân vào SAU Wh/km EWMA nên gây jitter mỗi tick;
            EWMA output triệt jitter đó mà không làm chậm Wh/km input.
            Kèm clamp cứng [0, range_km_max] để chặn ca nổ 476 km.
        (d) Freeze range khi xe dừng — dùng tốc độ tức thời (5 sample cuối ≈ 0.5s)
            thay vì avg_speed (60s) để phản ứng ngay khi xe dừng.
            Lý do: coeff behavior_factor là placeholder (chưa train), stop_ratio→1
            khi dừng làm factor≈0.7 → consumption↓ → range leo. Freeze là stopgap
            đúng ngữ nghĩa theo §4.3 giáo trình cho đến khi có coeff train thật.
            Cold-start đang dừng (range_ewma is None): cho tính 1 lần để seed.

        Args:
            soc_pct: State of Charge (%) từ BMS hoặc CC.
            soh_pct: State of Health (%) từ model hoặc giá trị cố định.
            speed_window: np.ndarray (60,) tốc độ (km/h).
            current_window: np.ndarray (60,) dòng điện (A, I>0 = discharge).

        Returns:
            Tuple (range_km, wh_per_km_ewma):
            - range_km: Quãng đường ước lượng đã làm mượt / đóng băng (km).
            - wh_per_km_ewma: Mức tiêu thụ EWMA hiện tại (Wh/km), dùng để log.
        """
        avg_current = float(np.mean(current_window))
        avg_speed   = float(np.mean(speed_window))

        # Công suất tức thời (W) — điện áp pack hardcode 72V
        pack_voltage = 72.0
        power_w = pack_voltage * abs(avg_current)

        # (a) Gate v_min: bỏ update khi xe gần dừng, giữ wh_per_km_ewma cũ
        if avg_speed > self.v_min_kmh:
            wh_per_km_new = power_w / avg_speed
            # (b) Clamp input trước EWMA — chặn outlier tốc độ thấp còn lọt qua
            wh_per_km_new = float(np.clip(wh_per_km_new, self.wh_per_km_min, self.wh_per_km_max))
        else:
            wh_per_km_new = self.wh_per_km_ewma if self.wh_per_km_ewma else 50.0

        self.wh_per_km_ewma = update_ewma_consumption(
            wh_per_km_new, alpha=self.ewma_alpha, prev_ewma=self.wh_per_km_ewma
        )

        # (d) Freeze range khi xe dừng/chạy chậm.
        # Dùng 5 sample cuối (~0.5s ở 10Hz) thay avg_speed (60s) để phản ứng ngay.
        recent_speed = float(np.mean(speed_window[-5:])) if len(speed_window) >= 5 else avg_speed
        if recent_speed < self.v_min_kmh and self.range_ewma is not None:
            return (float(self.range_ewma), float(self.wh_per_km_ewma))
        # cold-start đang dừng: range_ewma is None → tính 1 lần để seed, tick sau freeze

        # Tính behavior features + factor (Chương 3–4, không thay đổi)
        features        = compute_behavior_features(speed_window, current_window)
        behavior_factor = compute_behavior_factor(features)

        # Năng lượng khả dụng (Wh) và mức tiêu thụ hiệu dụng (Wh/km)
        available_energy_wh  = soc_pct / 100.0 * self.pack_capacity_wh * soh_pct / 100.0
        effective_consumption = self.wh_per_km_ewma * behavior_factor

        # (c) Range raw → clamp → EWMA output
        range_raw = available_energy_wh / effective_consumption if effective_consumption > 0 else 0.0
        range_raw = float(np.clip(range_raw, 0.0, self.range_km_max))

        if self.range_ewma is None:
            self.range_ewma = range_raw          # seed từ giá trị thật, không bò từ 0
        else:
            a = self.range_ewma_alpha
            self.range_ewma = a * range_raw + (1.0 - a) * self.range_ewma

        return float(self.range_ewma), float(self.wh_per_km_ewma)

    def get_state(self) -> dict:
        """
        Lấy trạng thái nội bộ (chủ yếu để debug/log).

        Returns:
            Dict chứa: wh_per_km_ewma, pack_capacity_wh, ...
        """
        return {
            "wh_per_km_ewma": self.wh_per_km_ewma,
            "range_ewma": self.range_ewma,
            "pack_capacity_wh": self.pack_capacity_wh,
            "ewma_alpha": self.ewma_alpha,
            "range_ewma_alpha": self.range_ewma_alpha,
        }
