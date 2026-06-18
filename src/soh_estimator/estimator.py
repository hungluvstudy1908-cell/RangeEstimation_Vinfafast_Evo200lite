"""
Ước lượng State of Health (SoH) dựa trên quãng đường tích lũy (odo).

Đây là placeholder kinh nghiệm defend được cho đồ án: tuyến tính theo số chu kỳ
sạc–xả tương đương. Thay bằng SoHCoulombCounter (giáo trình Chương 2) khi đã
gom đủ dữ liệu nguyên cycle sạc–xả sạch.

Tham số đọc từ configs/model.yaml mục soh:
  km_per_full_charge  : quãng đường ứng với 1 chu kỳ đầy (km), mặc định 70
  fade_pct_per_cycle  : % SoH mất mỗi chu kỳ, mặc định 0.0067 (≈50% sau 7500 chu kỳ)
  soh_floor           : SoH tối thiểu trả về (%), mặc định 80.0
"""

from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "model.yaml"


def _load_soh_config() -> dict:
    """Đọc mục soh: từ configs/model.yaml, trả về dict rỗng nếu thiếu."""
    try:
        import yaml

        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("soh", {})
    except (ImportError, FileNotFoundError):
        return {}


class SohEstimator:
    """
    Ước lượng SoH kinh nghiệm theo quãng đường (placeholder defend được cho đồ án).

    Công thức tuyến tính theo số chu kỳ tương đương:
        equiv_cycles = odo_km / km_per_full_charge
        soh = 100.0 - fade_pct_per_cycle * equiv_cycles
        soh = clamp(soh, soh_floor, 100.0)

    Ví dụ với odo=12842 km, km_per_full_charge=70:
        equiv_cycles ≈ 183  →  soh ≈ 98.8%
    """

    def __init__(
        self,
        km_per_full_charge: float = 70.0,
        fade_pct_per_cycle: float = 0.0067,
        soh_floor: float = 80.0,
    ):
        """
        Khởi tạo SohEstimator.

        Args:
            km_per_full_charge: Quãng đường trung bình một lần sạc đầy (km).
            fade_pct_per_cycle: Phần trăm SoH giảm mỗi chu kỳ sạc–xả.
            soh_floor         : Ngưỡng dưới của SoH trả về (%).
        """
        self.km_per_full_charge = km_per_full_charge
        self.fade_pct_per_cycle = fade_pct_per_cycle
        self.soh_floor = soh_floor

    @classmethod
    def from_config(cls) -> "SohEstimator":
        """
        Khởi tạo từ configs/model.yaml mục soh:.

        Trả về instance với tham số mặc định nếu config thiếu.
        """
        cfg = _load_soh_config()
        return cls(
            km_per_full_charge=float(cfg.get("km_per_full_charge", 70.0)),
            fade_pct_per_cycle=float(cfg.get("fade_pct_per_cycle", 0.0067)),
            soh_floor=float(cfg.get("soh_floor", 80.0)),
        )

    def estimate(self, odo_km: float) -> float:
        """
        Ước lượng SoH từ tổng quãng đường tích lũy.

        Args:
            odo_km: Odometer tích lũy (km), lấy từ CAN 0x201 → state.odo_km.

        Returns:
            SoH ước lượng (%, trong khoảng [soh_floor, 100.0]).
        """
        # Số chu kỳ sạc–xả tương đương từ tổng quãng đường
        equiv_cycles = max(0.0, odo_km) / self.km_per_full_charge
        soh = 100.0 - self.fade_pct_per_cycle * equiv_cycles
        return max(self.soh_floor, min(100.0, soh))
