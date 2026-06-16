"""
Thuật toán Coulomb Counting — ước lượng SoC #2.

Nguyên lý: tích phân dòng điện theo thời gian để theo dõi lượng điện
đã tiêu thụ từ mức khởi đầu.

    ΔSoC = -(I × Δt / 3600) / Q_ah × 100   [%]

Trong đó:
  I   = dòng điện (A), I > 0 khi xả (project convention)
  Δt  = khoảng thời gian thực giữa hai lần gọi (giây)
  Q_ah= dung lượng pin hiệu chỉnh (Ah)

Ưu điểm: đơn giản, phản hồi nhanh.
Hạn chế: drift tích lũy theo thời gian nếu sensor sai hoặc dt không chính xác.
→ Reset khi sạc đầy để giới hạn drift mỗi chu kỳ.

Lưu ý về quy ước dấu:
  Module này KHÔNG đảo dấu. Caller (decoder.py hoặc loader.py) đã đảm bảo
  I > 0 khi xả trước khi truyền vào update().
"""

import logging
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Đọc thông số pin từ config
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "battery_specs.yaml"


def _load_battery_specs() -> dict:
    """Đọc thông số pin từ configs/battery_specs.yaml."""
    if yaml is None:
        logger.warning("PyYAML not available, using default capacity.")
        return {}

    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Không tìm thấy configs/battery_specs.yaml, dùng giá trị mặc định.")
        return {}


_SPECS = _load_battery_specs()

# Dung lượng hiệu chỉnh đo thực nghiệm (Ah) — dùng cho tích phân
# Q_eff = 30.65 Ah (đo từ 15 file Evo200_Mixed, xem debugging_notes.md §1)
DEFAULT_CAPACITY_AH: float = _SPECS.get("capacity", {}).get("effective_ah", 30.65)

# Ngưỡng SoC BMS để coi là sạc đầy → reset Coulomb Counter
_CHARGE_COMPLETE_SOC: float = (
    _SPECS.get("coulomb_counter", {}).get("charge_complete_soc_pct", 98.0)
)


# ---------------------------------------------------------------------------
# Class chính
# ---------------------------------------------------------------------------

class CoulombCounter:
    """
    Ước lượng SoC bằng phương pháp Coulomb Counting (tích phân dòng điện).

    Khởi tạo SoC từ giá trị BMS lúc bật máy, sau đó cập nhật liên tục
    mỗi tick 10Hz trong main loop.

    Ví dụ sử dụng trong main loop::

        cc = CoulombCounter(initial_soc=soc_bms_first_sample)
        ...
        soc_cc = cc.update(current_a=decoded["pack_current_a"], dt=tick_dt)
    """

    def __init__(self, capacity_ah: float = DEFAULT_CAPACITY_AH,
                 initial_soc: float = 100.0):
        """
        Khởi tạo Coulomb Counter.

        Args:
            capacity_ah : Dung lượng pin hiệu chỉnh (Ah).
                          Mặc định lấy từ configs/battery_specs.yaml.
            initial_soc : SoC ban đầu (%), nên lấy từ soc_bms lúc khởi động.
                          Mặc định 100% nếu chưa có giá trị BMS.
        """
        self.capacity_ah = capacity_ah
        self.soc_cc      = float(initial_soc)

        logger.info(
            "CoulombCounter khởi tạo: SoC=%.1f%%, Q_eff=%.2fAh",
            self.soc_cc, self.capacity_ah
        )

    def update(self, current_a: float, dt: float) -> float:
        """
        Cập nhật SoC dựa trên dòng điện và khoảng thời gian thực.

        Công thức:
            ΔSoC = -(I × Δt / 3600) / Q_ah × 100
          - I > 0 (xả): ΔSoC âm → SoC giảm ✓
          - I < 0 (sạc): ΔSoC dương → SoC tăng ✓

        dt phải tính từ timestamp thực, KHÔNG hardcode 0.1 hay 1.0.
        Dataset Evo200 có sample rate ~7Hz (dt ≈ 0.143s), không phải 10Hz.
        Dùng dt sai → bias hệ thống (xem debugging_notes.md §1).

        Args:
            current_a: Dòng điện (A). Quy ước: I > 0 khi xả, I < 0 khi sạc.
            dt       : Thời gian thực từ sample trước (giây).
                       Ví dụ: dt = timestamp_current - timestamp_previous.

        Returns:
            SoC hiện tại sau khi cập nhật (%), kẹp trong [0.0, 100.0].
        """
        if dt <= 0:
            return self.soc_cc

        # Tích phân: ΔQ (Ah) = I × Δt / 3600
        # ΔSoC (%) = -ΔQ / Q_ah × 100
        # Dấu âm: dòng dương (xả) làm SoC giảm
        delta_soc = -(current_a * dt / 3600.0) / self.capacity_ah * 100.0

        self.soc_cc = max(0.0, min(100.0, self.soc_cc + delta_soc))

        return self.soc_cc

    def reset(self, new_soc: float) -> None:
        """
        Đặt lại SoC về giá trị mới — dùng khi sạc đầy hoặc re-anchor từ BMS.

        Gọi hàm này khi soc_bms >= ngưỡng sạc đầy (mặc định 98%) để
        giới hạn drift tích lũy mỗi chu kỳ sạc-xả.

        Args:
            new_soc: SoC mới (%), thường là giá trị từ BMS.
        """
        logger.info(
            "CoulombCounter reset: %.1f%% → %.1f%%", self.soc_cc, new_soc
        )
        self.soc_cc = float(new_soc)

    def should_reset(self, soc_bms: float) -> bool:
        """
        Kiểm tra có cần reset không dựa trên SoC BMS.

        Khi BMS báo đã sạc đầy (>= ngưỡng), Coulomb Counter nên
        re-anchor theo BMS để tránh tích lũy drift.

        Args:
            soc_bms: SoC từ BMS (%).

        Returns:
            True nếu nên gọi reset().
        """
        return soc_bms >= _CHARGE_COMPLETE_SOC
