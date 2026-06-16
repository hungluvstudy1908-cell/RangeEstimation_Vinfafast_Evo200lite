"""
Giải mã tín hiệu CAN từ xe VinFast Evo 200 Lite.

Module này nhận (can_id, data) từ reader.py và trả về dict tín hiệu
đã được chuyển sang đơn vị vật lý, tên snake_case, và đúng quy ước dấu.

Nguyên tắc quan trọng:
  - Đây là nơi DUY NHẤT đảo dấu dòng điện cho dữ liệu runtime CAN.
    CAN raw: I < 0 khi xả. Project convention: I > 0 khi xả.
    (Tương tự loader.py đảo dấu cho dữ liệu training CSV.)
  - Mỗi hàm _decode_* chỉ xử lý đúng một CAN ID — không lẫn lộn.
  - Hàm decode() trả về dict tín hiệu của khung đó, hoặc None nếu
    CAN ID không nhận ra (log warning và bỏ qua).

CAN ID lookup được đọc từ configs/can_ids.yaml tại lúc import.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load CAN ID lookup từ config — tránh hardcode số hex trong code
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "can_ids.yaml"

def _load_known_ids() -> set[int]:
    """Đọc tập hợp CAN ID đã biết từ configs/can_ids.yaml."""
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return {frame["id"] for frame in cfg.get("frames", [])}
    except FileNotFoundError:
        logger.warning("Không tìm thấy configs/can_ids.yaml, dùng ID mặc định.")
        return set()

_KNOWN_IDS = _load_known_ids()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode(can_id: int, data: bytes) -> dict | None:
    """
    Giải mã một khung CAN thành dict tín hiệu vật lý.

    Dispatch đến hàm decode chuyên biệt theo CAN ID.
    Trả về None và log warning nếu CAN ID không nhận ra.

    Args:
        can_id: Số nguyên CAN ID (ví dụ 0x30A = 778).
        data  : 8 byte payload của khung CAN.

    Returns:
        Dict tín hiệu đã decode, ví dụ {'soc_bms': 75.0}.
        None nếu CAN ID không có trong bảng.
    """
    if can_id == 0x102:
        return _decode_vehicle_status(data)
    if can_id == 0x201:
        return _decode_vehicle_motion(data)
    if can_id == 0x30A:
        return _decode_bms_soc(data)
    if can_id == 0x320:
        return _decode_bms_temperature(data)
    if can_id == 0x309:
        return _decode_bms_current(data)
    if can_id in (0x311, 0x312, 0x313, 0x314, 0x31A, 0x31B):
        return _decode_cell_voltage(can_id, data)

    # CAN ID lạ — chỉ log nếu không nằm trong whitelist đã biết
    if can_id not in _KNOWN_IDS:
        logger.debug("CAN ID không nhận ra: 0x%03X — bỏ qua", can_id)
    return None


# ---------------------------------------------------------------------------
# Private decode functions — mỗi hàm = một CAN ID
# ---------------------------------------------------------------------------

def _decode_vehicle_status(data: bytes) -> dict:
    """
    Giải mã CAN 0x102 — Trạng thái vận hành xe.

    data[0] encoding (4 trạng thái):
      0x30 = READY + ECO   (nibble thấp 0 = eco, nibble cao 3 = ready)
      0x31 = READY + SPORT (nibble thấp 1 = sport)
      0x20 = PARK  + ECO
      0x21 = PARK  + SPORT
    Giá trị khác → is_park=True (an toàn mặc định).

    Signals:
      is_ready    : True khi data[0] in {0x30, 0x31}
      is_kickstand: True khi data[1] != 0x00
      is_park     : nghịch đảo is_ready (trạng thái lạ → park)
      is_sport    : True khi nibble thấp = 1 (data[0] in {0x21, 0x31})
      is_eco      : True khi nibble thấp = 0 (data[0] in {0x20, 0x30})
    """
    is_ready     = data[0] in (0x30, 0x31)
    is_kickstand = data[1] != 0x00
    is_sport     = data[0] in (0x21, 0x31)
    is_eco       = data[0] in (0x20, 0x30)
    return {
        "is_ready"    : is_ready,
        "is_kickstand": is_kickstand,
        "is_park"     : not is_ready,
        "is_sport"    : is_sport,
        "is_eco"      : is_eco,
    }


def _decode_vehicle_motion(data: bytes) -> dict:
    """
    Giải mã CAN 0x201 — Tốc độ, đồng hồ km, phanh.

    Signals:
      speed_kmh: Tốc độ xe (km/h) = uint16(data[0:2]) / 10
      odo_km   : Đồng hồ km tích lũy (km) = uint24(data[3:6]) / 10
      is_brake : True khi đang đạp phanh (data[7] != 0x00)
    """
    raw_speed = (data[0] << 8) | data[1]
    raw_odo   = (data[3] << 16) | (data[4] << 8) | data[5]
    return {
        "speed_kmh": raw_speed / 10.0,
        "odo_km"   : raw_odo   / 10.0,
        "is_brake" : data[7] != 0x00,
    }


def _decode_bms_soc(data: bytes) -> dict:
    """
    Giải mã CAN 0x30A — SoC từ BMS xe.

    Signal:
      soc_bms: State of Charge từ BMS (%) = data[2], dải 0–100.
    """
    return {"soc_bms": float(data[2])}


def _decode_bms_temperature(data: bytes) -> dict:
    """
    Giải mã CAN 0x320 — Nhiệt độ module pin.

    Signal:
      temp_c: Nhiệt độ pin (°C) = data[3].
    """
    return {"temp_c": float(data[3])}


def _decode_bms_current(data: bytes) -> dict:
    """
    Giải mã CAN 0x309 — Dòng điện pack (có dấu).

    Decode int16 từ data[4:6], chia 100 để ra Ampere.

    ĐẢO DẤU tại đây:
      CAN raw  : I < 0 khi xả (convention của BMS xe).
      Project  : I > 0 khi xả (convention toàn hệ thống).
    Đảo dấu MỘT LẦN DUY NHẤT — Coulomb counter và model downstream
    đều giả định I > 0 khi xả.

    Signals:
      pack_current_a: Dòng điện pack (A), I > 0 khi xả.
    """
    raw = (data[4] << 8) | data[5]

    # Chuyển unsigned int16 sang signed int16
    if raw > 32767:
        raw -= 65536

    # raw / 100 → Ampere; đảo dấu vì CAN raw âm khi xả
    pack_current_a = -(raw / 100.0)

    return {"pack_current_a": pack_current_a}


def _decode_cell_voltage(can_id: int, data: bytes) -> dict:
    """
    Giải mã các frame CAN chứa điện áp cell pin (0x311–0x31B).

    Mỗi frame chứa 4 cell (trừ 0x31B chứa 2 cell cuối).
    Mỗi cell: uint16 × 0.0001 → Volt (dải thực tế ~3.0–3.65V/cell LFP).

    Mapping CAN ID → chỉ số cell (0-indexed):
      0x311: cell 0–3
      0x312: cell 4–7
      0x313: cell 8–11
      0x314: cell 12–15
      0x31A: cell 16–19
      0x31B: cell 20–21

    Args:
        can_id: CAN ID xác định nhóm cell.
        data  : 8 byte payload.

    Returns:
        Dict với key dạng 'cell_XX_v' (vd 'cell_01_v'), giá trị là Volt.
    """
    # Chỉ số cell đầu tiên của mỗi frame
    _FIRST_CELL = {
        0x311: 0,
        0x312: 4,
        0x313: 8,
        0x314: 12,
        0x31A: 16,
        0x31B: 20,
    }

    first = _FIRST_CELL[can_id]
    result = {}

    # Số cell trong frame: 4 cell bình thường, 2 cell cho frame cuối (0x31B)
    n_cells = 2 if can_id == 0x31B else 4

    for i in range(n_cells):
        raw = (data[i * 2] << 8) | data[i * 2 + 1]
        cell_idx = first + i + 1          # 1-indexed để tên nhất quán với xe
        key = f"cell_{cell_idx:02d}_v"
        result[key] = raw * 0.0001

    return result
