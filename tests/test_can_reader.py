"""
Unit tests cho src/can_reader/decoder.py.

Kiểm tra:
  - 0x309: sign convention đúng (I > 0 khi xả)
  - 0x311: cell voltage decode đúng (uint16 × 0.0001)
  - 0x201: speed và odo decode đúng
  - 0x30A: SoC BMS decode đúng
  - CAN ID lạ → trả về None
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.can_reader.decoder import decode


# ---------------------------------------------------------------------------
# 0x309 — Dòng điện pack (sign convention)
# ---------------------------------------------------------------------------

def test_decode_0x309_discharge_positive():
    """Khi xe xả (discharge), pack_current_a phải > 0 (project convention)."""
    # CAN raw: âm khi xả. Ví dụ raw = -1500 (0xFA24 as int16) → -15.00 A (raw)
    # Sau sign flip → +15.00 A
    raw_signed = -1500  # A × 100
    raw_uint16 = raw_signed + 65536  # = 64036 = 0xFA24
    data = bytes([0, 0, 0, 0, (raw_uint16 >> 8) & 0xFF, raw_uint16 & 0xFF, 0, 0])
    result = decode(0x309, data)
    assert result is not None
    assert "pack_current_a" in result
    assert result["pack_current_a"] > 0, "Discharge must give positive current (I > 0)"
    assert abs(result["pack_current_a"] - 15.0) < 0.01


def test_decode_0x309_charge_negative():
    """Khi sạc, pack_current_a phải < 0 (project convention)."""
    # CAN raw: dương khi sạc. raw = +500 (0x01F4) → +5.00 A (raw)
    # Sau sign flip → -5.00 A
    raw_uint16 = 500  # 5.00 A × 100
    data = bytes([0, 0, 0, 0, (raw_uint16 >> 8) & 0xFF, raw_uint16 & 0xFF, 0, 0])
    result = decode(0x309, data)
    assert result is not None
    assert result["pack_current_a"] < 0, "Charge must give negative current (I < 0)"
    assert abs(result["pack_current_a"] - (-5.0)) < 0.01


def test_decode_0x309_zero_current():
    """Dòng điện bằng 0 (xe đứng yên, không sạc) → pack_current_a = 0."""
    data = bytes(8)
    result = decode(0x309, data)
    assert result is not None
    assert result["pack_current_a"] == 0.0


# ---------------------------------------------------------------------------
# 0x311 — Cell voltage 1–4
# ---------------------------------------------------------------------------

def test_decode_0x311_cell_voltage():
    """Cell voltage decode đúng: uint16 × 0.0001 → V."""
    # Cell 1 = 32500 raw → 3.25 V (LFP typical)
    raw_cell1 = 32500
    data = bytes([
        (raw_cell1 >> 8) & 0xFF, raw_cell1 & 0xFF,  # cell_01_v
        0x7D, 0x00,                                   # cell_02_v = 32000 → 3.20V
        0x00, 0x00,                                   # cell_03_v = 0
        0x00, 0x00,                                   # cell_04_v = 0
    ])
    result = decode(0x311, data)
    assert result is not None
    assert "cell_01_v" in result
    assert "cell_02_v" in result
    assert abs(result["cell_01_v"] - 3.25) < 0.0001
    assert abs(result["cell_02_v"] - 3.20) < 0.0001


def test_decode_0x31b_two_cells_only():
    """0x31B chỉ có 2 cell (cell 21, 22) — không có cell_23_v."""
    raw = 33000  # 3.30 V
    data = bytes([
        (raw >> 8) & 0xFF, raw & 0xFF,  # cell_21_v
        (raw >> 8) & 0xFF, raw & 0xFF,  # cell_22_v
        0, 0, 0, 0,                     # padding, bỏ qua
    ])
    result = decode(0x31B, data)
    assert result is not None
    assert "cell_21_v" in result
    assert "cell_22_v" in result
    assert "cell_23_v" not in result
    assert abs(result["cell_21_v"] - 3.30) < 0.0001


# ---------------------------------------------------------------------------
# 0x201 — Speed + Odo + Brake
# ---------------------------------------------------------------------------

def test_decode_0x201_speed():
    """Speed decode: uint16(data[0:2]) / 10 → km/h."""
    raw_speed = 350  # 35.0 km/h
    data = bytes([(raw_speed >> 8) & 0xFF, raw_speed & 0xFF, 0, 0, 0, 0, 0, 0])
    result = decode(0x201, data)
    assert result is not None
    assert abs(result["speed_kmh"] - 35.0) < 0.01


def test_decode_0x201_brake():
    """is_brake = True khi data[7] != 0x00."""
    data = bytes([0, 0, 0, 0, 0, 0, 0, 0x01])
    result = decode(0x201, data)
    assert result["is_brake"] is True

    data_no_brake = bytes(8)
    result2 = decode(0x201, data_no_brake)
    assert result2["is_brake"] is False


# ---------------------------------------------------------------------------
# 0x30A — SoC BMS
# ---------------------------------------------------------------------------

def test_decode_0x30a_soc():
    """SoC BMS = data[2], range 0–100."""
    data = bytes([0, 0, 75, 0, 0, 0, 0, 0])
    result = decode(0x30A, data)
    assert result is not None
    assert result["soc_bms"] == 75.0


# ---------------------------------------------------------------------------
# Unknown CAN ID
# ---------------------------------------------------------------------------

def test_decode_unknown_can_id_returns_none():
    """CAN ID không có trong bảng → trả về None."""
    result = decode(0xDEAD, bytes(8))
    assert result is None
