"""
Test RuntimeLogger CSV writing.

Smoke tests để kiểm tra:
- Logger khởi tạo được
- Ghi row vào CSV được
- File created với header đúng
"""

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger.writer import RuntimeLogger


# Mock SharedState for testing (avoid importing main.py with all dependencies)
@dataclass
class MockState:
    """Simplified SharedState for testing."""
    timestamp: float = None
    pack_voltage_v: float = 72.0
    pack_current_a: float = 0.0
    temp_c: float = 25.0
    speed_kmh: float = 0.0
    odo_km: float = 0.0
    soc_bms: float = 50.0
    soc_cc: float = 50.0
    soc_model: float = 50.0
    soh: float = 95.8
    range_km: float = 0.0
    wh_per_km: float = 50.0


def test_logger_init():
    """Khởi tạo logger trong temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = RuntimeLogger(output_dir=Path(tmpdir))
        assert logger.filepath.exists()
        print(f"[OK] Logger initialized: {logger.filepath}")


def test_logger_write():
    """Ghi row vào CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = RuntimeLogger(output_dir=Path(tmpdir))

        # Create dummy state
        state = MockState()
        state.timestamp = 1700000000.0  # Unix timestamp
        state.pack_voltage_v = 72.5
        state.pack_current_a = 15.3
        state.temp_c = 25.0
        state.speed_kmh = 35.0
        state.soc_bms = 75.0
        state.soc_cc = 74.5
        state.soc_model = 74.8
        state.soh = 95.8
        state.range_km = 123.4
        state.wh_per_km = 50.0

        # Write to CSV
        logger.write(state)

        # Verify row was written
        row_count = logger.get_row_count()
        assert row_count == 1, f"Expected 1 row, got {row_count}"
        print(f"[OK] Logger wrote 1 row. File size: {logger.get_file_size_mb():.3f}MB")


def test_logger_multiple_writes():
    """Ghi nhiều rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = RuntimeLogger(output_dir=Path(tmpdir))

        # Write 10 rows
        state = MockState()
        for i in range(10):
            state.soc_bms = 75.0 - i
            state.soc_cc = 74.5 - i
            logger.write(state)

        row_count = logger.get_row_count()
        assert row_count == 10, f"Expected 10 rows, got {row_count}"
        print(f"[OK] Logger wrote 10 rows")


def test_logger_header():
    """Kiểm tra CSV header."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = RuntimeLogger(output_dir=Path(tmpdir))

        # Read first line (header)
        with open(logger.filepath, "r") as f:
            header = f.readline().strip()

        expected_header = ",".join(RuntimeLogger.HEADERS)
        assert header == expected_header, f"Header mismatch:\nGot: {header}\nExpected: {expected_header}"
        print(f"[OK] CSV header correct: {len(RuntimeLogger.HEADERS)} columns")


if __name__ == "__main__":
    print("Testing RuntimeLogger...")
    print()

    test_logger_init()
    test_logger_write()
    test_logger_multiple_writes()
    test_logger_header()

    print()
    print("All smoke tests passed!")
