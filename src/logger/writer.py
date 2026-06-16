"""
CSV Runtime Logger — ghi dữ liệu runtime vào CSV để retrain.

Module này ghi mỗi 1 giây một row chứa:
  - 7 signals từ CAN (voltage, current, temp, speed, odometer, + soc_bms)
  - 3 nguồn SoC (BMS, Coulomb Counter, CNN1D model)
  - SoH (State of Health)
  - Range estimation + Wh/km

File output: data/processed/runtime_YYYY-MM-DD_HHMMSS.csv

Schema: database_schema.md §2
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Output directory
DATA_PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


class RuntimeLogger:
    """
    CSV logger cho runtime data.

    Ghi một row mỗi lần write() được gọi (thường là 1Hz từ main loop).

    Ví dụ sử dụng::

        logger = RuntimeLogger(output_dir="data/processed")
        ...
        # In main loop (mỗi 1Hz tick):
        logger.write(state)  # state là SharedState instance
    """

    # CSV header — tên cột
    HEADERS = [
        "timestamp",  # ISO 8601
        "pack_voltage_v",
        "pack_current_a",
        "temp_c",
        "speed_kmh",
        "odo_km",
        "soc_bms",  # SoC #1
        "soc_cc",  # SoC #2
        "soc_model",  # SoC #3
        "soh",
        "range_km",
        "wh_per_km",
        "is_kickstand",
        "is_ready",
        "is_park",
        "is_brake",
        "is_eco",
        "is_sport",
        "err_model",   # soc_model - soc_bms (có dấu, %)
        "err_cc",      # soc_cc    - soc_bms (có dấu, %)
        "mae_model",   # MAE trượt 60s |err_model|
        "mae_cc",      # MAE trượt 60s |err_cc|
    ]

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Khởi tạo logger.

        Args:
            output_dir: Thư mục output (mặc định data/processed/).
                       Tạo nếu chưa tồn tại.
        """
        if output_dir is None:
            output_dir = DATA_PROCESSED_DIR

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tạo filename mới mỗi lần khởi động (tránh ghi đè)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filepath = self.output_dir / f"runtime_{timestamp_str}.csv"

        # Khởi tạo file + header
        self._init_file()

        logger.info(f"RuntimeLogger initialized: {self.filepath}")

    def _init_file(self) -> None:
        """
        Tạo file CSV + ghi header.

        Gọi một lần khi khởi tạo.
        """
        try:
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADERS)
                writer.writeheader()
            logger.info(f"Created CSV file: {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to create CSV file: {e}")
            raise

    def write(self, state) -> None:
        """
        Ghi một row vào CSV từ SharedState.

        Args:
            state: SharedState instance (từ src/main.py).

        Raises:
            Exception: Nếu ghi file thất bại.
        """
        try:
            # Prepare row data
            row = {
                "timestamp": (
                    datetime.fromtimestamp(state.timestamp).isoformat()
                    if state.timestamp
                    else datetime.now().isoformat()
                ),
                "pack_voltage_v": round(state.pack_voltage_v, 2),
                "pack_current_a": round(state.pack_current_a, 2),
                "temp_c": round(state.temp_c, 1),
                "speed_kmh": round(state.speed_kmh, 1),
                "odo_km": getattr(state, "odo_km", 0.0),  # May not be in state yet
                "soc_bms": round(state.soc_bms, 1),
                "soc_cc": round(state.soc_cc, 1),
                "soc_model": round(state.soc_model, 1),
                "soh": round(state.soh, 1),
                "range_km": round(state.range_km, 1),
                "wh_per_km": round(state.wh_per_km, 1),
                "is_kickstand": state.is_kickstand,
                "is_ready":     state.is_ready,
                "is_park":      state.is_park,
                "is_brake":     state.is_brake,
                "is_eco":       getattr(state, "is_eco",    False),
                "is_sport":     getattr(state, "is_sport",  False),
                "err_model":    round(getattr(state, "err_model",  0.0), 2),
                "err_cc":       round(getattr(state, "err_cc",     0.0), 2),
                "mae_model":    round(getattr(state, "mae_model",  0.0), 2),
                "mae_cc":       round(getattr(state, "mae_cc",     0.0), 2),
            }

            # Append row to CSV
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADERS)
                writer.writerow(row)

        except Exception as e:
            logger.error(f"Failed to write CSV row: {e}")
            # Don't raise — continue running even if logging fails

    def get_filepath(self) -> Path:
        """Lấy đường dẫn file CSV đang ghi."""
        return self.filepath

    def get_row_count(self) -> int:
        """
        Đếm số rows đã ghi (không tính header).

        Returns:
            Số rows dữ liệu.
        """
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return sum(1 for _ in f) - 1  # Trừ header
        except Exception as e:
            logger.warning(f"Failed to count rows: {e}")
            return 0

    def get_file_size_mb(self) -> float:
        """
        Lấy kích thước file hiện tại (MB).

        Returns:
            Kích thước file (MB).
        """
        try:
            return self.filepath.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0
