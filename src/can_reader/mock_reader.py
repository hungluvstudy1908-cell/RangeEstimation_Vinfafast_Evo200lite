"""
Mock CAN reader cho mục đích test — replay dữ liệu từ CSV thay vì đọc hardware.

Dùng khi không có xe thật hoặc Waveshare USB-CAN. Kích hoạt bằng env var:
    MOCK_CAN=1
    MOCK_CSV=data/raw/Evo200_Mixed1.csv  (tùy chọn, để đổi file)
    MOCK_CSV_START_ROW=513               (tùy chọn, bỏ qua N row đầu)

Interface giống WaveshareReader: connect(), disconnect(), read_frames().
read_frames() trả về list[dict] thay vì list[tuple[int, bytes]] — main loop
dùng isinstance(frame, dict) để nhận ra mock frame và bỏ qua decoder.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Đường dẫn tuyệt đối — resolve từ vị trí file này, không phụ thuộc working directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CSV = str(_PROJECT_ROOT / "data" / "raw" / "Evo200_Mixed1.csv")


class MockCanReader:
    """
    Replay dữ liệu từ CSV với tốc độ ~10Hz, thay thế WaveshareReader khi test.

    Mỗi lần read_frames() được gọi, trả về một row dưới dạng decoded dict
    (bypass decoder thật). Khi hết CSV, tự động loop về đầu để dashboard
    chạy liên tục mà không dừng.
    """

    def __init__(self, csv_path: str = _DEFAULT_CSV):
        """
        Args:
            csv_path: Đường dẫn file CSV Evo200. Dùng load_evo200_csv() để
                      đảm bảo đúng schema và sign convention (I > 0 khi xả).
        """
        self._csv_path = csv_path
        self._df = None
        self._index = 0

    def connect(self) -> None:
        """Đọc CSV vào bộ nhớ — tương đương mở serial connection."""
        from src.preprocessing.loader import load_evo200_csv
        self._df = load_evo200_csv(self._csv_path)

        start_row = int(os.environ.get("MOCK_CSV_START_ROW", "0"))
        if start_row > 0:
            start_row = min(start_row, len(self._df) - 1)
            self._df = self._df.iloc[start_row:].reset_index(drop=True)
            logger.info(
                "MockCanReader: bỏ qua %d rows đầu (MOCK_CSV_START_ROW=%d)",
                start_row, start_row,
            )

        self._index = 0
        logger.info(
            "MockCanReader: đã nạp %d rows từ %s",
            len(self._df), Path(self._csv_path).name,
        )

    def disconnect(self) -> None:
        """Giải phóng dữ liệu."""
        self._df = None
        logger.info("MockCanReader: ngắt kết nối")

    def read_frames(self) -> list:
        """
        Trả về row CSV tiếp theo dưới dạng decoded dict.

        Loop lại từ đầu khi hết CSV để dashboard chạy liên tục.

        Returns:
            List chứa 1 dict với keys: pack_voltage_v, pack_current_a,
            temp_c, speed_kmh, soc_bms.
            Trả về [] nếu chưa gọi connect().
        """
        if self._df is None:
            return []

        row = self._df.iloc[self._index]
        self._index += 1

        if self._index >= len(self._df):
            self._index = 0
            logger.info("MockCanReader: hết CSV, loop lại từ đầu")

        import math
        decoded = {
            "pack_voltage_v": float(row["pack_voltage_v"]),
            "pack_current_a": float(row["pack_current_a"]),
            "temp_c":         float(row["temp_c"]),
            "speed_kmh":      float(row["speed_kmh"]),
            "soc_bms":        float(row["soc_bms"]),
        }
        odo = float(row["odo_km"])
        if not math.isnan(odo):
            decoded["odo_km"] = odo
        return [decoded]
