"""
Đọc fix GPS từ module VK-162 (chip u-blox 7, USB, NMEA-0183 9600bps 1Hz).

Module này chạy một thread riêng, đọc liên tục các câu NMEA (RMC/GGA)
và lưu snapshot mới nhất (lat/lon/speed/fix/sats) để main loop đọc qua
get_latest(). KHÔNG được làm main loop chậm hoặc crash khi GPS không
cắm/không có fix — đây là tín hiệu phụ trợ, không bắt buộc cho vận hành.

Câu NMEA dùng:
  - RMC: vị trí (lat/lon), tốc độ (knot), trạng thái fix hợp lệ (A/V).
  - GGA: chất lượng fix (0=no fix,1=GPS,2=DGPS) và số vệ tinh.
"""

import logging
import os
import threading
from math import asin, cos, radians, sin, sqrt

import pynmea2
import serial

logger = logging.getLogger(__name__)

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUDRATE = 9600
_KNOT_TO_KMH = 1.852
_EARTH_RADIUS_KM = 6371.0
_MIN_STEP_KM = 0.002  # lọc nhiễu đứng yên — bỏ bước < 2m


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Khoảng cách great-circle giữa 2 điểm (độ thập phân), đơn vị km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * _EARTH_RADIUS_KM


class GpsReader:
    """
    Đọc NMEA từ VK-162 trên thread riêng, không chặn main loop.

    Cách dùng::

        gps_reader = GpsReader()
        gps_reader.start()
        ...
        snapshot = gps_reader.get_latest()  # {'lat', 'lon', 'speed_kmh', 'fix', 'sats', 'distance_km'}
        ...
        gps_reader.stop()

    Nếu không mở được cổng serial (không cắm GPS, sai port), reader
    log warning rồi dừng thread — get_latest() vẫn trả về dict với
    lat/lon=None, fix=0, không raise exception.
    """

    def __init__(self, port: str = None, baud: int = DEFAULT_BAUDRATE):
        """
        Khởi tạo GpsReader.

        Args:
            port: Đường dẫn cổng serial. Nếu None, lấy từ biến môi trường
                  GPS_PORT, mặc định '/dev/ttyACM0'.
            baud: Tốc độ baud, mặc định 9600 (chuẩn VK-162).
        """
        self.port = port or os.environ.get("GPS_PORT", DEFAULT_PORT)
        self.baud = baud

        self._ser = None
        self._thread = None
        self._lock = threading.Lock()
        self._running = False
        self._available = False

        # Quãng đường cộng dồn từ GPS (Haversine), tách khỏi odo CAN
        self._last_lat = None
        self._last_lon = None
        self._total_distance_km = 0.0

        self._latest = {
            "lat": None,
            "lon": None,
            "speed_kmh": 0.0,
            "fix": 0,
            "sats": 0,
            "distance_km": 0.0,
        }

    def start(self) -> None:
        """Tạo daemon thread chạy _read_loop(), trả về ngay không chặn."""
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        """
        Vòng lặp đọc + parse NMEA, chạy trong thread riêng.

        Mở serial thất bại → log warning và return ngay (không raise,
        không crash main process). Mỗi dòng parse lỗi → bỏ qua, đọc tiếp.
        """
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
            self._available = True
            logger.info("GPS kết nối tại %s (%d bps)", self.port, self.baud)
        except serial.SerialException as e:
            logger.warning("GPS not available tại %s, bỏ qua GPS (%s)", self.port, e)
            self._available = False
            return

        while self._running:
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                msg = pynmea2.parse(line)
            except (pynmea2.ParseError, UnicodeError, serial.SerialException):
                continue
            except Exception:
                continue

            self._update_from_message(msg)

    def _update_from_message(self, msg) -> None:
        """Cập nhật self._latest từ một câu NMEA đã parse thành công (RMC hoặc GGA)."""
        sentence_type = msg.sentence_type

        with self._lock:
            if sentence_type == "RMC":
                valid = msg.status == "A"
                lat = float(msg.latitude) if valid else None
                lon = float(msg.longitude) if valid else None
                self._latest["lat"] = lat
                self._latest["lon"] = lon
                spd_knots = msg.spd_over_grnd
                self._latest["speed_kmh"] = (
                    float(spd_knots) * _KNOT_TO_KMH if spd_knots is not None else 0.0
                )

                # Cộng dồn quãng đường (Haversine) — chỉ khi fix hợp lệ
                if valid:
                    if self._last_lat is not None and self._last_lon is not None:
                        step = _haversine_km(self._last_lat, self._last_lon, lat, lon)
                        if step > _MIN_STEP_KM:
                            self._total_distance_km += step
                    self._last_lat = lat
                    self._last_lon = lon
                    self._latest["distance_km"] = round(self._total_distance_km, 3)
            elif sentence_type == "GGA":
                self._latest["fix"] = int(msg.gps_qual) if msg.gps_qual is not None else 0
                self._latest["sats"] = int(msg.num_sats) if msg.num_sats else 0

    def get_latest(self) -> dict:
        """
        Lấy snapshot mới nhất, thread-safe.

        Returns:
            Dict {'lat', 'lon', 'speed_kmh', 'fix', 'sats', 'distance_km'}.
            lat/lon = None nếu chưa có fix hợp lệ hoặc GPS không available.
        """
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        """Dừng thread đọc và đóng cổng serial."""
        self._running = False
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
            logger.info("Đã ngắt kết nối GPS")
