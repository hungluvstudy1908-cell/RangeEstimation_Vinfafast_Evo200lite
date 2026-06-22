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
import time
from math import asin, cos, radians, sin, sqrt

import pynmea2
import serial

logger = logging.getLogger(__name__)

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUDRATE = 9600
GPS_SERIAL_TIMEOUT_S = 1.0  # timeout đọc serial — tách hằng số để log startup thấy giá trị thật
GPS_READ_SLEEP_S = 0.05     # sleep khi readline() rỗng — GPS 1Hz nên dư, tránh poll dồn USB bus

# Mirror cơ chế stale-detect/reopen của CanRxThread (xem src/can_reader/rx_thread.py).
# gps_stale dựa trên TUỔI SENTENCE (serial chết = không có câu NMEA nào) — KHÔNG dựa
# trên mất fix (vào hầm mà sentence vẫn chảy thì không coi là stale, không reconnect).
GPS_STALE_MS = 5000              # 5s không có sentence = serial chết
GPS_STALE_LATCH_MS = 2500        # mirror CAN_STALE_LATCH_MS
GPS_RECONNECT_COOLDOWN_MS = 3000  # mirror CAN_RECONNECT_COOLDOWN_MS

_KNOT_TO_KMH = 1.852
_EARTH_RADIUS_KM = 6371.0
_MIN_STEP_KM = 0.002  # lọc nhiễu đứng yên — bỏ bước < 2m

# GPS_READ_ONLY=1: thread vẫn readline() để giữ port sống/drain bus, NHƯNG bỏ
# qua parse NMEA và bỏ cập nhật shared state/distance hoàn toàn. Mục đích: cô
# lập "đọc serial GPS thuần USB I/O" khỏi "xử lý GPS (Python/GIL/lock)" khi so
# sánh ảnh hưởng lên CAN RX — pass (không stall CAN) → việc xử lý NMEA/lock
# phía Python góp phần gây tranh chấp; fail (vẫn stall) → bản thân I/O serial
# GPS trên USB bus là nguyên nhân chính.
GPS_READ_ONLY = os.environ.get("GPS_READ_ONLY") == "1"


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
    log warning rồi tự thử reopen theo cooldown (xem GPS_RECONNECT_COOLDOWN_MS)
    — thread KHÔNG dừng. get_latest() vẫn trả về dict với lat/lon=None, fix=0,
    không raise exception.
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

        # --- Metric chẩn đoán (chỉ đếm, không đổi hành vi đọc) ---
        self._iter_count = 0          # tăng mỗi vòng _read_loop — phát hiện spin
        self._sentence_count = 0      # tăng mỗi NMEA parse thành công
        self._parse_error_count = 0   # tăng mỗi lỗi parse/đọc
        self._last_sentence_mono = None  # time.monotonic() lần parse thành công gần nhất
        self._last_fix_mono = None       # time.monotonic() lần fix hợp lệ (RMC 'A') gần nhất
        self._lock_hold_ms_max = 0.0     # thời gian giữ self._lock lâu nhất, reset mỗi lần get_metrics()
        self._reconnect_count = 0        # tăng mỗi lần reopen serial THÀNH CÔNG (mirror can_reconnect_count)

        # --- Stale-detect + reopen (mirror CanRxThread, xem GPS_STALE_MS) ---
        self.gps_stale = False
        self.gps_stale_count = 0     # số episode stale (transition False→True)
        self.gps_recover_count = 0   # 1-1 với gps_stale_count, không double-count
        self.gps_serial_errors = 0   # số lần reopen fail
        self._stale_latch_until_mono = 0.0
        self._last_reconnect_mono = 0.0
        self._loop_started_mono = None  # mốc tham chiếu tuổi sentence khi chưa có sentence nào

        self._metrics_last_mono = time.monotonic()
        self._metrics_last_iter = 0
        self._metrics_last_sentence = 0
        self._metrics_last_error = 0

    def start(self) -> None:
        """Tạo daemon thread chạy _read_loop(), trả về ngay không chặn."""
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _open_serial(self) -> bool:
        """
        Mở (hoặc mở lại) cổng serial GPS — dùng chung cho startup và reconnect.

        Đóng self._ser cũ nếu còn mở trước khi mở lại cùng cổng.

        Returns:
            True nếu mở thành công, False nếu serial.SerialException (log warning).
        """
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            except Exception:
                pass

        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=GPS_SERIAL_TIMEOUT_S)
            self._available = True
            logger.info("GPS kết nối tại %s (%d bps)", self.port, self.baud)
            return True
        except serial.SerialException as e:
            # self._ser=None (không giữ object cũ đã close) — để _read_loop nhận biết
            # qua nhánh "if self._ser is None" và sleep, tránh busy-loop gọi readline()
            # trên port đã đóng (PortNotOpenError) cho tới cooldown reopen kế tiếp.
            self._ser = None
            logger.warning("GPS not available tại %s, bỏ qua GPS (%s)", self.port, e)
            self._available = False
            return False

    def _mark_stale(self, reason: str, age_ms: float) -> None:
        """
        Đánh dấu gps_stale=True và giữ latch tối thiểu GPS_STALE_LATCH_MS — chỉ
        log/đếm gps_stale_count 1 lần mỗi episode (transition False→True), mirror
        _mark_stale của CanRxThread.
        """
        if not self.gps_stale:
            self.gps_stale = True
            self.gps_stale_count += 1
            logger.warning("GPS_STALE_DETECTED reason=%s age_ms=%.0f", reason, age_ms)
        self._stale_latch_until_mono = time.monotonic() + GPS_STALE_LATCH_MS / 1000.0

    def _mark_recovered(self) -> None:
        """Tắt gps_stale sau khi latch GPS_STALE_LATCH_MS đã hết, mirror _mark_recovered của CanRxThread."""
        if self.gps_stale and time.monotonic() >= self._stale_latch_until_mono:
            self.gps_stale = False
            self.gps_recover_count += 1
            logger.info("GPS_RECOVERED reconnect_count=%d", self._reconnect_count)

    def _check_stale_and_maybe_reopen(self) -> None:
        """
        Phát hiện stale theo tuổi sentence gần nhất rồi tự reopen serial — mirror
        _check_stale_and_maybe_reopen của CanRxThread. Reopen 1 lần/cooldown,
        KHÔNG block vô hạn — nếu vẫn stale, lần check sau (cooldown kế tiếp) tự thử lại.
        """
        now_mono = time.monotonic()
        reference = self._last_sentence_mono if self._last_sentence_mono is not None else self._loop_started_mono
        if reference is None:
            return

        age_ms = (now_mono - reference) * 1000.0
        if age_ms <= GPS_STALE_MS:
            return

        self._mark_stale("age_timeout", age_ms)

        if (now_mono - self._last_reconnect_mono) * 1000.0 <= GPS_RECONNECT_COOLDOWN_MS:
            return

        attempt = self._reconnect_count + 1
        logger.warning("GPS_RECONNECT_ATTEMPT attempt=%d reason=age_timeout age_ms=%.0f", attempt, age_ms)
        self._last_reconnect_mono = now_mono
        if self._open_serial():
            self._reconnect_count += 1
            logger.info("GPS_RECONNECT_SUCCESS reconnect_count=%d", self._reconnect_count)
        else:
            self.gps_serial_errors += 1

    def _read_loop(self) -> None:
        """
        Vòng lặp đọc + parse NMEA, chạy trong thread riêng.

        Mở serial thất bại lúc start → KHÔNG return, đánh dấu gps_stale=True và
        vào nhịp reconnect-theo-cooldown của _check_stale_and_maybe_reopen() như
        khi serial chết giữa chừng. Thread chỉ dừng khi self._running=False.
        Mỗi dòng parse lỗi → bỏ qua, đọc tiếp.
        """
        self._loop_started_mono = time.monotonic()
        if not self._open_serial():
            self._mark_stale("startup_failed", 0.0)

        while self._running:
            self._iter_count += 1
            self._check_stale_and_maybe_reopen()

            if self._ser is None:
                time.sleep(GPS_READ_SLEEP_S)
                continue

            try:
                raw_line = self._ser.readline()
                if not raw_line:
                    time.sleep(GPS_READ_SLEEP_S)
                    continue
                if GPS_READ_ONLY:
                    # Drain bus, giữ port sống — KHÔNG parse, KHÔNG update state.
                    continue
                line = raw_line.decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                msg = pynmea2.parse(line)
            except (pynmea2.ParseError, UnicodeError, serial.SerialException):
                self._parse_error_count += 1
                continue
            except Exception:
                self._parse_error_count += 1
                continue

            self._sentence_count += 1
            self._last_sentence_mono = time.monotonic()
            self._mark_recovered()
            if msg.sentence_type == "RMC" and getattr(msg, "status", None) == "A":
                self._last_fix_mono = time.monotonic()

            self._update_from_message(msg)

    def _update_from_message(self, msg) -> None:
        """Cập nhật self._latest từ một câu NMEA đã parse thành công (RMC hoặc GGA)."""
        sentence_type = msg.sentence_type

        with self._lock:
            _lock_t0 = time.monotonic()
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

                # Cộng dồn quãng đường (Haversine) — chỉ khi fix hợp lệ.
                # GPS_READ_ONLY=1: bỏ qua nhánh này hoàn toàn, _total_distance_km giữ 0.
                if valid and not GPS_READ_ONLY:
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

            _hold_ms = (time.monotonic() - _lock_t0) * 1000.0
            if _hold_ms > self._lock_hold_ms_max:
                self._lock_hold_ms_max = _hold_ms

    def get_latest(self) -> dict:
        """
        Lấy snapshot mới nhất, thread-safe.

        Returns:
            Dict {'lat', 'lon', 'speed_kmh', 'fix', 'sats', 'distance_km'}.
            lat/lon = None nếu chưa có fix hợp lệ hoặc GPS không available.
        """
        with self._lock:
            return dict(self._latest)

    def get_metrics(self) -> dict:
        """
        Snapshot metric chẩn đoán (đọc qua main loop, KHÔNG dùng self._lock —
        các counter dùng ở đây chỉ là số nguyên/float ghi từ 1 thread nên đọc
        xấp xỉ là đủ cho mục đích chẩn đoán).

        Tính *_per_sec theo dt kể từ lần gọi get_metrics() trước đó, và tự
        reset gps_lock_hold_ms_max sau khi đọc.
        """
        now = time.monotonic()
        dt = now - self._metrics_last_mono
        if dt <= 0:
            dt = 1e-6

        iters = self._iter_count - self._metrics_last_iter
        sentences = self._sentence_count - self._metrics_last_sentence
        errors = self._parse_error_count - self._metrics_last_error

        metrics = {
            "gps_enabled": self._running,
            "gps_read_only": GPS_READ_ONLY,
            "gps_thread_alive": self._thread is not None and self._thread.is_alive(),
            "gps_available": self._available,
            "gps_iters_per_sec": iters / dt,
            "gps_sentences_per_sec": sentences / dt,
            "gps_parse_errors_per_sec": errors / dt,
            "gps_last_sentence_age_ms": (
                (now - self._last_sentence_mono) * 1000.0
                if self._last_sentence_mono is not None else None
            ),
            "gps_last_fix_age_ms": (
                (now - self._last_fix_mono) * 1000.0
                if self._last_fix_mono is not None else None
            ),
            "gps_lock_hold_ms_max": self._lock_hold_ms_max,
            "gps_reconnect_count": self._reconnect_count,
            "gps_stale": self.gps_stale,
            "gps_stale_count": self.gps_stale_count,
            "gps_recover_count": self.gps_recover_count,
            "gps_serial_errors": self.gps_serial_errors,
        }

        self._metrics_last_mono = now
        self._metrics_last_iter = self._iter_count
        self._metrics_last_sentence = self._sentence_count
        self._metrics_last_error = self._parse_error_count
        self._lock_hold_ms_max = 0.0

        return metrics

    def stop(self) -> None:
        """Dừng thread đọc và đóng cổng serial."""
        self._running = False
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
            logger.info("Đã ngắt kết nối GPS")
