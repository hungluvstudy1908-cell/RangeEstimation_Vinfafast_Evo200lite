"""
Ghi raw CAN frame ra CSV — lossless, chạy trong thread riêng.

Đọc từ raw_queue (queue.Queue không maxlen, do CanRxThread fan-out vào) và
ghi liên tục ra file `data/raw_can/raw_can_<timestamp>.csv`. Mục tiêu: giữ
ĐỦ mọi khung CAN nhận được trong phiên thu thập dài (4+ giờ), không phụ
thuộc tốc độ dashboard/model — đường raw này tách biệt hoàn toàn khỏi
proc_deque (đường lossy cho main loop).

Không fsync mỗi khung (quá chậm) — chỉ flush định kỳ và fsync thưa hơn.
"""

import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

WRITER_FLUSH_INTERVAL_S = 1.0
WRITER_FSYNC_INTERVAL_S = 20.0
_QUEUE_GET_TIMEOUT_S = 0.5
WRITER_IDLE_WARN_MS = 2000.0       # không record mới liên tục >2s khi thread vẫn alive
WRITER_IDLE_WARN_THROTTLE_S = 2.0  # throttle log WARNING khi idle kéo dài

DEFAULT_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw_can"


class RawCanWriter:
    """
    Ghi raw_queue ra CSV trong thread riêng, lossless.

    Cách dùng::

        raw_writer = RawCanWriter(raw_queue)
        raw_writer.start()
        ...
        raw_writer.stop()   # drain hết queue còn lại trước khi đóng file
    """

    HEADER = "t_wall,t_mono_ns,can_id,dlc,data_hex\n"

    def __init__(self, raw_queue: "queue.Queue", out_dir=DEFAULT_RAW_DIR):
        self._raw_queue = raw_queue
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self._out_dir / f"raw_can_{timestamp_str}.csv"
        self._file = open(self.filepath, "w", encoding="utf-8")
        self._file.write(self.HEADER)

        self._stop_event = threading.Event()
        self._thread = None

        # Metrics
        self.written_count = 0
        self.last_flush_mono = None
        self.last_write_mono = None      # cập nhật mỗi lần ghi 1 dòng ra file
        self._last_record_mono = time.monotonic()  # tuổi "no input" — phân biệt writer-chết vs không-input
        self._last_idle_warn_mono = 0.0

        logger.info("RawCanWriter: ghi raw CAN ra %s", self.filepath)

    @property
    def alive(self) -> bool:
        """True nếu thread writer đang chạy."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_raw_write_age_ms(self):
        """Tuổi (ms) của lần ghi dòng CSV gần nhất, None nếu chưa ghi dòng nào."""
        if self.last_write_mono is None:
            return None
        return (time.monotonic() - self.last_write_mono) * 1000.0

    @property
    def raw_writes_total(self):
        """Alias của written_count — tên rõ nghĩa hơn cho health log, KHÔNG phải counter riêng."""
        return self.written_count

    def start(self) -> None:
        """Start daemon thread chạy _run()."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Vòng lặp lấy record từ raw_queue, ghi CSV, flush/fsync định kỳ."""
        last_flush = time.monotonic()
        last_fsync = time.monotonic()

        while True:
            try:
                record = self._raw_queue.get(timeout=_QUEUE_GET_TIMEOUT_S)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                self._warn_if_idle()
                continue

            self._last_record_mono = time.monotonic()
            self._write_record(record)

            now = time.monotonic()
            if now - last_flush >= WRITER_FLUSH_INTERVAL_S:
                self._file.flush()
                last_flush = now
                self.last_flush_mono = now
            if now - last_fsync >= WRITER_FSYNC_INTERVAL_S:
                os.fsync(self._file.fileno())
                last_fsync = now

        # Drain nốt phần còn lại của raw_queue trước khi đóng (lossless)
        drained = 0
        while True:
            try:
                record = self._raw_queue.get_nowait()
            except queue.Empty:
                break
            self._write_record(record)
            drained += 1

        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        logger.info("RawCanWriter: đã drain %d khung còn lại khi shutdown, tổng written=%d", drained, self.written_count)

    def _write_record(self, record) -> None:
        """Ghi 1 dòng CSV từ record (t_wall, t_mono, can_id, dlc, data)."""
        t_wall, t_mono, can_id, dlc, data = record
        self._file.write(f"{t_wall:.6f},{t_mono},{can_id},{dlc},{data.hex()}\n")
        self.written_count += 1
        self.last_write_mono = time.monotonic()

    def _warn_if_idle(self) -> None:
        """Log WARNING throttled khi queue rỗng liên tục >WRITER_IDLE_WARN_MS — phân biệt writer-chết vs không-input."""
        now = time.monotonic()
        idle_ms = (now - self._last_record_mono) * 1000.0
        if idle_ms > WRITER_IDLE_WARN_MS and now - self._last_idle_warn_mono >= WRITER_IDLE_WARN_THROTTLE_S:
            logger.warning("RawCanWriter: raw queue idle %.0fms (writer alive, no input)", idle_ms)
            self._last_idle_warn_mono = now

    def stop(self) -> None:
        """Báo dừng — thread sẽ tự drain nốt queue + flush + fsync + close trước khi thoát."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
