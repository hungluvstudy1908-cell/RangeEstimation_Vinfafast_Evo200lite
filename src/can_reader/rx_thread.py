"""
Thread RX riêng cho CAN — drain serial liên tục, tách thành 2 đường fan-out.

Gốc rễ vấn đề (xem agent_docs/debugging_notes.md nếu có cập nhật): trước đây
`WaveshareReader.read_frames()` chỉ được gọi 1 lần/tick (100ms) trong main
loop. Giữa 2 lần gọi, byte mới tới chứa trong OS serial buffer (~4KB) — nếu
Waveshare bơm nhanh hơn tốc độ tick đọc, buffer tràn và mất khung âm thầm.

CanRxThread sở hữu serial — gọi `inner.read_frames()` liên tục trong thread
riêng, không phụ thuộc nhịp main loop. Mỗi khung đọc được fan-out ra 2 đường
với chính sách rớt khác nhau:
  - raw_queue: LOSSLESS (queue.Queue không maxlen) — ghi disk đầy đủ.
  - proc_deque: lossy OK (deque có maxlen) — main loop xử lý, rớt cũ khi bận
    không phải lỗi vì raw đã giữ đủ.

`read_frames()` của class này (cùng tên với WaveshareReader) chỉ drain
proc_deque — main loop gọi y như cũ, không cần biết có thread phía sau.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

RAW_QUEUE_SOFT_LIMIT = 2000   # qsize vượt → log WARNING (throttled ~1Hz)
RAW_QUEUE_HARD_LIMIT = 20000  # qsize vượt → log CRITICAL, KHÔNG crash
_RECONNECT_RETRY_DELAY_S = 1.0
_IDLE_SLEEP_S = 0.001  # tránh busy-wait 100% CPU khi không có frame mới

# Waveshare có thể "latch im lặng" (read_frames() trả rỗng, in_waiting=0,
# KHÔNG raise) khi USB bus bị giành bởi thiết bị khác (vd GPS serial) — đường
# except cũ không bắt được trường hợp này. CAN_STALE_MS/COOLDOWN: phát hiện
# CHỦ ĐỘNG theo tuổi frame gần nhất rồi tự reopen serial.
CAN_STALE_MS = 2000
CAN_RECONNECT_COOLDOWN_MS = 3000


class CanRxThread:
    """
    Thread đọc CAN liên tục từ WaveshareReader, fan-out ra raw_queue + proc_deque.

    Cách dùng trong main.py::

        can_reader = CanRxThread(inner, raw_queue, proc_deque)
        can_reader.connect()         # connect inner + start thread
        ...
        frames = can_reader.read_frames()   # drain proc_deque, gọi y như cũ
        ...
        can_reader.stop()
    """

    def __init__(self, inner, raw_queue, proc_deque):
        """
        Args:
            inner: WaveshareReader instance — CHỈ được đọc serial trong thread này.
            raw_queue: queue.Queue() không maxlen — lossless, cho RawCanWriter.
            proc_deque: collections.deque(maxlen=...) — lossy OK, cho main loop.
        """
        self._inner = inner
        self._raw_queue = raw_queue
        self._proc_deque = proc_deque

        self._stop_event = threading.Event()
        self._thread = None

        # Metrics
        self.rx_count = 0
        self.rx_errors = 0
        self.proc_dropped = 0
        self.last_frame_t_mono = None
        self.max_raw_qsize = 0

        self._last_soft_warn_mono = 0.0
        self._last_hard_warn_mono = 0.0

        # Metrics — proactive stale-detect + reopen (xem CAN_STALE_MS)
        self.can_reconnect_count = 0
        self.can_serial_errors = 0
        self.can_stale = False
        self._last_reconnect_mono = None

    @property
    def alive(self) -> bool:
        """True nếu thread RX đang chạy."""
        return self._thread is not None and self._thread.is_alive()

    def connect(self) -> None:
        """Connect inner reader (ưu tiên connect_with_retry) + start thread RX nếu chưa chạy."""
        from src.can_reader.reader import connect_with_retry

        if self._inner._ser is None or not self._inner._ser.is_open:
            retried = connect_with_retry(port=self._inner.port, baudrate=self._inner.baudrate)
            self._inner._ser = retried._ser
            self._inner._buffer = retried._buffer

        if not self.alive:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.info("CanRxThread: đã start thread RX")

    def _run(self) -> None:
        """Vòng lặp drain serial liên tục — chạy trong thread riêng."""
        while not self._stop_event.is_set():
            try:
                frames = self._inner.read_frames()
                t_mono = time.monotonic_ns()
                t_wall = time.time()

                for can_id, data in frames:
                    self._raw_queue.put((t_wall, t_mono, can_id, len(data), data))

                    if len(self._proc_deque) == self._proc_deque.maxlen:
                        self.proc_dropped += 1
                    self._proc_deque.append((can_id, data))

                    self.rx_count += 1

                if frames:
                    self.last_frame_t_mono = t_mono

                self._check_queue_watermark()
                self._check_stale_and_maybe_reopen()

                if not frames:
                    time.sleep(_IDLE_SLEEP_S)

            except Exception as e:
                self.rx_errors += 1
                logger.warning("CanRxThread: lỗi đọc serial (%s), thử reconnect...", e)
                self._reconnect_loop()

    def _check_stale_and_maybe_reopen(self) -> None:
        """
        Phát hiện stale CHỦ ĐỘNG theo tuổi frame gần nhất — bù cho việc latch
        im lặng của Waveshare không raise exception nên except ở _run() không
        bắt được. Reopen 1 lần/cooldown, KHÔNG block vô hạn (Waveshare có thể
        thật sự đang rút cáp) — nếu vẫn stale, lần check sau (cooldown kế tiếp)
        sẽ tự thử lại.
        """
        if self.last_frame_t_mono is None:
            return

        now = time.monotonic_ns()
        age_ms = (now - self.last_frame_t_mono) / 1e6
        self.can_stale = age_ms > CAN_STALE_MS
        if not self.can_stale:
            return

        cooled = (
            self._last_reconnect_mono is None
            or (now - self._last_reconnect_mono) / 1e6 > CAN_RECONNECT_COOLDOWN_MS
        )
        if not cooled:
            return

        try:
            logger.warning(
                "CanRxThread: CAN stale %.0fms → reopen serial (reconnect #%d)",
                age_ms, self.can_reconnect_count + 1,
            )
            self._inner.disconnect()
            self._inner.connect()
            self.can_reconnect_count += 1
            self._last_reconnect_mono = time.monotonic_ns()
            self.last_frame_t_mono = self._last_reconnect_mono  # reset đồng hồ stale
            self.can_stale = False
        except Exception as e:
            self.can_serial_errors += 1
            logger.error("CanRxThread: reopen failed: %s", e)
            time.sleep(0.5)

    def _check_queue_watermark(self) -> None:
        """Cảnh báo nếu raw_queue phình to bất thường — throttle log ~1Hz."""
        q = self._raw_queue.qsize()
        if q > self.max_raw_qsize:
            self.max_raw_qsize = q

        now = time.monotonic()
        if q > RAW_QUEUE_HARD_LIMIT and now - self._last_hard_warn_mono >= 1.0:
            logger.critical("CanRxThread: raw_queue qsize=%d VƯỢT HARD_LIMIT=%d — writer không theo kịp!", q, RAW_QUEUE_HARD_LIMIT)
            self._last_hard_warn_mono = now
        elif q > RAW_QUEUE_SOFT_LIMIT and now - self._last_soft_warn_mono >= 1.0:
            logger.warning("CanRxThread: raw_queue qsize=%d vượt SOFT_LIMIT=%d", q, RAW_QUEUE_SOFT_LIMIT)
            self._last_soft_warn_mono = now

    def _reconnect_loop(self) -> None:
        """Thử disconnect + connect_with_retry tới khi thành công hoặc thread bị stop."""
        from src.can_reader.reader import connect_with_retry

        try:
            self._inner.disconnect()
        except Exception:
            pass

        while not self._stop_event.is_set():
            try:
                retried = connect_with_retry(port=self._inner.port, baudrate=self._inner.baudrate, retry_delay_s=_RECONNECT_RETRY_DELAY_S)
                self._inner._ser = retried._ser
                self._inner._buffer = retried._buffer
                logger.info("CanRxThread: reconnect thành công")
                return
            except Exception as e:
                logger.error("CanRxThread: reconnect thất bại (%s), thử lại sau %.1fs", e, _RECONNECT_RETRY_DELAY_S)
                time.sleep(_RECONNECT_RETRY_DELAY_S)

    def read_frames(self) -> list:
        """
        Drain proc_deque — gọi từ main loop, KHÔNG raise (an toàn ngay cả khi thread RX lỗi).

        Returns:
            list[tuple[int, bytes]] — các khung (can_id, data) đang chờ xử lý.
        """
        frames = []
        try:
            while True:
                frames.append(self._proc_deque.popleft())
        except IndexError:
            pass
        return frames

    def last_frame_age_ms(self):
        """Tuổi (ms) của khung gần nhất nhận được, None nếu chưa có khung nào."""
        if self.last_frame_t_mono is None:
            return None
        return (time.monotonic_ns() - self.last_frame_t_mono) / 1e6

    @property
    def can_last_frame_age_ms(self):
        """Alias dạng property của last_frame_age_ms() — dùng cho health log + SharedState."""
        return self.last_frame_age_ms()

    def stop(self) -> None:
        """Dừng thread RX (không đóng serial — gọi disconnect() để đóng luôn)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def disconnect(self) -> None:
        """Dừng thread RX + đóng serial của inner reader."""
        self.stop()
        self._inner.disconnect()
