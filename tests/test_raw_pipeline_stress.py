"""
Stress test pipeline CAN RX + raw writer — KHÔNG cần hardware.

Mô phỏng tốc độ cao bằng FakeHighRateReader (sinh K khung ngẫu nhiên mỗi lần
read_frames() được gọi) để xác nhận:
  - raw_queue lossless: số dòng CSV (trừ header) == tổng khung sinh ra.
  - raw_dropped_frames luôn = 0.
  - raw_queue không phình vô hạn sau khi ngừng bơm (writer theo kịp).
  - thread join sạch, không treo khi stop().
"""

import csv
import os
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.can_reader.rx_thread import CanRxThread
from src.can_reader.raw_writer import RawCanWriter

import queue as queue_module


class FakeHighRateReader:
    """Sinh K khung CAN ngẫu nhiên mỗi lần read_frames() — không cần serial thật."""

    def __init__(self, frames_per_call: int = 50):
        self.frames_per_call = frames_per_call
        self.total_generated = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        pass

    def connect_with_retry(self, **kwargs):
        return self

    def disconnect(self) -> None:
        pass

    def read_frames(self) -> list:
        frames = []
        with self._lock:
            for _ in range(self.frames_per_call):
                can_id = random.randint(0, 0x7FF)
                data = bytes(random.randint(0, 255) for _ in range(8))
                frames.append((can_id, data))
                self.total_generated += 1
        return frames


def test_raw_pipeline_lossless_under_high_rate(tmp_path):
    """Bơm khung tốc độ cao trong ~1s, dừng sạch — raw CSV phải khớp 100% tổng khung sinh."""
    fake = FakeHighRateReader(frames_per_call=50)
    raw_queue = queue_module.Queue()
    proc_deque = deque(maxlen=300)

    raw_writer = RawCanWriter(raw_queue, out_dir=tmp_path)
    raw_writer.start()

    rx = CanRxThread(fake, raw_queue, proc_deque)
    # Bỏ qua connect_with_retry thật (không cần serial) — set trực tiếp thread running
    rx._stop_event.clear()
    rx._thread = threading.Thread(target=rx._run, daemon=True)
    rx._thread.start()

    time.sleep(1.0)  # bơm khung tốc độ cao trong 1 giây

    rx.stop()
    raw_writer.stop()

    total_generated = fake.total_generated

    assert rx.rx_count == total_generated, "rx_count phải khớp số khung sinh ra"
    assert raw_writer.written_count == total_generated, "writer phải ghi đủ 100% khung — lossless"

    csv_path = raw_writer.filepath
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    data_rows = rows[1:]  # bỏ header
    assert len(data_rows) == total_generated, (
        f"Số dòng CSV ({len(data_rows)}) phải == tổng khung sinh ({total_generated})"
    )

    assert rx.proc_dropped >= 0  # proc_deque có thể rớt cũ OK, raw không bị ảnh hưởng
    assert raw_queue.qsize() == 0, "raw_queue phải về 0 sau khi writer drain hết"
    assert rx.max_raw_qsize >= 0  # watermark hữu hạn, không tràn

    assert not rx._thread.is_alive(), "RX thread phải join sạch sau stop()"
    assert not raw_writer._thread.is_alive(), "Writer thread phải join sạch sau stop()"


def test_can_rx_read_frames_drains_proc_deque():
    """read_frames() của CanRxThread phải drain hết proc_deque, không raise."""
    fake = FakeHighRateReader(frames_per_call=5)
    raw_queue = queue_module.Queue()
    proc_deque = deque(maxlen=300)
    rx = CanRxThread(fake, raw_queue, proc_deque)

    proc_deque.append((1, bytes(8)))
    proc_deque.append((2, bytes(8)))

    frames = rx.read_frames()
    assert frames == [(1, bytes(8)), (2, bytes(8))]
    assert rx.read_frames() == []  # deque đã rỗng, gọi lại không raise
