"""
Entry point — VinFast Evo 200 SoC/SoH runtime system.

Single-thread 10Hz main loop:
  - Tick 10Hz: CAN read + decode + update SoC #1 (BMS) + SoC #2 (Coulomb Counter)
  - Tick 1Hz (mỗi 10 ticks): resample + SoC #3 (CNN1D) + range + log

Web server chạy thread riêng, chỉ ĐỌC state qua lock.

Architecture: service_architecture.md §4
"""

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.can_reader.reader import WaveshareReader
from src.can_reader.decoder import decode as decode_can
from src.coulomb_counter.counter import CoulombCounter
from src.gps.reader import GpsReader
from src.logger.writer import RuntimeLogger
from src.preprocessing.normalize import apply_minmax
from src.range_estimator.estimator import RangeEstimator
from src.soc_inference.inference import SocInference
from src.soh_estimator.estimator import SohEstimator

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

TICK_HZ = 10  # Main loop frequency (Hz)
TICK_DT = 1.0 / TICK_HZ  # Time per tick (seconds)
INFERENCE_EVERY_N_TICK = 10  # Inference every 1 second
RING_BUFFER_SIZE = 60  # Keep 6 seconds of data at 10Hz (needed by range_estimator)

CAN_PORT = "/dev/ttyUSB0"  # Waveshare USB-CAN port
CAN_BAUDRATE = 2_000_000  # 2 Mbps

WEB_SERVER_HOST = "0.0.0.0"
WEB_SERVER_PORT = 5000


# ============================================================================
# SharedState — Dữ liệu chia sẻ giữa main loop và web server
# ============================================================================


@dataclass
class SharedState:
    """
    Trạng thái hệ thống chia sẻ giữa main loop (ghi) và web server (đọc).

    Được protect bởi threading.Lock — không bao giờ truy cập trực tiếp,
    phải lock trước.
    """

    # Ring buffers — lưu 10 sample gần nhất (1 giây ở 10Hz)
    voltage_buffer: list = field(default_factory=list)
    current_buffer: list = field(default_factory=list)
    speed_buffer: list = field(default_factory=list)
    temp_buffer: list = field(default_factory=list)

    # Latest signals (updated mỗi 100ms tick)
    timestamp: Optional[float] = None
    pack_voltage_v: float = 72.0
    pack_current_a: float = 0.0
    temp_c: float = 25.0
    speed_kmh: float = 0.0
    odo_km: float = 0.0

    # Cờ trạng thái xe — cập nhật từ CAN 0x102 (kickstand/ready/park/eco/sport) và 0x201 (brake)
    is_kickstand: bool = False
    is_ready: bool = False
    is_park: bool = False
    is_brake: bool = False
    is_eco: bool = False
    is_sport: bool = False

    # State of Charge — 3 sources
    soc_bms: float = 50.0  # SoC #1: BMS (passthrough)
    soc_cc: float = 50.0  # SoC #2: Coulomb Counter
    soc_model: float = 50.0  # SoC #3: CNN1D model (updated 1Hz)

    # State of Health (updated 1Hz)
    soh: float = 95.8

    # Range estimation (updated 1Hz) — range_km giữ alias = range_model (tương thích cũ)
    range_km: float = 0.0
    range_bms: float = 0.0
    range_cc: float = 0.0
    range_model: float = 0.0
    wh_per_km: float = 50.0

    # GPS (updated 1Hz, từ VK-162 — None nếu chưa có fix hoặc GPS không cắm)
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_speed_kmh: float = 0.0
    gps_fix: int = 0
    gps_sats: int = 0
    gps_distance_km: float = 0.0
    route_total_km: float = 40.0
    route_error_km: float = 40.0


    # Sai số tức thời & MAE trượt so với BMS (tính 1Hz)
    err_model: float = 0.0   # soc_model - soc_bms (có dấu)
    err_cc: float = 0.0      # soc_cc - soc_bms (có dấu)
    mae_model: float = 0.0   # MAE trượt 60s của |err_model|
    mae_cc: float = 0.0      # MAE trượt 60s của |err_cc|
    mae_warm: bool = False    # đã đủ warm-up (voltage_buffer đầy 60 sample) chưa

    # BMS cell voltages (22 cells, updated mỗi khi CAN 0x311-0x31B đến)
    cell_data: list = field(default_factory=lambda: [0.0] * 22)

    # Công suất tức thời (W) = total_cell_voltage × |current|
    pack_power_w: float = 0.0

    # Metadata
    tick: int = 0
    fps_actual: float = 0.0  # Actual loop frequency

    def append_buffer(self, max_size: int = RING_BUFFER_SIZE) -> None:
        """
        Chụp snapshot state hiện tại vào ring buffer. Gọi MỘT LẦN MỖI TICK
        sau khi drain hết frames — không gọi mỗi frame riêng lẻ.

        Đọc từ state fields (đã được cập nhật qua toàn bộ frames của tick)
        để tránh nhồi 0.0 vào buffer khi frame CAN chỉ chứa một tín hiệu.
        """
        self.voltage_buffer.append(self.pack_voltage_v)
        self.current_buffer.append(self.pack_current_a)
        self.speed_buffer.append(self.speed_kmh)
        self.temp_buffer.append(self.temp_c)

        # Giữ tối đa max_size sample
        if len(self.voltage_buffer) > max_size:
            self.voltage_buffer.pop(0)
            self.current_buffer.pop(0)
            self.speed_buffer.pop(0)
            self.temp_buffer.pop(0)

    def get_buffers_as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Lấy ring buffers dưới dạng numpy arrays.

        Returns:
            (voltage_array, current_array, speed_array, temp_array)
        """
        return (
            np.array(self.voltage_buffer, dtype=np.float32),
            np.array(self.current_buffer, dtype=np.float32),
            np.array(self.speed_buffer, dtype=np.float32),
            np.array(self.temp_buffer, dtype=np.float32),
        )

    def get_dict(self) -> dict:
        """Lấy toàn bộ state dưới dạng dict (để serialize/log)."""
        return {
            "timestamp": self.timestamp,
            "pack_voltage_v": self.pack_voltage_v,
            "pack_current_a": self.pack_current_a,
            "temp_c": self.temp_c,
            "speed_kmh": self.speed_kmh,
            "odo_km": self.odo_km,
            "soc_bms": self.soc_bms,
            "soc_cc": self.soc_cc,
            "soc_model": self.soc_model,
            "soh": self.soh,
            "range_km": self.range_km,
            "range_bms": self.range_bms,
            "range_cc": self.range_cc,
            "range_model": self.range_model,
            "wh_per_km": self.wh_per_km,
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "gps_speed_kmh": self.gps_speed_kmh,
            "gps_fix": self.gps_fix,
            "gps_sats": self.gps_sats,
            "gps_distance_km": self.gps_distance_km,
            "pack_power_w": self.pack_power_w,
            "cell_data": list(self.cell_data),
            "is_kickstand": self.is_kickstand,
            "is_ready":     self.is_ready,
            "is_park":      self.is_park,
            "is_brake":     self.is_brake,
            "is_eco":       self.is_eco,
            "is_sport":     self.is_sport,
            "err_model":    self.err_model,
            "err_cc":       self.err_cc,
            "mae_model":    self.mae_model,
            "mae_cc":       self.mae_cc,
            "tick": self.tick,
            "fps_actual": self.fps_actual,
        }


# ============================================================================
# System Initialization
# ============================================================================


def init_system() -> Tuple[WaveshareReader, CoulombCounter, SocInference, SohEstimator, Dict[str, RangeEstimator], RuntimeLogger, GpsReader]:
    """
    Khởi tạo tất cả modules.

    Returns:
        Tuple (can_reader, coulomb_counter, soc_inference, soh_estimator, range_estimators, logger, gps_reader)
        range_estimators là dict {"bms", "cc", "model"} — mỗi nguồn SoC một
        instance riêng vì EWMA/warmup/freeze là trạng thái nội bộ per-instance.

    Raises:
        Exception: Nếu khởi tạo bất kỳ module nào thất bại.
    """
    logger.info("Initializing system...")

    # CAN reader — dùng MockCanReader khi MOCK_CAN=1 (test không cần xe)
    if os.environ.get("MOCK_CAN") == "1":
        from src.can_reader.mock_reader import MockCanReader
        _csv = os.environ.get("MOCK_CSV") or str(
            Path(__file__).parent.parent / "data" / "raw" / "Evo200_Mixed1.csv"
        )
        can_reader = MockCanReader(csv_path=_csv)
        logger.info(f"MockCanReader: replay từ {_csv}")
    else:
        logger.info(f"Connecting to Waveshare at {CAN_PORT}...")
        can_reader = WaveshareReader(port=CAN_PORT, baudrate=CAN_BAUDRATE)
    can_reader.connect()
    logger.info("✓ CAN reader connected")

    # Coulomb Counter
    coulomb_counter = CoulombCounter(initial_soc=50.0)
    logger.info("✓ Coulomb Counter initialized")

    # SoC Inference (CNN1D model) — đọc label_scale từ config (100.0 nếu label /100 khi train)
    _cfg_path = Path(__file__).parent.parent / "configs" / "model.yaml"
    with open(_cfg_path, encoding="utf-8") as _f:
        _cfg_init = yaml.safe_load(_f)
    _label_scale = float(_cfg_init.get("label_scale", 1.0))
    soc_inference = SocInference(model_path="models/soc_cnn1d.tflite", label_scale=_label_scale)
    logger.info(f"✓ SoC Inference (CNN1D) initialized  label_scale={_label_scale}")

    # SoH Estimator — kinh nghiệm theo odo, tham số từ model.yaml mục soh:
    soh_estimator = SohEstimator.from_config()
    logger.info("✓ SoH Estimator initialized (odo-based)")

    # Range Estimators — 3 instance riêng cho BMS/CC/Model (EWMA/warmup/freeze per-instance)
    range_estimators = {"bms": RangeEstimator(), "cc": RangeEstimator(), "model": RangeEstimator()}
    logger.info("✓ Range Estimators initialized (bms/cc/model)")

    # Runtime CSV Logger
    runtime_logger = RuntimeLogger()
    logger.info("✓ CSV Logger initialized")

    # GPS Reader (VK-162) — không chặn nếu GPS không cắm, xem reader.py
    gps_reader = GpsReader()
    gps_reader.start()
    logger.info("✓ GPS Reader started")

    logger.info("System initialization complete ✓")
    return can_reader, coulomb_counter, soc_inference, soh_estimator, range_estimators, runtime_logger, gps_reader


# ============================================================================
# Main Loop Skeleton (to be filled in next commits)
# ============================================================================


def main_loop(
    can_reader: WaveshareReader,
    coulomb_counter: CoulombCounter,
    soc_inference: SocInference,
    soh_estimator: SohEstimator,
    range_estimators: Dict[str, RangeEstimator],
    runtime_logger: RuntimeLogger,
    gps_reader: GpsReader,
    state: SharedState,
    lock: threading.Lock,
    socketio=None,
) -> None:
    """
    Single-thread main loop chạy ở ~10Hz.

    Cấu trúc:
      - Mỗi tick 100ms: CAN read + decode + update SoC#1, SoC#2 + emit SocketIO
      - Mỗi 10 tick (1 giây): resample + SoC#3 (CNN1D) + SoH (odo) + 3 range + GPS + log

    Args:
        can_reader: WaveshareReader instance.
        coulomb_counter: CoulombCounter instance.
        soc_inference: SocInference instance.
        soh_estimator: SohEstimator instance.
        range_estimators: Dict {"bms", "cc", "model"} → RangeEstimator instance riêng.
        runtime_logger: RuntimeLogger instance (CSV writer).
        gps_reader: GpsReader instance (đọc nền, không chặn nếu không có GPS).
        state: SharedState (protected by lock).
        lock: threading.Lock để sync access với web server.
        socketio: SocketIO instance để push events, None nếu không dùng.
    """
    logger.info("Starting main loop (10Hz)...")

    # Đọc feature_norm một lần — min/max thật từ training data để normalize runtime
    _cfg_path = Path(__file__).parent.parent / "configs" / "model.yaml"
    with open(_cfg_path, encoding="utf-8") as _f:
        _cfg_loop = yaml.safe_load(_f)
    feature_norm = _cfg_loop["normalization"]

    tick = 0
    tick_times = deque(maxlen=TICK_HZ)  # Track FPS using fixed-size deque
    cc_anchored = False  # Anchor CC từ BMS frame đầu tiên, không dùng initial_soc cứng
    mae_model_window = deque(maxlen=60)  # window MAE trượt 60s @1Hz
    mae_cc_window    = deque(maxlen=60)

    while True:
        t0 = time.monotonic()

        # ~~100ms TICK: CAN read + decode + update SoC#1 (BMS) + SoC#2 (CC)~~
        try:
            frames = can_reader.read_frames()

            for frame in frames:
                # Mock mode trả về decoded dict trực tiếp; hardware trả về (can_id, bytes)
                if isinstance(frame, dict):
                    decoded = frame
                else:
                    can_id, data = frame
                    decoded = decode_can(can_id, data)
                    if decoded is None:
                        continue

                with lock:
                    # Timestamp
                    state.timestamp = time.time()

                    # Update latest signals
                    state.pack_voltage_v = decoded.get("pack_voltage_v", state.pack_voltage_v)
                    state.pack_current_a = decoded.get("pack_current_a", state.pack_current_a)
                    state.temp_c = decoded.get("temp_c", state.temp_c)
                    state.speed_kmh = decoded.get("speed_kmh", state.speed_kmh)
                    state.odo_km    = decoded.get("odo_km",    state.odo_km)

                    # Cờ trạng thái xe — giữ giá trị cũ nếu frame này không chứa cờ
                    state.is_kickstand = decoded.get("is_kickstand", state.is_kickstand)
                    state.is_ready     = decoded.get("is_ready",     state.is_ready)
                    state.is_park      = decoded.get("is_park",      state.is_park)
                    state.is_brake     = decoded.get("is_brake",     state.is_brake)
                    state.is_eco       = decoded.get("is_eco",       state.is_eco)
                    state.is_sport     = decoded.get("is_sport",     state.is_sport)

                    # SoC #1: BMS (passthrough từ CAN)
                    state.soc_bms = decoded.get("soc_bms", state.soc_bms)

                    # Anchor CC từ BMS frame đầu tiên — tránh dùng initial_soc=50 cứng
                    if not cc_anchored and "soc_bms" in decoded:
                        coulomb_counter.reset(decoded["soc_bms"])
                        cc_anchored = True
                        logger.info(
                            "CC anchored to BMS at startup: %.1f%%", decoded["soc_bms"]
                        )

                    # 22 cell voltages — key dạng 'cell_01_v' ... 'cell_22_v'
                    for i in range(1, 23):
                        key = f"cell_{i:02d}_v"
                        if key in decoded:
                            state.cell_data[i - 1] = decoded[key]

                    # Công suất + cập nhật pack_voltage_v từ tổng cell
                    # Guard >0: mock không có cell data → giữ pack_voltage_v từ CSV
                    _tcv = sum(state.cell_data)
                    if _tcv > 0:
                        state.pack_voltage_v = _tcv          # nguồn voltage cho CNN kênh voltage
                        state.pack_power_w = _tcv * abs(state.pack_current_a)

                # Emit SocketIO events ngoài lock (tránh giữ lock khi network)
                if socketio is not None:
                    socketio.emit("update_dash", {
                        "speed": round(state.speed_kmh, 1),
                        "soc": int(state.soc_bms),
                        "soc_cc": round(state.soc_cc, 1),
                        "soc_model": round(state.soc_model, 1),
                        "temp": int(state.temp_c),
                        "odo": round(state.odo_km, 1),
                        "power": round(state.pack_power_w, 1),
                        "range_km": round(state.range_km, 1),
                        "range_bms": round(state.range_bms, 1),
                        "range_cc": round(state.range_cc, 1),
                        "range_model": round(state.range_model, 1),
                        "soh": round(state.soh, 1),
                        "is_kickstand": state.is_kickstand,
                        "is_park":      state.is_park,
                        "is_ready":     state.is_ready,
                        "is_brake":     state.is_brake,
                        "is_eco":       state.is_eco,
                        "is_sport":     state.is_sport,
                    })
                    if any(v > 0 for v in state.cell_data):
                        socketio.emit("update_bms", {
                            "voltage": round(sum(state.cell_data), 1),
                            "current": round(state.pack_current_a, 2),
                            "power": round(state.pack_power_w, 1),
                            "cells": [round(c, 4) for c in state.cell_data],
                        })

            # Một lần mỗi tick — sau khi drain hết frames:
            # chụp snapshot state vào buffer + cập nhật SoC #2 (CC)
            with lock:
                state.append_buffer()
                state.soc_cc = coulomb_counter.update(
                    current_a=state.pack_current_a, dt=TICK_DT
                )
                if cc_anchored and coulomb_counter.should_reset(state.soc_bms):
                    coulomb_counter.reset(state.soc_bms)
                    state.soc_cc = coulomb_counter.soc_cc
                    logger.info(f"Coulomb Counter reset to {state.soc_bms:.1f}%")

        except Exception as e:
            logger.warning(f"CAN read error: {e}, attempting reconnect...")
            try:
                can_reader.disconnect()
                can_reader.connect()
                logger.info("CAN reconnected successfully")
            except Exception as e2:
                logger.error(f"CAN reconnect failed: {e2}")

        # ~~1Hz TICK (mỗi 10 ticks): resample + SoC#3 + range + log~~
        if tick % INFERENCE_EVERY_N_TICK == 0:
            try:
                # Read state buffers (locked, fast)
                with lock:
                    v_arr, i_arr, spd_arr, tmp_arr = state.get_buffers_as_arrays()
                    soc_bms_now = state.soc_bms   # snapshot cho warm-up fallback + range BMS
                    soc_cc_snap = state.soc_cc    # snapshot cho range CC
                    odo_km_now  = state.odo_km    # snapshot cho SohEstimator

                    if len(v_arr) < 2:
                        # Not enough samples yet
                        tick += 1
                        continue

                # Prepare input window (unlocked, can be slow)
                # Shape must be (1, window_size, 4) for TFLite model
                window_size = 60
                if len(v_arr) < window_size:
                    # Pad with edge values to reach window_size
                    v_pad = np.pad(v_arr, (window_size - len(v_arr), 0), mode="edge")
                    i_pad = np.pad(i_arr, (window_size - len(i_arr), 0), mode="edge")
                    spd_pad = np.pad(spd_arr, (window_size - len(spd_arr), 0), mode="edge")
                    tmp_pad = np.pad(tmp_arr, (window_size - len(tmp_arr), 0), mode="edge")
                else:
                    # Use last window_size samples
                    v_pad = v_arr[-window_size:]
                    i_pad = i_arr[-window_size:]
                    spd_pad = spd_arr[-window_size:]
                    tmp_pad = tmp_arr[-window_size:]

                # Stack to (window_size, 4)
                window = np.stack([v_pad, i_pad, tmp_pad, spd_pad], axis=1).astype(np.float32)

                # Normalize input dùng min/max thật từ training data
                window_df   = pd.DataFrame(window, columns=["pack_voltage_v", "pack_current_a", "temp_c", "speed_kmh"])
                window_norm = apply_minmax(window_df, feature_norm).to_numpy(dtype=np.float32)

                # Channels-last batch cho TFLite: (1, window_size, 4)
                window_batch = window_norm[np.newaxis, :]  # (1, 60, 4) channels-last

                # SoC #3: CNN1D model inference
                soc_model = soc_inference.predict(window_batch)

                # Warm-up gate (1.2): chỉ dùng model khi đã có ≥60 sample thật.
                # Khi buffer chưa đầy, cạnh padding làm model thấy dữ liệu giả → sai lệch lớn.
                has_real_samples = len(v_arr) >= window_size
                if not has_real_samples:
                    soc_model = soc_bms_now  # giữ BMS làm fallback trong warm-up

                # SoH từ odo (1.1): tách hoàn toàn khỏi CNN1D
                soh = soh_estimator.estimate(odo_km_now)

                # Range estimation — 3 nguồn SoC, dùng chung speed/current window + SoH
                # (consumption độc lập nguồn SoC, xem range_estimator.py)
                range_model, wh_per_km = range_estimators["model"].update_and_estimate(
                    soc_pct=soc_model,
                    soh_pct=soh,
                    speed_window=spd_pad[-60:],  # Last 60 from padded array
                    current_window=i_pad[-60:],
                )
                range_bms, _ = range_estimators["bms"].update_and_estimate(
                    soc_pct=soc_bms_now,
                    soh_pct=soh,
                    speed_window=spd_pad[-60:],
                    current_window=i_pad[-60:],
                )
                range_cc, _ = range_estimators["cc"].update_and_estimate(
                    soc_pct=soc_cc_snap,
                    soh_pct=soh,
                    speed_window=spd_pad[-60:],
                    current_window=i_pad[-60:],
                )

                # GPS snapshot — đồng bộ theo state.timestamp (mốc chung của row 1Hz)
                gps_snapshot = gps_reader.get_latest()

                # Write results + tính sai số & MAE (locked, fast)
                with lock:
                    state.soc_model = soc_model
                    state.soh = soh
                    state.range_model = range_model
                    state.range_bms   = range_bms
                    state.range_cc    = range_cc
                    state.range_km    = range_model   # alias giữ tương thích display/logger cũ
                    state.wh_per_km   = wh_per_km

                    state.gps_lat = gps_snapshot["lat"]
                    state.gps_lon = gps_snapshot["lon"]
                    state.gps_speed_kmh = gps_snapshot["speed_kmh"]
                    state.gps_fix = gps_snapshot["fix"]
                    state.gps_sats = gps_snapshot["sats"]
                    state.gps_distance_km = gps_snapshot["distance_km"]

                    # ========================================================
                    # [THÊM MỚI TẠI ĐÂY 1] Tính sai số lộ trình (x - gps)
                    # Giả sử bạn đã khởi tạo state.route_total_km = 40.0 (hoặc độ dài lộ trình của bạn)
                    # trong class SharedState
                    # ========================================================
                    # Lấy tổng lộ trình, nếu chưa khai báo thì mặc định là 40km
                    total_route = getattr(state, 'route_total_km', 40.0) 
                    state.route_error_km = round(total_route - state.gps_distance_km, 3)
                    # Đảm bảo sai số không bị âm nếu xe chạy quá lộ trình dự kiến
                    if state.route_error_km < 0:
                        state.route_error_km = 0.0
                    # ========================================================

                    # Sai số tức thời so với BMS (ground truth)
                    err_model = soc_model - state.soc_bms
                    err_cc    = state.soc_cc - state.soc_bms

                    # Gate warm-up: chỉ tích MAE khi voltage_buffer đã đầy 60 sample
                    mae_warm = len(state.voltage_buffer) >= window_size
                    if mae_warm:
                        mae_model_window.append(abs(err_model))
                        mae_cc_window.append(abs(err_cc))
                        mae_model = sum(mae_model_window) / len(mae_model_window)
                        mae_cc    = sum(mae_cc_window)    / len(mae_cc_window)
                    else:
                        mae_model = 0.0
                        mae_cc    = 0.0

                    state.err_model = err_model
                    state.err_cc    = err_cc
                    state.mae_model = mae_model
                    state.mae_cc    = mae_cc
                    state.mae_warm  = mae_warm
                
                Kết thúc with lock
                
                # ========================================================
                # [THÊM MỚI TẠI ĐÂY 2] Ghi dữ liệu ra file CSV
                # Thực hiện NGOÀI khối lock để không block tiến trình đọc CAN
                # ========================================================
                # Chỉ ghi log khi đã bắt được tọa độ và xe đang thực sự di chuyển
                if state.gps_lat is not None and state.gps_speed_kmh > 1.0:
                    try:
                        import csv
                        import time
                        import os
                        
                        csv_file = "comparison_results.csv"
                        
                        # Tạo header nếu file chưa tồn tại
                        if not os.path.exists(csv_file):
                            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerow(["Timestamp", "Coulomb_Range_km", "CNN1D_Range_km", "Route_Error_km"])
                        
                        # Ghi dòng dữ liệu mới
                        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                time.strftime("%H:%M:%S"),
                                round(state.range_cc, 2),
                                round(state.range_model, 2),
                                state.route_error_km
                            ])
                    except Exception as e:
                        logger.error(f"Lỗi khi ghi file CSV so sánh: {e}")
                # ========================================================

                # Log to CSV (outside lock)
                runtime_logger.write(state)

                # Phát sai số & MAE cho client (1Hz, ngoài lock)
                if socketio is not None:
                    socketio.emit("update_mae", {
                        "t":         time.time(),
                        "err_model": round(err_model, 2),
                        "err_cc":    round(err_cc,    2),
                        "mae_model": round(mae_model, 2),
                        "mae_cc":    round(mae_cc,    2),
                        "warm":      mae_warm,
                    })

                # Log to console
                logger.info(
                    f"[1Hz] SoC: BMS={state.soc_bms:.1f}%, "
                    f"CC={state.soc_cc:.1f}%, Model={state.soc_model:.1f}% | "
                    f"Range: BMS={state.range_bms:.1f} CC={state.range_cc:.1f} "
                    f"Model={state.range_model:.1f} km | SoH={state.soh:.1f}% | "
                    f"GPS={state.gps_distance_km:.2f}km | Sai số={state.route_error_km:.2f}km | "
                    f"Log: {runtime_logger.get_row_count()} rows, {runtime_logger.get_file_size_mb():.1f}MB"
                )

            except Exception as e:
                logger.error(f"Inference error: {e}")

        tick += 1

        # Track FPS (using fixed-size deque for efficiency)
        elapsed = time.monotonic() - t0
        tick_times.append(elapsed)

        if len(tick_times) == TICK_HZ:  # Full window available
            tick_sum = sum(tick_times)
            if tick_sum > 0:
                with lock:
                    state.fps_actual = TICK_HZ / tick_sum

        # Keep 10Hz timing
        sleep_time = max(0, TICK_DT - elapsed)
        time.sleep(sleep_time)


def start_web_server(state: SharedState, lock: threading.Lock):
    """
    Khởi động Flask + SocketIO web server trên thread riêng.

    Web server chỉ ĐỌC state (không sửa), thông qua lock.
    Trả về socketio instance để main_loop() emit events.

    Args:
        state: SharedState instance.
        lock: threading.Lock để sync access.

    Returns:
        socketio instance để emit từ main loop.
    """
    from flask import Flask, jsonify, render_template
    from flask_socketio import SocketIO
    from pathlib import Path

    template_dir = Path(__file__).parent / "display" / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    @app.route("/")
    def index():
        """Dashboard chính — 3 SoC sources + speed + power bar."""
        return render_template("index.html")

    @app.route("/bms")
    def bms_page():
        """BMS detail page — 22 cell voltages."""
        return render_template("bms.html")

    @app.route("/api/state")
    def get_state():
        """REST fallback — lấy current state (JSON)."""
        with lock:
            return jsonify(state.get_dict())

    @app.route("/health")
    def health():
        """Health check."""
        return jsonify({"status": "ok"})

    def run_server():
        logger.info(f"Starting web server at {WEB_SERVER_HOST}:{WEB_SERVER_PORT}...")
        socketio.run(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT, debug=False)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info("✓ Web server thread started (SocketIO)")
    return socketio


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("VinFast Evo 200 SoC/SoH Runtime System")
    logger.info("=" * 80)

    # Shared state + lock
    state = SharedState()
    lock = threading.Lock()

    # Initialize all modules
    try:
        can_reader, coulomb_counter, soc_inference, soh_estimator, range_estimators, runtime_logger, gps_reader = init_system()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    # Start web server (background thread), nhận socketio để emit từ main loop
    socketio = start_web_server(state, lock)

    # Main loop (this thread)
    try:
        main_loop(
            can_reader,
            coulomb_counter,
            soc_inference,
            soh_estimator,
            range_estimators,
            runtime_logger,
            gps_reader,
            state,
            lock,
            socketio,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        can_reader.disconnect()
        gps_reader.stop()
        logger.info("Goodbye!")


if __name__ == "__main__":
    main()
