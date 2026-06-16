"""
Test Preprocessing Module — sign flip, startup skip, normalization, windowing.

Smoke tests để kiểm tra:
- Sign flip: I > 0 khi xả (project convention)
- Startup skip: bỏ các sample đầu khi cảm biến chưa init
- Normalization: minmax và zscore hoạt động đúng
- Windowing: tạo sliding window dataset đúng shape
"""

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import normalize, loader, windowing


def test_sign_flip():
    """Kiểm tra đảo dấu dòng điện: CSV gốc (I<0 xả) -> project (I>0 xả)."""
    # Tạo sample CSV tạm
    csv_data = """Thoi Gian,Dien Ap (V),Dong Dien (A),SOC (%),Nhiet Do (C),Van toc (km/h),ODO (km)
10:00:00,75.2,-15.5,80.5,25.0,50.0,100.0
10:00:01,75.1,-15.6,80.4,25.1,50.5,100.1
10:00:02,75.0,-15.4,80.3,25.0,51.0,100.2
"""
    with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_data)
        temp_path = f.name

    try:
        df = loader.load_evo200_csv(temp_path)

        # CSV gốc có I < 0 khi xả (-15.5, -15.6, -15.4)
        # Sau load_evo200_csv, phải là I > 0 (15.5, 15.6, 15.4)
        assert (df['pack_current_a'] > 0).all(), \
            f"Expected all currents > 0 (discharge), got: {df['pack_current_a'].values}"

        assert np.isclose(df['pack_current_a'].iloc[0], 15.5, atol=0.1), \
            f"Expected 15.5, got {df['pack_current_a'].iloc[0]}"

        print("[OK] Sign flip: CSV I<0 (discharge) -> project I>0 (discharge)")

    finally:
        Path(temp_path).unlink()


def test_startup_skip():
    """Kiểm tra skip sensor startup (SOC=0, Temp=0 đầu file)."""
    # CSV với sensor startup chưa init (đầu tiên 2 dòng có SOC=0, Temp=0)
    csv_data = """Thoi Gian,Dien Ap (V),Dong Dien (A),SOC (%),Nhiet Do (C),Van toc (km/h),ODO (km)
10:00:00,75.0,-10.0,0.0,0.0,0.0,100.0
10:00:01,75.1,-10.1,0.0,0.0,5.0,100.1
10:00:02,75.2,-10.2,50.0,25.0,50.0,100.2
10:00:03,75.3,-10.3,51.0,25.1,51.0,100.3
"""
    with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_data)
        temp_path = f.name

    try:
        df = loader.load_evo200_csv(temp_path)

        # Phải skip 2 dòng đầu (SOC=0, Temp=0)
        # Lấy từ dòng 3 (10:00:02)
        assert len(df) == 2, f"Expected 2 rows after skip, got {len(df)}"
        assert df['soc_bms'].iloc[0] >= 50.0, \
            f"Expected first valid SoC >= 50, got {df['soc_bms'].iloc[0]}"
        assert df['temp_c'].iloc[0] >= 25.0, \
            f"Expected first valid Temp >= 25, got {df['temp_c'].iloc[0]}"

        print("[OK] Startup skip: removed sensor init samples (SOC=0, Temp=0)")

    finally:
        Path(temp_path).unlink()


def test_normalize_minmax():
    """Kiểm tra chuẩn hóa Min-Max đưa giá trị về [0, 1]."""
    df = pd.DataFrame({
        'voltage': [60.0, 70.0, 80.0],
        'current': [0.0, 12.5, 25.0],
    })

    df_norm = normalize.normalize_minmax(df)

    # Min-Max: (x - min) / (max - min)
    # voltage: (60-60)/(80-60) = 0, (70-60)/20 = 0.5, (80-60)/20 = 1.0
    assert np.isclose(df_norm['voltage'].iloc[0], 0.0), \
        f"Expected 0.0, got {df_norm['voltage'].iloc[0]}"
    assert np.isclose(df_norm['voltage'].iloc[1], 0.5), \
        f"Expected 0.5, got {df_norm['voltage'].iloc[1]}"
    assert np.isclose(df_norm['voltage'].iloc[2], 1.0), \
        f"Expected 1.0, got {df_norm['voltage'].iloc[2]}"

    # current: (0-0)/(25-0) = 0, (12.5-0)/25 = 0.5, (25-0)/25 = 1.0
    assert np.isclose(df_norm['current'].iloc[1], 0.5), \
        f"Expected 0.5, got {df_norm['current'].iloc[1]}"

    print("[OK] Min-Max normalization: values in [0, 1]")


def test_normalize_zscore():
    """Kiểm tra chuẩn hóa Z-score: meanapprox0, stdapprox1."""
    df = pd.DataFrame({
        'value': [10.0, 20.0, 30.0],
    })

    df_norm = normalize.normalize_zscore(df)

    # mean = 20, std = 10
    # (10-20)/10 = -1, (20-20)/10 = 0, (30-20)/10 = 1
    assert np.isclose(df_norm['value'].iloc[0], -1.0), \
        f"Expected -1.0, got {df_norm['value'].iloc[0]}"
    assert np.isclose(df_norm['value'].iloc[1], 0.0), \
        f"Expected 0.0, got {df_norm['value'].iloc[1]}"
    assert np.isclose(df_norm['value'].iloc[2], 1.0), \
        f"Expected 1.0, got {df_norm['value'].iloc[2]}"

    # Kiểm tra meanapprox0 (std varies due to small sample size)
    assert np.isclose(df_norm['value'].mean(), 0.0, atol=1e-10)

    print("[OK] Z-score normalization: meanapprox0, stdapprox1")


def test_create_cnn1d_dataset():
    """Kiểm tra tạo sliding window dataset với shape (n_windows, window_size, 4)."""
    # Tạo DataFrame với 100 sample
    df = pd.DataFrame({
        'pack_voltage_v': np.linspace(60, 80, 100),
        'pack_current_a': np.linspace(0, 25, 100),
        'temp_c': np.linspace(-10, 60, 100),
        'speed_kmh': np.linspace(0, 80, 100),
        'soc_bms': np.linspace(100, 0, 100),  # Giảm từ 100% xuống 0%
    })

    window_size = 10
    step = 1

    x, y = windowing.create_cnn1d_dataset(df, window_size=window_size, step=step)

    # n_windows = (100 - 10) / 1 + 1 = 91
    assert x.shape == (91, 10, 4), \
        f"Expected shape (91, 10, 4), got {x.shape}"
    assert y.shape == (91,), \
        f"Expected y shape (91,), got {y.shape}"

    # Kiểm tra first window: 10 sample đầu
    assert np.allclose(x[0, :, 0], df['pack_voltage_v'].iloc[0:10].values), \
        "First window voltage doesn't match"

    # Kiểm tra target y[0] = soc_bms[9] (timestep cuối của window đầu)
    assert np.isclose(y[0], df['soc_bms'].iloc[9]), \
        f"Expected y[0]={df['soc_bms'].iloc[9]}, got {y[0]}"

    print(f"[OK] Sliding window dataset: shape {x.shape}, {len(y)} targets")


def test_create_cnn1d_dataset_non_overlapping():
    """Kiểm tra non-overlapping window (step=window_size)."""
    df = pd.DataFrame({
        'pack_voltage_v': np.arange(100),
        'pack_current_a': np.arange(100),
        'temp_c': np.arange(100),
        'speed_kmh': np.arange(100),
        'soc_bms': np.arange(100, 0, -1),
    })

    window_size = 10
    step = 10  # Non-overlapping

    x, y = windowing.create_cnn1d_dataset(df, window_size=window_size, step=step)

    # n_windows = (100 - 10) / 10 + 1 = 10
    assert x.shape == (10, 10, 4), \
        f"Expected shape (10, 10, 4), got {x.shape}"
    assert len(y) == 10

    print("[OK] Non-overlapping windows: step=window_size works")


def test_create_cnn1d_dataset_short_data():
    """Kiểm tra ValueError khi dữ liệu quá ngắn."""
    df = pd.DataFrame({
        'pack_voltage_v': [75.0, 75.1],
        'pack_current_a': [10.0, 10.1],
        'temp_c': [25.0, 25.1],
        'speed_kmh': [50.0, 50.1],
        'soc_bms': [80.0, 79.5],
    })

    try:
        windowing.create_cnn1d_dataset(df, window_size=10, step=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "ngắn hơn window_size" in str(e)
        print("[OK] ValueError for short data correctly raised")


def test_keep_only_y_end():
    """Kiểm tra hàm lấy mẫu thưa trên mảng nhãn."""
    y = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ])

    y_sparse = windowing.keep_only_y_end(y, step=2)

    # Mỗi step-th phần tử: [1, 3, 5] và [6, 8, 10]
    assert y_sparse.shape == (2, 3), \
        f"Expected shape (2, 3), got {y_sparse.shape}"
    assert np.array_equal(y_sparse[0], [1, 3, 5])
    assert np.array_equal(y_sparse[1], [6, 8, 10])

    print("[OK] keep_only_y_end: sparse sampling works")


def test_timestamp_parsing_midnight_crossing():
    """Kiểm tra parse timestamp qua nửa đêm."""
    # Tạo CSV với timestamp qua nửa đêm (23:59:xx -> 00:00:xx)
    csv_data = """Thoi Gian,Dien Ap (V),Dong Dien (A),SOC (%),Nhiet Do (C),Van toc (km/h),ODO (km)
23:59:58,75.0,-10.0,50.0,25.0,50.0,100.0
23:59:59,75.1,-10.1,50.1,25.1,50.1,100.1
00:00:00,75.2,-10.2,50.2,25.0,50.2,100.2
00:00:01,75.3,-10.3,50.3,25.1,50.3,100.3
"""
    with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(csv_data)
        temp_path = f.name

    try:
        df = loader.load_evo200_csv(temp_path)

        # Timestamp phải tăng đơn điệu
        timestamps = df['timestamp'].values
        time_diffs = np.diff(timestamps.astype(np.int64))

        assert (time_diffs > 0).all(), \
            "Timestamps should be monotonically increasing"

        print("[OK] Midnight crossing: timestamps monotonically increasing")

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    print("Testing Preprocessing Module...")
    print()

    test_sign_flip()
    test_startup_skip()
    test_normalize_minmax()
    test_normalize_zscore()
    test_create_cnn1d_dataset()
    test_create_cnn1d_dataset_non_overlapping()
    test_create_cnn1d_dataset_short_data()
    test_keep_only_y_end()
    test_timestamp_parsing_midnight_crossing()

    print()
    print("All preprocessing smoke tests passed!")
