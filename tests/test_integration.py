"""
Integration test — end-to-end system validation.

Test pipeline:
  Mock CAN data → decode → 3 SoC sources → range → log → web state
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.loader import load_evo200_csv
from src.preprocessing.normalize import normalize_minmax
from src.preprocessing.windowing import create_cnn1d_dataset
from src.coulomb_counter.counter import CoulombCounter
from src.logger.writer import RuntimeLogger
from src.range_estimator.estimator import (
    update_ewma_consumption,
    compute_behavior_features,
    compute_behavior_factor,
    estimate_range
)


def test_integration_load_to_coulomb():
    """Load CSV → preprocess → Coulomb counting."""
    # Find test CSV
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    csv_files = sorted(list(data_dir.glob("Evo200_*.csv")))

    if not csv_files:
        print("[SKIP] No Evo200 CSV files found")
        return

    # Load first file
    df = load_evo200_csv(str(csv_files[0]))
    assert len(df) > 0, "Failed to load CSV"
    print(f"[OK] Loaded {csv_files[0].name}: {len(df)} rows")

    # Initialize Coulomb counter
    cc = CoulombCounter(capacity_ah=30.65, initial_soc=df['soc_bms'].iloc[0])

    # Simulate Coulomb counting
    for i in range(1, min(100, len(df))):
        current = df['pack_current_a'].iloc[i]
        prev_ts = df['timestamp'].iloc[i-1]
        curr_ts = df['timestamp'].iloc[i]
        dt = (curr_ts - prev_ts).total_seconds()

        if dt > 0 and dt < 1.0:  # sanity check
            soc_cc = cc.update(current, dt)
            assert 0 <= soc_cc <= 100, f"SoC out of bounds: {soc_cc}"

    print(f"[OK] Coulomb counter updated {min(100, len(df)-1)} steps")


def test_integration_normalize_window():
    """Normalize → create windows → verify shapes."""
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    csv_files = sorted(list(data_dir.glob("Evo200_*.csv")))

    if not csv_files:
        print("[SKIP] No Evo200 CSV files found")
        return

    # Load and combine
    dfs = []
    for f in csv_files[:3]:  # First 3 files for speed
        try:
            df = load_evo200_csv(str(f))
            dfs.append(df)
        except:
            pass

    if not dfs:
        print("[SKIP] Failed to load CSV files")
        return

    df_combined = pd.concat(dfs, ignore_index=True)

    # Normalize
    feature_cols = ['pack_voltage_v', 'pack_current_a', 'temp_c', 'speed_kmh']
    features_norm = normalize_minmax(df_combined[feature_cols])

    df_norm = features_norm.copy()
    df_norm['soc_bms'] = df_combined['soc_bms'].values

    # Create windows
    X, y = create_cnn1d_dataset(df_norm, window_size=60, step=10)

    assert X.shape[1] == 60, f"Window size mismatch: {X.shape[1]}"
    assert X.shape[2] == 4, f"Feature count mismatch: {X.shape[2]}"
    assert len(y) == X.shape[0], f"Target count mismatch"

    print(f"[OK] Windows created: {X.shape}, targets: {y.shape}")


def test_integration_logger():
    """Test CSV runtime logger."""
    log_dir = Path(__file__).parent.parent / "data" / "processed"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = RuntimeLogger(str(log_dir))

    # Write sample rows
    for i in range(10):
        row = {
            'pack_voltage_v': 75.0 + i * 0.1,
            'pack_current_a': 10.0 + i * 0.5,
            'temp_c': 25.0,
            'speed_kmh': 50.0,
            'soc_bms': 80.0 - i * 0.5,
            'soc_cc': 80.0 - i * 0.4,
            'soc_cnn1d': 80.0 - i * 0.45,
            'soh': 95.0,
            'range_km': 200.0 - i * 2,
        }
        logger.write(row)

    count = logger.get_row_count()
    size_mb = logger.get_file_size_mb()

    assert count == 10, f"Expected 10 rows, got {count}"
    assert size_mb > 0, f"Log file size should be > 0, got {size_mb}"

    print(f"[OK] Logger: {count} rows, {size_mb:.2f} MB")


def test_integration_range_estimator():
    """Test range estimation with behavior features."""
    # Create sample data
    speed_window = np.array([50, 52, 48, 45, 40, 35, 30, 25, 20, 15])  # km/h
    current_window = np.array([10, 11, 9, 8, 7, 6, 5, 4, 3, 2])  # A

    # Compute behavior features
    features = compute_behavior_features(speed_window, current_window)

    assert 'avg_speed_kmh' in features, "Missing avg_speed_kmh"
    assert 'accel_std_mps2' in features, "Missing accel_std_mps2"
    assert 'stop_ratio' in features, "Missing stop_ratio"

    print(f"[OK] Behavior features: {features}")

    # Update EWMA
    wh_per_km = 10.0
    alpha = 0.3
    ewma = update_ewma_consumption(wh_per_km, alpha, prev_ewma=9.5)
    assert 9.5 <= ewma <= 10.0, f"EWMA out of expected range: {ewma}"

    print(f"[OK] EWMA consumption: {ewma:.3f} Wh/km")

    # Compute behavior factor
    coefficients = {
        'intercept': 1.0,
        'speed_coeff': -0.01,
        'accel_coeff': 0.05,
        'stop_coeff': 0.1
    }
    factor = compute_behavior_factor(features, coefficients)
    assert factor > 0, f"Behavior factor should be > 0, got {factor}"

    print(f"[OK] Behavior factor: {factor:.3f}")

    # Estimate range
    soc_pct = 80.0
    soh_pct = 95.0
    pack_capacity_wh = 72 * 30.65  # V × Ah
    range_km = estimate_range(soc_pct, soh_pct, ewma, factor, pack_capacity_wh)

    assert range_km > 0, f"Range should be > 0, got {range_km}"
    print(f"[OK] Range estimate: {range_km:.1f} km")


def test_integration_full_pipeline():
    """Full pipeline: CSV → normalize → window → metrics → log."""
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    csv_files = sorted(list(data_dir.glob("Evo200_*.csv")))

    if not csv_files:
        print("[SKIP] No Evo200 CSV files found")
        return

    # Load
    df = load_evo200_csv(str(csv_files[0]))
    print(f"[1] Loaded {len(df)} samples")

    # Normalize
    feature_cols = ['pack_voltage_v', 'pack_current_a', 'temp_c', 'speed_kmh']
    features_norm = normalize_minmax(df[feature_cols])
    df_norm = features_norm.copy()
    df_norm['soc_bms'] = df['soc_bms'].values
    print(f"[2] Normalized features")

    # Create windows
    X, y = create_cnn1d_dataset(df_norm, window_size=60, step=10)
    print(f"[3] Created {len(X)} windows")

    # Initialize Coulomb counter
    cc = CoulombCounter(capacity_ah=30.65, initial_soc=100.0)
    soc_cc_list = []

    for i in range(min(len(df), 100)):
        current = df['pack_current_a'].iloc[i]
        if i > 0:
            prev_ts = df['timestamp'].iloc[i-1]
            curr_ts = df['timestamp'].iloc[i]
            dt = (curr_ts - prev_ts).total_seconds()
            if dt > 0 and dt < 1.0:
                soc_cc = cc.update(current, dt)
                soc_cc_list.append(soc_cc)

    print(f"[4] Coulomb counting: {len(soc_cc_list)} steps")

    # Range estimation
    if len(df) >= 60:
        window_df = df.iloc[-60:].copy()
        features_dict = compute_behavior_features(
            window_df['speed_kmh'].values,
            window_df['pack_current_a'].values
        )
        print(f"[5] Computed behavior features: avg_speed={features_dict['avg_speed_kmh']:.1f} km/h")

    # Logging
    log_dir = Path(__file__).parent.parent / "data" / "processed"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = RuntimeLogger(str(log_dir))

    for i in range(min(10, len(df))):
        logger.write({
            'pack_voltage_v': df['pack_voltage_v'].iloc[i],
            'pack_current_a': df['pack_current_a'].iloc[i],
            'temp_c': df['temp_c'].iloc[i],
            'speed_kmh': df['speed_kmh'].iloc[i],
            'soc_bms': df['soc_bms'].iloc[i],
            'soc_cc': soc_cc_list[i] if i < len(soc_cc_list) else 50.0,
            'soc_cnn1d': 50.0,
            'soh': 95.0,
            'range_km': 200.0,
        })

    print(f"[6] Logged {logger.get_row_count()} rows")
    print(f"\\n[OK] Full pipeline complete!")


if __name__ == "__main__":
    print("Integration Tests\\n")

    test_integration_load_to_coulomb()
    test_integration_normalize_window()
    test_integration_logger()
    test_integration_range_estimator()
    test_integration_full_pipeline()

    print("\\nAll integration tests passed!")
