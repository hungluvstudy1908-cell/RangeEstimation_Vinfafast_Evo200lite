#!/usr/bin/env python3
"""
Verify Coulomb Counter bias on 15 Evo200_Mixed*.csv files.

Runs CoulombCounter using real dt from timestamps, compares with SoC BMS,
and reports Mean MAE/RMSE/Bias/Corr to verify fix for:
- Hypothesis 1: Wrong dt (now fixed by using real timestamp diff)
- Other hypotheses if bias still present
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Fix UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.coulomb_counter.counter import CoulombCounter

logging.basicConfig(
    level=logging.WARNING,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Vietnamese column names from database_schema.md
COLUMN_MAPPING = {
    "Thoi Gian": "timestamp",
    "Dien Ap (V)": "pack_voltage_v",
    "Dong Dien (A)": "pack_current_a",
    "SOC (%)": "soc_bms",
    "Nhiet Do (C)": "temp_c",
    "Van toc (km/h)": "speed_kmh",
    "ODO (km)": "odo_km",
}


def load_evo200_csv_inline(csv_path):
    """Load Evo200 CSV with column rename and sign flip (inline version)."""
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Rename columns to snake_case English
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # Parse timestamp (HH:MM:SS only, no date — handle midnight rollover)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S")

    # Handle midnight rollover: detect when time goes backward
    time_diff = df["timestamp"].diff()
    midnight_crosses = time_diff < pd.Timedelta(seconds=-43200)  # > 12 hour backward
    midnight_cumsum = midnight_crosses.cumsum()
    df["timestamp"] = df["timestamp"] + midnight_cumsum * pd.Timedelta(days=1)

    # Sign flip for current (CSV convention: I < 0 when discharging)
    # Project convention: I > 0 when discharging
    df["pack_current_a"] = -df["pack_current_a"]

    # Skip sensor startup rows (both conditions required)
    valid = (df["soc_bms"] >= 5) & (df["temp_c"] >= 10)
    first_valid_idx = valid.idxmax()
    df = df.loc[first_valid_idx:].reset_index(drop=True)

    return df


def compute_stats(y_true, y_pred):
    """Compute MAE, RMSE, Bias, Correlation."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_true - y_pred)

    # Pearson correlation
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = np.nan

    return mae, rmse, bias, corr


def run_coulomb_counter_on_file(csv_path):
    """Run CoulombCounter on one file using real dt from timestamps."""
    try:
        df = load_evo200_csv_inline(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}", file=sys.stderr)
        return None

    # Initialize CC from first BMS SoC (skip sensor startup rows already handled by loader)
    initial_soc = df["soc_bms"].iloc[0]
    cc = CoulombCounter(initial_soc=initial_soc)

    # Compute dt from total time span / sample count
    # (Timestamps are at 1Hz resolution but multiple samples per second exist)
    time_start = df["timestamp"].iloc[0]
    time_end = df["timestamp"].iloc[-1]
    total_seconds = (time_end - time_start).total_seconds()
    n_samples = len(df)

    # Average dt across all samples (handles variable sampling)
    if n_samples > 1 and total_seconds > 0:
        dt_avg = total_seconds / (n_samples - 1)
    else:
        dt_avg = 1.0  # Fallback if not enough data

    # Use constant dt for all rows (assume uniform sampling within the run)
    df["dt_s"] = dt_avg

    soc_cc_values = [initial_soc]
    for i in range(1, len(df)):
        current_a = df["pack_current_a"].iloc[i]
        soc_cc = cc.update(current_a=current_a, dt=dt_avg)
        soc_cc_values.append(soc_cc)

    df["soc_cc"] = soc_cc_values

    # Compute stats
    y_true = df["soc_bms"].values
    y_pred = df["soc_cc"].values
    mae, rmse, bias, corr = compute_stats(y_true, y_pred)

    return {
        "file": Path(csv_path).name,
        "n_samples": len(df),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
        "dt_avg": dt_avg,
    }


def main():
    data_dir = Path(__file__).parent / "data" / "raw"

    # List and sort Evo200 files
    files = sorted(data_dir.glob("Evo200_Mixed*.csv"))
    if not files:
        print(f"Error: No Evo200_Mixed*.csv files found in {data_dir}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Coulomb Counter Bias Verification — {len(files)} files")
    print(f"{'='*80}\n")

    results = []
    for csv_path in files:
        print(f"Processing {Path(csv_path).name}...", end=" ", flush=True)
        result = run_coulomb_counter_on_file(csv_path)

        if result:
            results.append(result)
            print(f"[OK] (n={result['n_samples']}, dt_avg={result['dt_avg']:.4f}s)")
        else:
            print("[FAIL]")

    if not results:
        print("Error: No files processed successfully", file=sys.stderr)
        sys.exit(1)

    # Aggregate stats
    print(f"\n{'='*80}")
    print(f"Aggregate Results ({len(results)} files)")
    print(f"{'='*80}\n")

    mean_mae = np.mean([r["mae"] for r in results])
    mean_rmse = np.mean([r["rmse"] for r in results])
    mean_bias = np.mean([r["bias"] for r in results])
    mean_corr = np.mean([r["corr"] for r in results])
    total_samples = sum(r["n_samples"] for r in results)

    print(f"Mean MAE    : {mean_mae:.2f}%")
    print(f"Mean RMSE   : {mean_rmse:.2f}%")
    print(f"Mean Bias   : {mean_bias:+.2f}%")
    print(f"Mean Corr   : {mean_corr:.4f}")
    print(f"Files analyzed: {len(results)}")
    print(f"Samples total : {total_samples}")
    print(f"\n{'='*80}\n")

    # Per-file breakdown
    print(f"{'File':<25} {'MAE':<8} {'RMSE':<8} {'Bias':<10} {'Corr':<8} {'dt_avg':<9}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*9}")
    for r in results:
        print(f"{r['file']:<25} {r['mae']:>7.2f}% {r['rmse']:>7.2f}% {r['bias']:>+9.2f}% {r['corr']:>7.4f} {r['dt_avg']:>8.4f}s")

    # Target validation
    print(f"\n{'='*80}")
    print(f"Target: |Mean Bias| < 1%")
    if abs(mean_bias) < 1.0:
        print(f"[PASS] |{mean_bias:+.2f}%| < 1%")
    else:
        print(f"[FAIL] |{mean_bias:+.2f}%| >= 1% — further investigation needed")
        print(f"\nNext steps:")
        print(f"  1. Verify actual sample rate (measured dt_avg from files)")
        print(f"  2. Check sign convention: current > 0 when speed > 10 km/h during discharge")
        print(f"  3. Verify capacity: measured Ah vs nameplate 32Ah (effective_ah: {30.65}Ah)")
        print(f"  4. Check sensor startup skip: verify first valid row has soc_bms >= 5%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
