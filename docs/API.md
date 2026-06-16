# API Reference

## Overview

This document describes the public API for the VinFast Evo 200 SoC monitoring system.

## Modules

### Preprocessing

#### `src.preprocessing.loader`

Load and parse Evo200 CSV files.

```python
from src.preprocessing.loader import load_evo200_csv, list_evo200_files

# Load single CSV file
df = load_evo200_csv('data/raw/Evo200_Mixed1.csv')
# Returns DataFrame with columns:
#   - pack_voltage_v: Pack voltage (V)
#   - pack_current_a: Pack current (A), I>0 = discharge
#   - temp_c: Pack temperature (°C)
#   - speed_kmh: Vehicle speed (km/h)
#   - soc_bms: SoC from BMS (%)
#   - timestamp: Sample timestamp (datetime64)

# List all Evo200 files
files = list_evo200_files('data/raw/')
```

#### `src.preprocessing.normalize`

Normalize features to [0, 1] range.

```python
from src.preprocessing.normalize import normalize_minmax, normalize_zscore

# Min-max normalization to [0, 1]
df_norm = normalize_minmax(df[['pack_voltage_v', 'pack_current_a']])

# Z-score normalization (mean=0, std=1)
df_norm = normalize_zscore(df[['pack_voltage_v', 'pack_current_a']])
```

#### `src.preprocessing.windowing`

Create sliding window dataset for CNN1D.

```python
from src.preprocessing.windowing import create_cnn1d_dataset

# Create windowed dataset
# Input: DataFrame with normalized features + soc_bms target
# Output: (X, y) numpy arrays
#   - X shape: (n_windows, window_size, n_features)
#   - y shape: (n_windows,)

X, y = create_cnn1d_dataset(
    df,
    window_size=60,    # 6 seconds at 10Hz
    step=10            # Overlapping windows
)
```

### CAN Reader

#### `src.can_reader.reader`

USB-CAN communication interface.

```python
from src.can_reader.reader import WaveshareReader

reader = WaveshareReader(port='/dev/ttyUSB0', baudrate=115200)
reader.connect()

# Read single CAN frame (blocking, timeout=0.1s)
can_id, data = reader.read_frame()

reader.close()
```

#### `src.can_reader.decoder`

Decode CAN frames into signals.

```python
from src.can_reader.decoder import decode

# Decode CAN frame
signals = decode(can_id=0x308, data=data)
# Returns dict like:
# {
#     'cell_v_1': 3.45,
#     'cell_v_2': 3.46,
#     ...
# }
```

### Coulomb Counter

#### `src.coulomb_counter.counter`

Coulomb counting for SoC estimation.

```python
from src.coulomb_counter.counter import CoulombCounter

cc = CoulombCounter(
    capacity_ah=30.65,      # Evo 200 capacity
    initial_soc=80.0        # Starting SoC (%)
)

# Update with current sample
soc_cc = cc.update(
    current_a=15.0,  # Current (A), I>0 = discharge
    dt=0.1           # Time since last update (seconds)
)
# Returns: SoC in [0, 100]%

# Check if should reset to BMS
if cc.should_reset(soc_bms=98.0):
    cc.reset(new_soc=98.0)
```

### SoC Inference

#### `src.soc_inference.inference`

CNN1D model inference using TFLite.

```python
from src.soc_inference.inference import SocInference

soc_model = SocInference(model_path='models/soc_cnn1d.tflite')

# Predict SoC from normalized window
soc_cnn1d, soh = soc_model.predict(
    window=X[0]  # shape (60, 4)
)
# Returns: (soc_percent, soh_percent)
```

### Range Estimator

#### `src.range_estimator.estimator`

Range estimation using behavior features.

```python
from src.range_estimator.estimator import (
    update_ewma_consumption,
    compute_behavior_features,
    compute_behavior_factor,
    estimate_range
)

# Chapter 1: Update EWMA consumption baseline
ewma = update_ewma_consumption(
    new_wh_per_km=10.5,
    alpha=0.3,              # Learning rate
    prev_ewma=10.0
)

# Chapter 3: Compute behavior features from 60-second windows
features = compute_behavior_features(
    speed_window=speed_array,      # shape (60,)
    current_window=current_array   # shape (60,)
)
# Returns dict:
# {
#     'avg_speed_kmh': 50.0,
#     'accel_std_mps2': 1.2,
#     'stop_ratio': 0.15
# }

# Chapter 4: Compute behavior factor via linear regression
factor = compute_behavior_factor(features, coefficients)

# Estimate remaining range
range_km = estimate_range(
    soc_pct=80.0,
    soh_pct=95.0,
    wh_per_km_ewma=ewma,
    behavior_factor=factor,
    pack_capacity_wh=2206.8  # 72V × 30.65Ah
)
```

### Logger

#### `src.logger.writer`

Runtime CSV logging.

```python
from src.logger.writer import RuntimeLogger

logger = RuntimeLogger(output_dir='data/processed/')

# Log runtime sample
logger.write({
    'pack_voltage_v': 75.2,
    'pack_current_a': 15.5,
    'temp_c': 25.0,
    'speed_kmh': 50.0,
    'soc_bms': 80.5,
    'soc_cc': 80.3,
    'soc_cnn1d': 80.4,
    'soh': 95.0,
    'range_km': 200.5,
})

# Check logging status
count = logger.get_row_count()
size_mb = logger.get_file_size_mb()
```

### Web Display

#### `src.display.web`

Flask web server for dashboard.

```python
from src.display.web import create_app

app = create_app(shared_state=shared_state)

# Run server (blocking)
app.run(host='0.0.0.0', port=8080, debug=False)
```

**Endpoints:**
- `GET /` — HTML dashboard with 3 battery icons
- `GET /api/state` — JSON state (for AJAX polling)
- `GET /api/history` — Recent history data

**Response format:**
```json
{
  "timestamp": "2026-06-04T10:30:15Z",
  "soc_bms_percent": 80.5,
  "soc_cc_percent": 80.3,
  "soc_cnn1d_percent": 80.4,
  "soh_percent": 95.0,
  "range_km": 200.5,
  "wh_per_km": 10.5,
  "pack_voltage_v": 75.2,
  "pack_current_a": 15.5,
  "temp_c": 25.0,
  "speed_kmh": 50.0,
  "can_status": "connected"
}
```

### Main Loop

#### `src.main`

Entry point for runtime system.

```python
from src.main import main_loop

# Run 10Hz main loop (blocking until SIGINT)
main_loop()
```

**Workflow:**
1. **10Hz tick**: Read CAN → decode → update Coulomb → buffer samples
2. **1Hz tick**: Resample → normalize window → CNN1D inference → range → log
3. **Flask thread**: Web server (daemon)

## Configuration Files

### `configs/battery_specs.yaml`

```yaml
pack_capacity_ah: 30.65
nominal_voltage_v: 72.0
min_voltage_v: 60.0
max_voltage_v: 80.0
min_current_a: -25.0
max_current_a: 35.0
min_temp_c: -10.0
max_temp_c: 60.0
min_speed_kmh: 0.0
max_speed_kmh: 80.0
soc_reset_threshold_percent: 98.0  # Coulomb reset trigger
```

### `configs/can_ids.yaml`

```yaml
can_frames:
  voltage_1_4: 0x308
  voltage_5_8: 0x309
  voltage_9_12: 0x30a
  # ... 8 more frames ...
  pack_current: 0x312
  pack_voltage: 0x313
  temperature: 0x314
  speed_odometer: 0x315
```

### `configs/range_estimator.yaml`

```yaml
ewma_alpha: 0.3  # Consumption baseline learning rate
pack_capacity_wh: 2206.8  # 72V × 30.65Ah
behavior_coefficients:
  intercept: 1.0
  speed_coeff: -0.01
  accel_coeff: 0.05
  stop_coeff: 0.1
window_size_samples: 60  # 6 seconds at 10Hz
```

### `configs/model.yaml`

```yaml
model_path: models/soc_cnn1d.tflite
window_size: 60
feature_cols: [pack_voltage_v, pack_current_a, temp_c, speed_kmh]
target_col: soc_bms
demo_mode: false
```

## Error Handling

All modules follow consistent error handling:

```python
try:
    df = load_evo200_csv('data/raw/invalid.csv')
except FileNotFoundError:
    print("CSV file not found")
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific module tests:
```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_coulomb_counter.py -v
pytest tests/test_integration.py -v
```

## Constants and Units

All units are SI unless noted:

| Variable | Unit | Range | Notes |
|----------|------|-------|-------|
| voltage | V (volt) | 60-80 | Pack voltage |
| current | A (ampere) | -25 to 35 | I>0 = discharge |
| temperature | °C | -10 to 60 | Pack temp |
| speed | km/h | 0-80 | Vehicle speed |
| soc | % | 0-100 | State of charge |
| soh | % | 0-100 | State of health |
| capacity | Ah | 30.65 | Evo 200 LFP |
| energy | Wh | 2206.8 | 72V × 30.65Ah |
| time | s | - | Always seconds |
| range | km | - | Remaining distance |
| consumption | Wh/km | - | Energy per km |

## Examples

### Complete Pipeline

```python
from src.preprocessing.loader import load_evo200_csv
from src.preprocessing.normalize import normalize_minmax
from src.preprocessing.windowing import create_cnn1d_dataset
from src.soc_inference.inference import SocInference
from src.range_estimator.estimator import estimate_range

# Load and preprocess
df = load_evo200_csv('data/raw/Evo200_Mixed1.csv')
features_norm = normalize_minmax(df[['pack_voltage_v', 'pack_current_a', 'temp_c', 'speed_kmh']])
df_norm = features_norm.copy()
df_norm['soc_bms'] = df['soc_bms'].values

# Create window
X, y = create_cnn1d_dataset(df_norm, window_size=60, step=1)

# Predict
soc_model = SocInference('models/soc_cnn1d.tflite')
soc_cnn1d, soh = soc_model.predict(X[0])

# Estimate range
range_km = estimate_range(
    soc_pct=soc_cnn1d,
    soh_pct=soh,
    wh_per_km_ewma=10.5,
    behavior_factor=0.95,
    pack_capacity_wh=2206.8
)
```
