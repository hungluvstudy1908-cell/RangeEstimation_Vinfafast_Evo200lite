# Development Guide

## Project Overview

VinFast Evo 200 SoC/Range monitoring system:
- **Real-time CAN data** → 3 parallel SoC estimators
- **CNN1D deep learning** model for SoC prediction
- **Range estimation** using behavior features
- **Web dashboard** with live visualization
- **Raspberry Pi 4** deployment ready

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                  Vehicle (CAN Bus)                   │
│      Pack Voltage, Current, Temp, Cell Voltages      │
└──────────────────────┬──────────────────────────────┘
                       │ USB-CAN Adapter
                       ↓
┌──────────────────────────────────────────────────────┐
│                  Raspberry Pi 4                       │
│  ┌────────────────────────────────────────────────┐  │
│  │           Main Loop (10Hz)                      │  │
│  │  ├─ CAN Reader (read frames)                   │  │
│  │  ├─ Decoder (convert to signals)               │  │
│  │  ├─ SoC #1: BMS pass-through                   │  │
│  │  ├─ SoC #2: Coulomb counting                   │  │
│  │  └─ Ring buffer (60 samples)                   │  │
│  │                                                 │  │
│  │  1Hz tick: Resample → Normalize → CNN1D        │  │
│  │  └─ SoC #3: Model prediction                   │  │
│  │  └─ Range estimation → CSV log                 │  │
│  └────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────┐  │
│  │       Flask Web Server (port 8080)             │  │
│  │  GET / → Dashboard with 3 battery icons        │  │
│  │  GET /api/state → JSON for AJAX polling        │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
                       │
                       ↓
                  Browser/Mobile
```

### Directory Structure

```
SourceCode/
├── src/                          # Runtime code
│   ├── preprocessing/            # Data loading & normalization
│   │   ├── loader.py            # CSV loader, sign conventions
│   │   ├── normalize.py         # Minmax/zscore normalization
│   │   └── windowing.py         # Sliding window dataset
│   ├── can_reader/              # CAN communication
│   │   ├── reader.py            # USB-CAN interface (Waveshare)
│   │   └── decoder.py           # CAN frame→signal conversion
│   ├── coulomb_counter/         # Coulomb counting SoC #2
│   │   └── counter.py           # CoulombCounter class
│   ├── soc_inference/           # CNN1D inference (SoC #3)
│   │   └── inference.py         # TFLite model wrapper
│   ├── range_estimator/         # Range prediction
│   │   └── estimator.py         # EWMA + behavior features
│   ├── display/                 # Web dashboard
│   │   ├── web.py               # Flask server
│   │   └── templates/
│   │       └── index.html       # 3-icon dashboard
│   ├── logger/                  # CSV runtime logging
│   │   └── writer.py            # RuntimeLogger class
│   └── main.py                  # 10Hz main loop entry point
│
├── configs/                      # YAML configuration
│   ├── battery_specs.yaml       # Pack capacity, voltage bounds
│   ├── can_ids.yaml             # CAN frame mappings
│   ├── model.yaml               # Model path, features
│   ├── range_estimator.yaml     # EWMA α, coefficients
│   └── preprocessing.yaml       # Normalization bounds
│
├── notebooks/                    # Training & analysis
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_train_cnn1d.ipynb     # Model training
│   └── 03_export_tflite.ipynb   # TFLite conversion
│
├── tests/                        # Unit & integration tests
│   ├── test_preprocessing.py    # Sign convention, normalization
│   ├── test_coulomb_counter.py  # CC update, bounds
│   ├── test_soc_inference.py    # TFLite inference
│   ├── test_logger.py           # CSV writing
│   ├── test_display.py          # Flask endpoints
│   └── test_integration.py      # End-to-end pipeline
│
├── data/                         # Dataset directories
│   ├── raw/                     # Evo200_*.csv training data
│   └── processed/               # runtime_YYYY-MM-DD_*.csv logs
│
├── models/                       # Trained models
│   ├── soc_cnn1d.pt            # PyTorch weights
│   └── soc_cnn1d.tflite        # TFLite for Pi runtime
│
├── deployment/                   # Pi 4 deployment
│   ├── setup-pi.sh              # Installation script
│   ├── soc-monitor.service      # systemd service
│   └── README.md                # Deployment guide
│
├── docs/                         # Documentation
│   ├── INSTALLATION.md          # Setup guide
│   ├── API.md                   # Module reference
│   ├── DEVELOPMENT.md           # This file
│   └── PERFORMANCE.md           # Latency/memory benchmarks
│
├── agent_docs/                   # Detailed technical docs (created by Claude)
│   ├── service_architecture.md  # 10Hz loop, 3 SoC design
│   ├── database_schema.md       # CSV schema, sign conventions
│   ├── code_conventions.md      # Naming, docstrings
│   ├── debugging_notes.md       # Known issues, fixes
│   ├── running_tests.md         # Test execution
│   └── building_the_project.md  # Build/run commands
│
└── README.md                     # Project overview
```

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone and enter project
git clone <repo> && cd SourceCode

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt
pip install pytest pytest-cov matplotlib
```

### 2. Add New Feature

#### Example: Add SoC filter

**Step 1: Create module**

```python
# src/filters/kalman.py
class KalmanFilter:
    """Kalman filter for SoC smoothing."""
    def __init__(self, q=0.01, r=0.1):
        self.q = q
        self.r = r
    
    def update(self, measurement):
        # Implementation
        return filtered_value
```

**Step 2: Add tests**

```python
# tests/test_kalman.py
def test_kalman_convergence():
    kf = KalmanFilter(q=0.01, r=0.1)
    # Test convergence property
    pass
```

**Step 3: Integrate into main loop**

```python
# src/main.py
from src.filters.kalman import KalmanFilter

# In MainLoop.__init__:
self.soc_filter = KalmanFilter()

# In main_loop():
soc_filtered = self.soc_filter.update(soc_cnn1d)
```

**Step 4: Update config**

```yaml
# configs/filtering.yaml
kalman:
  q: 0.01  # Process noise
  r: 0.1   # Measurement noise
```

**Step 5: Test & commit**

```bash
pytest tests/test_kalman.py -v
git add src/filters/ tests/ configs/filtering.yaml
git commit -m "feat: Add Kalman filter for SoC smoothing"
```

### 3. Code Conventions

#### Naming

- **Constants**: `UPPER_CASE` with units in comment
  ```python
  PACK_CAPACITY_AH = 30.65  # Evo 200 LFP nominal capacity
  WINDOW_SIZE_SAMPLES = 60  # 6 seconds at 10Hz
  ```

- **Variables**: `snake_case` with clear names
  ```python
  pack_voltage_v = 75.2
  coulomb_bias_pct = 0.35
  ```

- **Functions**: `snake_case`, verb-first
  ```python
  def update_coulomb_counter(current_a, dt):
      pass
  
  def compute_behavior_features(speed_array, current_array):
      pass
  ```

#### Docstrings

```python
def estimate_range(soc_pct, soh_pct, ewma, factor, pack_wh):
    """Estimate remaining range from SoC and consumption.
    
    Args:
        soc_pct: State of charge (%)
        soh_pct: State of health (%)
        ewma: EWMA consumption baseline (Wh/km)
        factor: Behavior factor (1.0 = baseline)
        pack_wh: Pack energy capacity (Wh)
    
    Returns:
        float: Estimated range (km)
    """
    # Formula: range_km = (SoC/100 * Energy * SoH/100) / (EWMA * factor)
    return (soc_pct / 100.0 * pack_wh * soh_pct / 100.0) / (ewma * factor)
```

#### Type Hints

```python
import numpy as np
from typing import Tuple, Dict, Optional

def load_evo200_csv(filepath: str) -> pd.DataFrame:
    """Load and parse Evo200 CSV file."""
    pass

def predict(self, window: np.ndarray) -> Tuple[float, float]:
    """Predict (SoC%, SoH%) from normalized window."""
    pass
```

### 4. Testing

#### Run All Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

#### Run Specific Test

```bash
pytest tests/test_coulomb_counter.py::test_bounds_upper -v
```

#### Write Test

```python
# tests/test_my_feature.py
import pytest
from src.my_module import my_function

def test_basic_case():
    """Test basic functionality."""
    result = my_function(input_value=42)
    assert result == expected_value

def test_edge_case():
    """Test boundary conditions."""
    with pytest.raises(ValueError):
        my_function(invalid_input=-100)

def test_performance():
    """Test performance constraint."""
    import time
    start = time.time()
    result = my_function(large_input)
    elapsed = time.time() - start
    assert elapsed < 0.1  # < 100ms
```

### 5. Debugging

#### Enable Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"SoC: {soc_cc:.1f}%")
logger.info(f"Range estimate: {range_km:.0f} km")
logger.warning(f"CAN buffer underrun")
logger.error(f"Model inference failed: {e}")
```

#### Inspect CAN Data

```bash
# Run CAN monitor
python3 -c "
from src.can_reader.reader import WaveshareReader
reader = WaveshareReader('/dev/ttyUSB0')
reader.connect()
for _ in range(10):
    can_id, data = reader.read_frame()
    print(f'0x{can_id:03x}: {data.hex()}')"
```

#### Inspect CSV Data

```python
import pandas as pd
df = pd.read_csv('data/processed/runtime_2026-06-04_103015.csv')
print(df.head(20))
print(df.describe())
df.plot(y=['soc_bms', 'soc_cc', 'soc_cnn1d'], subplots=True)
```

### 6. Performance Profiling

#### CPU Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
model.predict(window)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

#### Memory Profiling

```python
from memory_profiler import profile

@profile
def main_loop():
    # Code to profile
    pass

# Run: python -m memory_profiler my_script.py
```

## Git Workflow

### Commit Messages

```
feat: Add Kalman filter for SoC smoothing
fix: Correct sign convention in Coulomb counter
refactor: Extract CAN decoder into separate module
test: Add integration tests for range estimator
docs: Update API documentation for filters
chore: Update requirements.txt with new dependencies
```

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/kalman-filter

# Make commits
git commit -m "feat: Add KalmanFilter class"
git commit -m "test: Add KalmanFilter unit tests"

# Push and create PR
git push origin feature/kalman-filter

# After review, merge
git checkout main
git merge --no-ff feature/kalman-filter
```

## Deployment Checklist

Before pushing to Pi:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code follows conventions: naming, docstrings, type hints
- [ ] No hardcoded paths (use config files)
- [ ] Configuration in YAML (not Python)
- [ ] TFLite model exists and is compatible
- [ ] Tested on dev machine with mock CAN
- [ ] Documented new features in docs/
- [ ] Commit message clearly describes changes

## Common Tasks

### Add New CAN Signal

1. Update `configs/can_ids.yaml` with frame ID
2. Add decoder function in `src/can_reader/decoder.py`
3. Test with `python3 -m src.can_reader.reader` (mock data)
4. Integrate in `src/main.py` ring buffer
5. Add config parameter in `configs/preprocessing.yaml`

### Retrain Model

1. Run `notebooks/02_train_cnn1d.ipynb` on latest data
2. Verify metrics (MAE < 0.02)
3. Export to TFLite in `notebooks/03_export_tflite.ipynb`
4. Copy `models/soc_cnn1d.tflite` to Pi
5. Restart systemd service: `sudo systemctl restart soc-monitor`

### Analyze Runtime Logs

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load latest log
df = pd.read_csv('data/processed/runtime_latest.csv', parse_dates=['timestamp'])

# Plot SoC evolution
fig, ax = plt.subplots()
ax.plot(df['timestamp'], df['soc_bms'], label='BMS')
ax.plot(df['timestamp'], df['soc_cc'], label='Coulomb')
ax.plot(df['timestamp'], df['soc_cnn1d'], label='CNN1D')
ax.legend()
plt.show()

# Compute statistics
print(f"SoC range: {df['soc_bms'].min():.1f}% - {df['soc_bms'].max():.1f}%")
print(f"Coulomb bias: {(df['soc_cc'] - df['soc_bms']).mean():.2f}%")
```

## Resources

- **CAN Protocol**: [python-can docs](https://python-can.readthedocs.io/)
- **TFLite**: [TensorFlow Lite guide](https://www.tensorflow.org/lite/guide)
- **Flask**: [Flask documentation](https://flask.palletsprojects.com/)
- **Raspberry Pi**: [Official docs](https://www.raspberrypi.com/documentation/)

## Support

For help:
1. Check `agent_docs/debugging_notes.md` for known issues
2. Review test files for usage examples
3. Search git history: `git log --grep="keyword"`
4. Check pull request discussions for context
