# VinFast Evo 200 SoC Project — Local Setup Guide

## Quick Start (Windows/macOS/Linux)

### Step 1: Clone/Download Project
```bash
cd your-projects-folder
git clone <repository> DoAn-SourceCode
cd DoAn-SourceCode
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install All Dependencies
```bash
pip install --upgrade pip
pip install numpy pandas pyyaml flask flask-socketio pyserial python-can torch matplotlib pytest

# For Jupyter notebooks (optional):
pip install jupyter notebook
```

### Step 4: Verify Installation
```bash
python -m pytest tests/ -v
```

Expected output:
```
test_preprocessing.py ✓ PASSED (9 tests)
test_coulomb_counter.py ✓ PASSED (15 tests)
test_logger.py ✓ PASSED (4 tests)
test_display.py ✓ PASSED (6 tests)
test_soc_inference.py ✓ PASSED (7 tests)
test_integration.py ✓ PASSED (5 tests)
═════════════════════════════════════════════
TOTAL: 46 PASSED ✓
```

### Step 5: Run Training Notebook
```bash
# Option A: Jupyter
jupyter notebook notebooks/02_train_cnn1d.ipynb

# Option B: Run directly (if you prefer Python)
python notebooks/02_train_cnn1d.ipynb
```

### Step 6: Export to TFLite (Optional - for Pi deployment)
```bash
jupyter notebook notebooks/03_export_tflite.ipynb
```

---

## Detailed Dependencies

### Core Runtime (Required)
```
numpy          ≥ 1.19    — Numerical computing
pandas         ≥ 1.1     — Data manipulation
pyyaml         ≥ 5.3     — Config file parsing
flask          ≥ 1.1     — Web server
flask-socketio ≥ 4.0     — Real-time updates
pyserial       ≥ 3.5     — Serial communication (CAN)
python-can     ≥ 3.1     — CAN bus interface
```

### Training (Optional)
```
torch           ≥ 1.7    — Deep learning framework
matplotlib      ≥ 3.3    — Plotting
jupyter         ≥ 1.0    — Notebook environment
```

### Testing (Optional)
```
pytest          ≥ 6.0    — Test runner
```

### Pi Deployment (Optional)
```
tflite-runtime  ≥ 2.5    — TFLite inference (Pi only)
```

---

## Installation Troubleshooting

### Issue: "No module named pip"
**Solution:** Ensure you're using a proper Python installation (not MSYS2). Download from python.org.

### Issue: SSL certificate error
**Solution:** Update certificates or use trusted hosts:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy pandas
```

### Issue: numpy/pandas installation fails
**Solution:** Install pre-built wheels:
```bash
pip install numpy pandas --only-binary :all:
```

Or use conda (easier):
```bash
conda create -n soc-env python=3.9
conda activate soc-env
conda install numpy pandas pyyaml flask flask-socketio pyserial python-can torch
```

### Issue: "requirements not met"
**Solution:** Upgrade pip, setuptools, wheel:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Verify Setup Works

Run these commands to verify each component:

### 1. Test Imports
```bash
python -c "from src.preprocessing.loader import load_evo200_csv; print('✓ Preprocessing OK')"
python -c "from src.coulomb_counter.counter import CoulombCounter; print('✓ Coulomb Counter OK')"
python -c "from src.range_estimator.estimator import estimate_range; print('✓ Range Estimator OK')"
python -c "from src.logger.writer import RuntimeLogger; print('✓ Logger OK')"
python -c "from src.display.web import create_app; print('✓ Display OK')"
```

### 2. Test CSV Loading
```bash
python verify_coulomb_bias.py
# Should output statistics for all 15 CSV files
```

### 3. Run Unit Tests
```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_coulomb_counter.py -v
pytest tests/test_integration.py -v
```

### 4. Test Web Server
```bash
python -c "
from src.display.web import create_app
from src.main import SharedState
state = SharedState()
app = create_app(state)
print('Web server created successfully!')
print('Run: python -m flask --app src.display.web run --host 0.0.0.0 --port 8080')
"
```

---

## Project Structure

```
DoAn-SourceCode/
├── src/                              # Runtime code
│   ├── preprocessing/                # Data loading & normalization
│   ├── can_reader/                   # CAN communication
│   ├── coulomb_counter/              # Coulomb counter SoC estimator
│   ├── soc_inference/                # CNN1D model inference
│   ├── range_estimator/              # Range prediction
│   ├── display/                      # Web dashboard
│   ├── logger/                       # CSV logging
│   └── main.py                       # Main entry point (10Hz loop)
│
├── notebooks/                        # Training & analysis
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_train_cnn1d.ipynb         # Model training
│   └── 03_export_tflite.ipynb       # TFLite conversion
│
├── tests/                            # Unit & integration tests
│   ├── test_preprocessing.py        # Sign convention, normalization tests
│   ├── test_coulomb_counter.py      # Coulomb counter tests
│   ├── test_soc_inference.py        # TFLite inference tests
│   ├── test_logger.py               # CSV logging tests
│   ├── test_display.py              # Web server tests
│   └── test_integration.py          # End-to-end system tests
│
├── configs/                          # YAML configuration
│   ├── battery_specs.yaml           # Pack capacity, voltage bounds
│   ├── can_ids.yaml                 # CAN frame mappings
│   ├── model.yaml                   # Model path & features
│   ├── range_estimator.yaml         # EWMA α, coefficients
│   └── preprocessing.yaml           # Normalization bounds
│
├── data/                             # Datasets
│   ├── raw/                         # Evo200_Mixed*.csv (15 training files)
│   └── processed/                   # Runtime logs (runtime_*.csv)
│
├── models/                           # Trained models
│   ├── soc_cnn1d.pt                # PyTorch weights
│   └── soc_cnn1d.tflite            # TFLite for Pi runtime
│
├── deployment/                       # Pi 4 deployment
│   ├── setup-pi.sh                  # Auto-installation script
│   ├── soc-monitor.service          # systemd service
│   └── README.md                    # Deployment guide
│
├── docs/                             # User documentation
│   ├── INSTALLATION.md              # Installation guide
│   ├── API.md                       # API reference
│   ├── DEVELOPMENT.md               # Developer guide
│   └── PERFORMANCE.md               # Performance benchmarks
│
├── agent_docs/                       # Technical documentation
│   ├── service_architecture.md      # System architecture
│   ├── database_schema.md           # CSV schemas
│   ├── debugging_notes.md           # Known issues
│   └── ...
│
├── verify_coulomb_bias.py           # Verification script
├── requirements.txt                 # Dependencies list
└── README.md                        # Project overview
```

---

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v --cov=src

# Run specific test
pytest tests/test_coulomb_counter.py::test_update_discharge -v

# Train model (interactive)
jupyter notebook notebooks/02_train_cnn1d.ipynb

# Verify Coulomb bias
python verify_coulomb_bias.py

# Check code style
python -m py_compile src/main.py tests/*.py

# Format code (if black installed)
black src/ tests/ --line-length 100
```

---

## Next Steps

1. **Setup complete?** → Run `pytest tests/ -v`
2. **Want to train?** → Open `notebooks/02_train_cnn1d.ipynb`
3. **Deploy to Pi?** → See `deployment/README.md`
4. **Need API reference?** → Check `docs/API.md`
5. **Debugging issues?** → See `agent_docs/debugging_notes.md`

---

## Support

- **API Documentation:** `docs/API.md`
- **Development Guide:** `docs/DEVELOPMENT.md`
- **Deployment Guide:** `deployment/README.md`
- **Performance Notes:** `docs/PERFORMANCE.md`
- **Known Issues:** `agent_docs/debugging_notes.md`

---

**Good luck! 🚀**
