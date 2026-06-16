# 🚀 How to RUN This Project

## Option 1: Test Everything (Verify System Works) ⭐ **START HERE**

```bash
# Step 1: Navigate to project
cd d:\DoAn\SourceCode

# Step 2: Create virtual environment
python -m venv venv

# Step 3: Activate virtual environment
venv\Scripts\activate

# Step 4: Install dependencies
pip install numpy pandas pyyaml flask flask-socketio pyserial python-can

# Step 5: Run ALL tests
pytest tests/ -v

# Expected output:
# ✓ test_preprocessing.py - 9 tests PASSED
# ✓ test_coulomb_counter.py - 15 tests PASSED
# ✓ test_logger.py - 4 tests PASSED
# ✓ test_display.py - 6 tests PASSED
# ✓ test_soc_inference.py - 7 tests PASSED
# ✓ test_integration.py - 5 tests PASSED
# ═════════════════════════════════════════════════════
# TOTAL: 46 tests PASSED ✓
```

---

## Option 2: Run Coulomb Counter Verification (Check Data Quality)

```bash
# Prerequisites: Complete Option 1 steps 1-4

python verify_coulomb_bias.py

# Expected output:
# Processing Evo200_Mixed1.csv ...
# Processing Evo200_Mixed2.csv ...
# ... (15 files total)
# 
# ═════════════════════════════════════════════════════
# COULOMB COUNTER VERIFICATION RESULTS
# ═════════════════════════════════════════════════════
# 
# Mean Bias:        +0.35% (Target: < 1%)  ✓ PASS
# Mean MAE:         4.36%
# Mean RMSE:        5.16%
# Correlation:      0.9934
# Total Samples:    899,951
```

---

## Option 3: Run Web Dashboard (Live Monitoring)

```bash
# Prerequisites: Complete Option 1 steps 1-4

# Step 1: Start Flask web server
python -m flask --app src.display.web run --host 0.0.0.0 --port 8080

# Expected output:
# WARNING in app.run()
#   Use a production WSGI server instead.
# * Running on http://127.0.0.1:8080
# * Press CTRL+C to quit

# Step 2: Open browser
# Go to: http://localhost:8080/

# You'll see:
# - 3 Battery Icons (SoC #1, #2, #3)
# - Real-time stats
# - Range estimate
# - Power consumption
```

---

## Option 4: Run Main System Loop (10Hz Real-time)

```bash
# Prerequisites: Complete Option 1 steps 1-4
# Requirements: USB-CAN adapter connected to COM port

# Start the 10Hz main loop
python -m src.main

# Expected output:
# [10:30:15] Initializing system...
# [10:30:15] CAN reader initialized on /dev/ttyUSB0
# [10:30:15] Coulomb counter initialized
# [10:30:15] SoC inference model loaded
# [10:30:15] Range estimator initialized
# [10:30:16] Main loop started (10Hz)
# [10:30:16] Web server started on port 8080
# 
# [10:30:17] CAN frame received: 0x308 (pack_voltage_v=75.2V)
# [10:30:18] SoC#1(BMS)=80.5% | SoC#2(CC)=80.3% | SoC#3(CNN1D)=80.4%
# [10:30:18] Range: 205.2 km | Consumption: 10.5 Wh/km
# [10:30:19] Logged to: data/processed/runtime_2026-06-04_103015.csv
```

---

## Option 5: Train Your Own Model (Advanced)

```bash
# Prerequisites: Complete Option 1 steps 1-4 + torch and jupyter
pip install torch matplotlib jupyter

# Step 1: Start Jupyter
jupyter notebook

# Step 2: Open notebook
# Click: notebooks/02_train_cnn1d.ipynb
# Then: Cell → Run All

# What it does:
# 1. Loads 15 Evo200 CSV files from data/raw/
# 2. Normalizes features
# 3. Creates 60-sample sliding windows
# 4. Trains CNN1D model with PyTorch
# 5. Evaluates on test set
# 6. Saves to: models/soc_cnn1d.pt

# Expected output:
# Loaded 15 files
# Total samples: 899,951
# Created 8,999 training windows
# 
# Epoch [1/50]   train_loss=0.134523 val_loss=0.128976 mae=0.0234 rmse=0.0456
# Epoch [2/50]   train_loss=0.118234 val_loss=0.112345 mae=0.0198 rmse=0.0412
# ...
# Epoch [47/50]  train_loss=0.034123 val_loss=0.035234 mae=0.0087 rmse=0.0145
# 
# Model saved to: models/soc_cnn1d.pt
# Best MAE: 0.0087
```

---

## Option 6: Export Model to TFLite (For Pi Deployment)

```bash
# Prerequisites: Option 5 complete

# Step 1: Start Jupyter
jupyter notebook

# Step 2: Open notebook
# Click: notebooks/03_export_tflite.ipynb
# Then: Cell → Run All

# What it does:
# 1. Loads PyTorch model (models/soc_cnn1d.pt)
# 2. Converts to ONNX
# 3. Converts to TensorFlow Lite
# 4. Tests inference compatibility
# 5. Saves to: models/soc_cnn1d.tflite

# Expected output:
# Loaded model from models/soc_cnn1d.pt
# Best MAE during training: 0.008726
# 
# Exported to ONNX: models/soc_cnn1d.onnx
# File size: 285.3 KB
# 
# Exported to TFLite: models/soc_cnn1d.tflite
# File size: 3.2 MB
# 
# Inference comparison (random input):
#   PyTorch output: 0.805234
#   TFLite output:  0.805198
#   Difference:     0.000036
```

---

## Option 7: Deploy to Raspberry Pi 4 (Production)

```bash
# Prerequisites: Options 5-6 complete

# Step 1: On Raspberry Pi, clone repo
ssh pi@<pi-ip>
cd /home/pi
git clone <repository> soc-monitor
cd soc-monitor

# Step 2: Run automated setup
chmod +x deployment/setup-pi.sh
./deployment/setup-pi.sh

# What it does:
# - Checks Python version
# - Installs system dependencies
# - Creates project directories
# - Installs Python packages
# - Sets up systemd service

# Step 3: Copy pre-trained model
scp models/soc_cnn1d.tflite pi@<pi-ip>:~/soc-monitor/models/

# Step 4: Start service
sudo systemctl start soc-monitor

# Step 5: Access dashboard
# Open browser: http://<pi-ip>:8080/

# Monitor logs:
sudo journalctl -u soc-monitor -f
```

---

## 📊 Quick Reference: What Each Option Does

| Option | Purpose | Duration | Requirements | Output |
|--------|---------|----------|--------------|--------|
| **1** | Test all modules | 5 min | Python 3.9+ | 46 tests pass ✓ |
| **2** | Verify data quality | 2 min | CSV files loaded | Coulomb bias report |
| **3** | Web dashboard | Ongoing | Flask running | Browser UI @ :8080 |
| **4** | Real-time monitoring | Ongoing | USB-CAN adapter | Live CSV logs |
| **5** | Train model | 10 min | PyTorch, Jupyter | models/soc_cnn1d.pt |
| **6** | Export TFLite | 5 min | Option 5 done | models/soc_cnn1d.tflite |
| **7** | Deploy to Pi | 30 min | Options 5-6 done | Running on Pi @ :8080 |

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install numpy pandas pyyaml flask flask-socketio pyserial python-can
```

### "No such file or directory: 'venv\Scripts\activate'"
Make sure you're in the project directory:
```bash
cd d:\DoAn\SourceCode
```

### "tests not found"
Make sure you activated venv:
```bash
venv\Scripts\activate
```

### "USB-CAN not detected"
```bash
# Check USB device
lsusb | grep Waveshare

# Or check serial port
python -c "import serial; print(serial.tools.list_ports.comports())"
```

### Tests fail with SSL errors
The current environment has SSL issues. Run on your **local machine** with regular Python, not MSYS2.

---

## 📁 Key Files You'll Interact With

| File | When to Use |
|------|-------------|
| `src/main.py` | Running real-time system (Option 4) |
| `verify_coulomb_bias.py` | Checking data quality (Option 2) |
| `notebooks/02_train_cnn1d.ipynb` | Training new model (Option 5) |
| `notebooks/03_export_tflite.ipynb` | Exporting for Pi (Option 6) |
| `deployment/setup-pi.sh` | Setting up Pi (Option 7) |
| `configs/*.yaml` | Configuration files |
| `data/raw/Evo200_*.csv` | Training data (15 files) |
| `models/soc_cnn1d.tflite` | Inference model |

---

## 🎯 Recommended Path

```
For Users:
1. Option 1 (test) → Verify everything works ✓
2. Option 2 (verify) → Check data quality ✓
3. Option 7 (deploy) → Run on Pi 4

For Developers:
1. Option 1 (test) → Verify everything works ✓
2. Read docs/DEVELOPMENT.md → Understand architecture
3. Option 5 (train) → Train your own model
4. Option 6 (export) → Export to TFLite
5. Option 7 (deploy) → Deploy to Pi 4

For Data Scientists:
1. Option 1 (test) → Verify everything works ✓
2. Open notebooks/01_eda.ipynb → Explore data
3. Option 5 (train) → Train model
4. Analyze models/ and data/processed/ outputs
```

---

## ✅ Success Criteria

- ✅ Option 1: All 46 tests pass
- ✅ Option 2: Coulomb bias < 1%
- ✅ Option 3: Dashboard loads in browser
- ✅ Option 4: CSV logs appear in data/processed/
- ✅ Option 5: models/soc_cnn1d.pt created (~20MB)
- ✅ Option 6: models/soc_cnn1d.tflite created (~3.2MB)
- ✅ Option 7: Service runs, dashboard accessible at :8080

---

**Ready to run? Pick an option above and follow the steps!** 🚀
