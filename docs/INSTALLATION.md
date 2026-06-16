# Installation Guide

## Overview

This guide covers installation for both **development** (laptop/desktop) and **production** (Raspberry Pi 4) environments.

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- pip (Python package manager)

### Steps

#### 1. Clone Repository

```bash
git clone <repository-url>
cd SourceCode
```

#### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Download Dataset

Place Evo200 CSV files in `data/raw/`:
```bash
ls data/raw/Evo200_*.csv
```

Expected 15 files, ~2.2GB total.

#### 5. Download Pre-trained Model

If available, place in `models/`:
```bash
soc_cnn1d.pt       # PyTorch weights
soc_cnn1d.tflite   # TFLite for Pi runtime
```

#### 6. Verify Installation

```bash
# Test imports
python3 -c "from src.preprocessing.loader import load_evo200_csv; print('✓ OK')"

# Run tests
pytest tests/

# Run smoke tests
python3 tests/test_preprocessing.py
python3 tests/test_coulomb_counter.py
```

## Production Setup (Raspberry Pi 4)

### Requirements

- Raspberry Pi 4 (8GB RAM)
- Waveshare USB-CAN adapter
- MicroSD card (32GB+)
- Power supply (5V 3A)
- Raspberry Pi OS Lite (bullseye)

### Automated Setup

```bash
# SSH into Pi
ssh pi@<pi-ip>

# Clone project
cd /home/pi
git clone <repository-url> soc-monitor
cd soc-monitor

# Run setup script
chmod +x deployment/setup-pi.sh
./deployment/setup-pi.sh

# Start service
sudo systemctl start soc-monitor
sudo systemctl enable soc-monitor
```

See [deployment/README.md](../deployment/README.md) for detailed steps.

### Manual Setup (if needed)

```bash
# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install Python dependencies
sudo apt-get install python3-pip python3-dev

# 3. Install pip packages
pip3 install numpy pandas pyyaml flask flask-socketio tflite-runtime pyserial

# 4. Install CAN tools
pip3 install python-can

# 5. Create directories
mkdir -p ~/soc-monitor/{data/raw,data/processed,models,logs,configs}

# 6. Copy project files
# (Manually copy src/, configs/, etc.)

# 7. Install systemd service
sudo cp deployment/soc-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable soc-monitor

# 8. Start service
sudo systemctl start soc-monitor
```

## Dependency Notes

### Core Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| numpy | Numerical computing | ≥1.19 |
| pandas | Data manipulation | ≥1.1 |
| pyyaml | Configuration files | ≥5.3 |
| flask | Web server | ≥1.1 |
| pyserial | Serial communication | ≥3.5 |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| torch | Model training (dev only) |
| onnx | ONNX export (dev only) |
| tensorflow | TFLite conversion (dev only) |
| pytest | Testing |
| matplotlib | Plotting (notebooks) |

### Pi-Specific

| Package | Purpose |
|---------|---------|
| tflite-runtime | Lightweight inference |
| python-can | CAN bus interface |

## Hardware Setup

### USB-CAN Adapter

1. Connect Waveshare USB-CAN to Pi's USB port
2. Check detection: `lsusb | grep Waveshare`
3. Verify serial port: `ls -la /dev/ttyUSB*`
4. Test communication: See [docs/DEVELOPMENT.md](./DEVELOPMENT.md)

### Serial Connection (Optional)

For OBD-II display module (if used):
```bash
# Check serial port
ls -la /dev/ttyAMA0

# Enable UART
sudo raspi-config nonint do_serial 0
```

## Troubleshooting

### Import Errors

```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Verify src/ structure
ls -la src/preprocessing/
ls -la src/can_reader/
```

### Missing Dependencies

```bash
# Install specific package
pip install numpy==1.19.0

# Check installed versions
pip list | grep numpy
```

### Raspberry Pi Specific

```bash
# Check available memory
free -h

# Check disk space
df -h

# Monitor Pi temperature
vcgencmd measure_temp
```

## Uninstall

```bash
# Dev environment
deactivate
rm -rf venv/

# Pi production
sudo systemctl stop soc-monitor
sudo systemctl disable soc-monitor
sudo rm /etc/systemd/system/soc-monitor.service
rm -rf ~/soc-monitor
```

## Next Steps

- Read [docs/API.md](./API.md) for module reference
- Follow [docs/DEVELOPMENT.md](./DEVELOPMENT.md) for contributing
- See [deployment/README.md](../deployment/README.md) for production deployment

## Support

For issues:
1. Check logs: `journalctl -u soc-monitor -f` (Pi) or `pytest tests/ -v` (dev)
2. Verify config files in `configs/`
3. Test imports: `python3 -c "from src.main import main_loop"`
