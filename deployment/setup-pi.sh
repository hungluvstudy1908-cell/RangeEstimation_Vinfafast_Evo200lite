#!/bin/bash
# Setup VinFast Evo 200 SoC Monitor on Raspberry Pi 4

set -e

PROJECT_DIR="/home/pi/soc-monitor"
SERVICE_NAME="soc-monitor"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "=== VinFast Evo 200 SoC Monitor — Pi 4 Setup ==="
echo

# 1. Check Python version
echo "[1/8] Checking Python..."
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"
if [[ ! "$PYTHON_VERSION" > "3.8" ]]; then
    echo "  ERROR: Python 3.9+ required"
    exit 1
fi
echo "  ✓ OK"
echo

# 2. Install system dependencies
echo "[2/8] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libatlas-base-dev \
    libjasper-dev \
    libtiff5 \
    libjasper1 \
    libharfbuzz0b \
    libwebp6 \
    libtiff5 \
    libjasper1 \
    libhyphen0 \
    libopenjp2-7 \
    libharfbuzz0b \
    libwebp6
echo "  ✓ OK"
echo

# 3. Create project directory
echo "[3/8] Setting up project directory..."
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR
echo "  ✓ Created $PROJECT_DIR"
echo

# 4. Install Python dependencies
echo "[4/8] Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    numpy \
    pandas \
    pyyaml \
    flask \
    flask-socketio \
    pyserial \
    tflite-runtime
echo "  ✓ OK"
echo

# 5. Install CAN driver (if Waveshare USB-CAN)
echo "[5/8] Setting up CAN interface..."
pip3 install python-can
echo "  Note: USB-CAN adapter should auto-enumerate as /dev/ttyUSB0"
echo "  ✓ OK"
echo

# 6. Create directories
echo "[6/8] Creating data directories..."
mkdir -p $PROJECT_DIR/data/raw
mkdir -p $PROJECT_DIR/data/processed
mkdir -p $PROJECT_DIR/models
mkdir -p $PROJECT_DIR/logs
echo "  ✓ OK"
echo

# 7. Install systemd service
echo "[7/8] Installing systemd service..."
sudo cp deployment/soc-monitor.service $SERVICE_FILE
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
echo "  ✓ Service installed"
echo

# 8. Download pre-trained model
echo "[8/8] Model status check..."
if [ ! -f "$PROJECT_DIR/models/soc_cnn1d.tflite" ]; then
    echo "  WARNING: models/soc_cnn1d.tflite not found"
    echo "  Run Task 24 (export-tflite.ipynb) to generate it"
else
    echo "  ✓ Model found"
fi
echo

echo "=== Setup Complete ==="
echo
echo "Next steps:"
echo "  1. Copy pre-trained model to $PROJECT_DIR/models/soc_cnn1d.tflite"
echo "  2. Connect USB-CAN adapter to Pi"
echo "  3. Start service:"
echo "     sudo systemctl start soc-monitor"
echo "  4. Check status:"
echo "     sudo systemctl status soc-monitor"
echo "  5. View logs:"
echo "     sudo journalctl -u soc-monitor -f"
echo "  6. Access web dashboard:"
echo "     http://<pi-ip>:8080/"
echo
