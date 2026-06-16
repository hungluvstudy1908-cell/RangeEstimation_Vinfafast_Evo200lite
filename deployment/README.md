# Deployment Guide — VinFast Evo 200 SoC Monitor on Raspberry Pi 4

## Overview

This directory contains deployment scripts and configuration for running the SoC/range monitoring system on Raspberry Pi 4 (8GB).

## Hardware Requirements

- **Raspberry Pi 4** (8GB RAM minimum)
- **Waveshare USB-CAN adapter** (2Mbps) connected via USB
- **MicroSD card** (32GB+)
- **Power supply** (5V 3A)

## Software Requirements

- **Raspberry Pi OS Lite** (32-bit, Python 3.9+)
- **systemd** (for service management)

## Installation Steps

### 1. Prepare Raspberry Pi

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Set hostname and enable SSH
sudo raspi-config

# Enable SPI/I2C if needed
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0
```

### 2. Clone Project

```bash
cd /home/pi
git clone <repository> soc-monitor
cd soc-monitor
```

### 3. Run Setup Script

```bash
chmod +x deployment/setup-pi.sh
./deployment/setup-pi.sh
```

This will:
- Check Python version (3.9+)
- Install system dependencies
- Create project directory structure
- Install Python packages
- Install systemd service

### 4. Add Pre-trained Model

Copy the TFLite model exported in Task 24:

```bash
cp ../notebooks/models/soc_cnn1d.tflite /home/pi/soc-monitor/models/
```

### 5. Connect CAN Adapter

Plug the Waveshare USB-CAN adapter into Pi's USB port. Check connection:

```bash
ls -la /dev/ttyUSB*
```

### 6. Start Service

```bash
sudo systemctl start soc-monitor
sudo systemctl status soc-monitor
```

### 7. Verify Logs

```bash
sudo journalctl -u soc-monitor -f
```

Expected output:
```
[10:30:15] CAN reader initialized
[10:30:16] Coulomb counter initialized
[10:30:17] Main loop started (10Hz)
[10:30:20] Web server listening on port 8080
```

### 8. Access Dashboard

Open browser and navigate to:
```
http://<pi-ip>:8080/
```

You should see 3 battery icons representing:
- **Green**: BMS SoC (from CAN)
- **Amber**: Coulomb Counter SoC
- **Red**: CNN1D model SoC

## Configuration

Edit config files in `/home/pi/soc-monitor/configs/`:

### battery_specs.yaml
```yaml
pack_capacity_ah: 30.65  # Evo 200 LFP capacity
nominal_voltage_v: 72.0
soh_full_percent: 100.0
```

### can_ids.yaml
```yaml
can_frames:
  frame_1: 0x308  # Cell voltages 1-4
  frame_2: 0x309  # Cell voltages 5-8
  # ...etc
```

### range_estimator.yaml
```yaml
ewma_alpha: 0.3           # Learning rate
pack_capacity_wh: 2206.8  # 72V × 30.65Ah
behavior_coefficients:
    intercept: 1.0
    speed_coeff: -0.01
    accel_coeff: 0.05
    stop_coeff: 0.1
```

## Monitoring

### Service Status
```bash
sudo systemctl status soc-monitor
```

### Real-time Logs
```bash
sudo journalctl -u soc-monitor -f
sudo tail -f /var/log/syslog | grep soc-monitor
```

### CSV Runtime Logs
```bash
ls -lh /home/pi/soc-monitor/data/processed/
```

## Troubleshooting

### USB-CAN not detected
```bash
# Check USB device
lsusb | grep "Waveshare"

# Reset USB port
echo 1 > /sys/bus/usb/devices/1-1/authorized
sleep 2
echo 1 > /sys/bus/usb/devices/1-1/authorized
```

### Service won't start
```bash
# Check systemd service file
sudo systemctl cat soc-monitor

# Test Python import
python3 -c "import src.main"

# Run directly for debugging
cd /home/pi/soc-monitor
python3 -m src.main
```

### High memory usage
Adjust Docker memory limits in service file:
```
MemoryLimit=300M
```

### Web dashboard not accessible
```bash
# Check Flask binding
curl -v http://localhost:8080/

# Check firewall
sudo ufw allow 8080
```

## Performance Notes

- **10Hz main loop**: Update cycle every 100ms
- **1Hz inference**: CNN1D prediction every 1 second (on separate thread)
- **Memory footprint**: ~180MB resident
- **CPU usage**: 20-30% on single core
- **Latency**: <50ms CAN→display

## Uninstall

```bash
# Stop service
sudo systemctl stop soc-monitor
sudo systemctl disable soc-monitor

# Remove service
sudo rm /etc/systemd/system/soc-monitor.service
sudo systemctl daemon-reload

# Remove project (optional)
rm -rf /home/pi/soc-monitor
```

## Safety Notes

⚠️ **Important:**
- Never disconnect USB-CAN while system is running
- Monitor battery temperature in configs/battery_specs.yaml
- Log files may grow to 1GB/week — set up log rotation if needed
- Always test on bench before vehicle deployment

## Support

For issues, check:
1. `docs/PERFORMANCE.md` — latency/memory benchmarks
2. `agent_docs/debugging_notes.md` — known issues and fixes
3. Service logs: `journalctl -u soc-monitor -p err -f`
