# Raspberry Pi Setup Guide

Complete guide for setting up OpenFlight on a Raspberry Pi 5 with the 7" touchscreen display.

## Prerequisites

Make sure you have all the hardware. See the **[Parts List](PARTS.md)** for what to buy.

**Required:**
- Raspberry Pi 5 (4GB+ recommended)
- 7" Touchscreen Display
- MicroSD Card (32GB+)
- 27W USB-C Power Supply (official Pi 5 PSU recommended)
- OPS243-A Doppler Radar + USB cable
- SparkFun SEN-14262 sound detector (wired per the [Sound Trigger Wiring Guide](sound-trigger-wiring.md))

**Optional:**
- K-LD7 + FTDI adapter (×2) — for launch angle and club path (see [Parts List](PARTS.md))

## Initial Setup

### 1. Install Raspberry Pi OS

Use Raspberry Pi Imager to flash **Raspberry Pi OS (64-bit)** to your SD card.

### 2. Clone and Install

```bash
cd ~
git clone https://github.com/jewbetcha/openflight.git
cd openflight

# Run the setup script (handles everything)
./scripts/setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all Python dependencies
- Install Node.js dependencies
- Build the UI
- Run tests to verify installation

Or manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
python -m venv .venv --system-site-packages
source .venv/bin/activate
uv pip install -e ".[ui]"
cd ui && npm install && npm run build && cd ..
```

## Radar Setup (One-Time)

The OPS243-A needs a one-time configuration to enable rolling buffer mode with hardware sound triggering. This saves settings to flash memory so it boots in the correct mode every time.

> **Why?** The OPS243-A has a firmware bug where the HOST_INT pin mode switches unexpectedly when entering rolling buffer mode at runtime. Saving to flash and power cycling bypasses this. Confirmed by OmniPreSense engineering.

### 1. Configure and Save

```bash
uv run python scripts/test_rolling_buffer_persist.py --setup
```

### 2. Power Cycle

Unplug the radar's USB cable, wait 3 seconds, plug it back in.

### 3. Verify

```bash
uv run python scripts/test_rolling_buffer_persist.py --test
```

Make a sound near the SEN-14262 — you should see trigger data with I/Q samples.

## K-LD7 Angle Radar Setup

Each K-LD7 connects via a 3.3V FTDI USB-to-serial adapter and appears as `/dev/ttyUSB*`.

### Stable Device Names (udev rules)

USB serial devices can swap between `/dev/ttyUSB0` and `/dev/ttyUSB1` after a reboot depending on enumeration order. To assign fixed names based on each FTDI adapter's unique serial number:

```bash
# Find the serial numbers for each adapter
udevadm info -a /dev/ttyUSB0 | grep '{serial}' | head -1
udevadm info -a /dev/ttyUSB1 | grep '{serial}' | head -1
```

Create a udev rule with the serial numbers:

```bash
sudo tee /etc/udev/rules.d/99-kld7.rules << 'EOF'
SUBSYSTEM=="tty", ATTRS{serial}=="FTXXXXXX", SYMLINK+="kld7_vertical"
SUBSYSTEM=="tty", ATTRS{serial}=="FTYYYYYY", SYMLINK+="kld7_horizontal"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
```

Replace `FTXXXXXX` and `FTYYYYYY` with the actual serial numbers. Now the radars are always at `/dev/kld7_vertical` and `/dev/kld7_horizontal` regardless of plug order.

### Mounting

- **Vertical unit** — measures launch angle. Mount with the antenna plane vertical, aimed at the hitting area.
- **Horizontal unit** — measures club path / aim direction. Mount with the antenna plane horizontal.

Both should be positioned near the OPS243-A, 3-5 feet behind the tee.

### Angle Offset Calibration

The raw RADC angle needs an offset to match real launch angles. The offset depends on your mounting geometry (sensor height, angle, distance from ball).

1. Start a session with `--kld7 --kld7-angle-offset 8`
2. Hit 5-10 shots with a known club (7-iron recommended)
3. Compare reported launch angles to expected values:
   - Wedge: 24-30°, 7-iron: 16-18°, 5-iron: 12-14°, Driver: 10-14°
4. Adjust the offset: if angles read 5° too low, increase offset by 5

Typical offsets are 5-10°. The exact value depends on your mounting position.

See [K-LD7 Troubleshooting](kld7-troubleshooting.md) for more details.

## Running OpenFlight

### Kiosk Mode (Fullscreen — Recommended)

```bash
# Default: rolling buffer + sound trigger
./scripts/start-kiosk.sh

# With K-LD7 angle radar
./scripts/start-kiosk.sh --kld7 --kld7-angle-offset 8

# Mock mode (no hardware needed)
./scripts/start-kiosk.sh --mock
```

### Manual Start

```bash
openflight-server                # With radar
openflight-server --mock         # No hardware
```

Then open `http://localhost:8080`.

### Running Over SSH

```bash
DISPLAY=:0 ./scripts/start-kiosk.sh
```

## Auto-Start on Boot

### Enable the Service

```bash
sudo cp ~/openflight/scripts/openflight.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable openflight
sudo systemctl start openflight
```

### Service Management

```bash
sudo systemctl status openflight --no-pager   # Check status
journalctl -u openflight -f                    # View logs
sudo systemctl stop openflight                 # Stop
sudo systemctl restart openflight              # Restart
sudo systemctl disable openflight              # Disable auto-start
```

To modify the service:
```bash
sudo nano /etc/systemd/system/openflight.service
sudo systemctl daemon-reload
sudo systemctl restart openflight
```

## Observability (Grafana Cloud)

OpenFlight can ship session logs to Grafana Cloud for long-term analysis.

```bash
sudo ./scripts/setup_alloy.sh
sudo vim /etc/alloy/credentials.env
```

See [observability.md](observability.md) for full setup and LogQL queries.

## Troubleshooting

### Radar Not Detected

```bash
ls /dev/ttyACM* /dev/ttyUSB*
openflight --port /dev/ttyACM0 --info
```

### Sound Trigger Not Working

See the [Sound Trigger Wiring Guide — Troubleshooting](sound-trigger-wiring.md#troubleshooting).

### K-LD7 Not Connecting

```bash
# Check USB devices
ls /dev/ttyUSB* /dev/kld7_*

# Test standalone
uv run python scripts/test_kld7.py
```

Look for `[KLD7] Connected on /dev/ttyUSB...` in the server logs. See [K-LD7 Troubleshooting](kld7-troubleshooting.md) for "Wrong length reply" and other connection issues.

### Service Won't Start

```bash
journalctl -u openflight --no-pager -n 50

# If service is masked
sudo systemctl unmask openflight
sudo cp ~/openflight/scripts/openflight.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable openflight
```

### Slow UI Updates

Check for WebSocket instability:
```bash
journalctl -u openflight -f
```

Look for "Client disconnected/connected" messages.

### Display Issues Over SSH

Use `DISPLAY=:0` prefix for commands that need the Pi's display.

## CLI Reference

### Kiosk

```bash
./scripts/start-kiosk.sh                                      # Default
./scripts/start-kiosk.sh --mock                                # No hardware
./scripts/start-kiosk.sh --kld7 --kld7-angle-offset 8          # With angle radar
./scripts/start-kiosk.sh --port 3000                           # Custom port
```

### Server

```bash
openflight-server                    # Start with radar
openflight-server --mock             # Mock mode
openflight-server --web-port 3000    # Custom port
```

### Testing

```bash
uv run python scripts/test_rolling_buffer_persist.py --test    # Sound trigger
uv run python scripts/test_sound_trigger_hardware.py           # Direct trigger test
uv run python scripts/test_kld7.py                             # K-LD7 standalone
uv run pytest tests/ -v                                        # Full test suite
```
