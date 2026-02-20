# Mock Hardware Testing Guide

## Overview

The mock Arduino controller allows you to test the entire streaming robot control pipeline **without physical hardware**. It simulates realistic sensor behavior based on motor commands.

## Features

✓ **Realistic Sensor Simulation** — Pressure and EMG respond to simulated motor movements  
✓ **No Hardware Required** — Test anywhere with just a webcam  
✓ **Full Pipeline Testing** — End-to-end vision → audio → sensors → fusion → control  
✓ **Command History Tracking** — Log all commands for analysis  
✓ **Configurable Noise** — Adjust sensor noise for robustness testing  

## Quick Start

### Run Unit Tests (Validate Components)

```bash
python tests/sanity/test_mock_arduino.py
```

Expected output:
```
RESULTS: 5 passed, 0 failed
✓ Basic operations
✓ Sensor response to motors
✓ Command history
✓ Factory function
✓ Simulation state export
```

### Run Mock Streaming Demo (Live Pipeline)

```bash
python demo_mock_streaming.py --duration 30 --fps 30
```

**Options:**
- `--duration N` — Run for N seconds (default: 30)
- `--fps N` — Target frames per second (default: 30, max 60)
- `--webcam ID` — Webcam device ID (default: 0)
- `--device cpu|cuda` — Compute device (default: auto-detect)
- `--save-video` — Save output to `demo_output.mp4`

### Integration in Production Code

Use the factory function to seamlessly switch between mock and real hardware:

```python
from hardware_io.mock_arduino_controller import create_arduino_controller

# Use mock hardware for testing
arduino = create_arduino_controller(mock_mode=True)

# Or use real hardware (requires port configuration)
# arduino = create_arduino_controller(mock_mode=False, port="/dev/ttyUSB0")

arduino.connect()
arduino.send_command(angle1=45, angle2=30, angle3=60, gripper_force=50)
sensors = arduino.read_sensors()
```

## Sensor Simulation Details

### Pressure Sensor
- **Baseline:** Randomly initialized 40-60 (ADC 0-1023)
- **Gripper effect:** Increases with gripper force (0-100%)
- **Position effect:** Responds to vertical arm position (motor 2)
- **Noise:** Gaussian with configurable std dev

### EMG Sensors (3 channels)
- **Baseline:** Randomly initialized 100-150 per channel
- **Motor response:** Each channel responds to corresponding motor activity
- **Temporal behavior:** Oscillating pattern simulating muscle activation
- **Noise:** Channel-specific Gaussian noise

### Response Model

```python
pressure = baseline_pressure + gripper_effect + position_effect + noise

emg[i] = baseline_emg[i] + motor_activity[i] * 100 + 
         oscillation[i] + noise[i]
```

## Mock vs Real Hardware

| Feature | Mock | Real |
|---------|------|------|
| No hardware required | ✓ | ✗ |
| Realistic sensor sim | ✓ | N/A |
| Arduino communication | Stub | Serial |
| Sensor noise | Configurable | Fixed |
| Testing speed | Fast | Normal |
| Deployment | ✗ | ✓ |

## API Reference

### Arduino Controller (recommended: use factory)

Use the factory `create_arduino_controller()` to select mock vs real hardware from configuration or explicitly via `mock_mode`.

```python
from hardware_io.mock_arduino_controller import create_arduino_controller

# Create based on config (config/streaming_config.yaml -> arduino.mode)
arduino = create_arduino_controller(noise_level=0.1)

# Or explicitly request mock or real
# Mock (testing):
arduino = create_arduino_controller(mock_mode=True, noise_level=0.1)

# Real (requires port):
# arduino = create_arduino_controller(mock_mode=False, port='/dev/ttyUSB0')

arduino.connect()

# Send motor command
arduino.send_command(
   angle1=45.0,      # Joint 1 angle (degrees)
   angle2=30.0,      # Joint 2 angle (degrees)
   angle3=60.0,      # Joint 3 angle (degrees)
   gripper_force=50.0 # Gripper force (0-100%)
)

# Read sensors
sensors = arduino.read_sensors()

# Get status / disconnect
status = arduino.get_status()
arduino.disconnect()
```

Notes:
- The underlying `MockArduinoController` class remains available for direct instantiation in unit tests or advanced scenarios, but using the factory centralizes configuration and makes demos/config-driven switches easier.

## Testing Workflow

### 1. Component Tests (Unit Level)
```bash
python tests/sanity/test_mock_arduino.py
```
Validates sensor simulation and command handling.

### 2. Pipeline Tests (Integration Level)
```bash
python tests/sanity/test_streaming_pipeline.py
```
Tests all encoders, fusion, and controller together.

### 3. Live Demo (System Level)
```bash
python demo_mock_streaming.py --duration 60 --fps 60
```
End-to-end demo with visual feedback.

## Transition to Real Hardware

When ready to use actual hardware:

1. **Update configuration:**
   ```yaml
   # config/streaming_config.yaml
   arduino:
     port: "/dev/ttyUSB0"  # or "COM3" on Windows
     baud_rate: 115200
   ```

2. **Switch in code:**
   ```python
   # Change from mock to real
   arduino = create_arduino_controller(mock_mode=False)
   ```

3. **Upload Arduino firmware** to Uno with motor/sensor code

4. **Run streaming pipeline:**
   ```bash
   python streaming_robot_control.py --config config/streaming_config.yaml
   ```

## Troubleshooting

### "ModuleNotFoundError: No module named 'io'"

The built-in Python `io` module shadows the custom one. Use:
```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'io'))
from mock_arduino_controller import MockArduinoController
```

### Sensor values out of range

Check noise level (default 0.1):
```python
arduino = MockArduinoController(noise_level=0.05)  # Lower noise
```

### Low FPS in demo

Reduce target FPS:
```bash
python demo_mock_streaming.py --fps 30
```

Or use CPU device if GPU is unavailable:
```bash
python demo_mock_streaming.py --device cpu
```

## Example Output

When running the mock demo, you'll see:

```
Frame: 1800 | FPS: 59.8
Robot Position (m):
  X=0.453  Y=0.687  Z=0.521
Gripper Force: 62.3%
Pressure: 78.5
EMG: 142.1, 156.3, 149.8
MOCK Arduino (simulated) | Commands: 1800
```

## Next Steps

Once testing is complete with mock hardware:

1. **Obtain Arduino pins** for your sensors
2. **Update config/streaming_config.yaml** with actual pin assignments
3. **Upload firmware** to Arduino Uno
4. **Connect hardware** via USB serial
5. **Run with mock_mode=False** for production deployment
