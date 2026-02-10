# Quick Reference: Dual Camera Setup

## Common Commands

### 1. **Production Pipeline (Real Arduino)**

```bash
# Single camera (default)
python streaming_robot_control.py --config config/streaming_config.yaml --device cpu

# Dual cameras (cameras 0 and 1)
python streaming_robot_control.py --config config/streaming_config.yaml --device cpu --webcam 0 1

# Dual cameras (cameras 1 and 2)
python streaming_robot_control.py --config config/streaming_config.yaml --device cpu --webcam 1 2

# With GPU acceleration (if available)
python streaming_robot_control.py --config config/streaming_config.yaml --device cuda --webcam 0 1

# With custom duration
python streaming_robot_control.py --config config/streaming_config.yaml --device cpu --webcam 0 1
```

### 2. **Demo Mode (Mock Hardware - No Arduino Required)**

```bash
# Single camera, 30 seconds
python demo_mock_streaming.py --duration 30 --fps 30 --device cpu

# Dual cameras, 30 seconds
python demo_mock_streaming.py --duration 30 --fps 30 --device cpu --webcam 0 1

# Single camera, 60 seconds, save output video
python demo_mock_streaming.py --duration 60 --fps 30 --device cpu --save-video

# Dual cameras, continuous (until 'q' pressed)
python demo_mock_streaming.py --duration 9999 --fps 30 --device cpu --webcam 0 1
```

## Finding Your Camera IDs

### Linux / WSL
```bash
# Method 1: List video devices
ls -la /dev/video*

# Method 2: Using ffmpeg
ffmpeg -f v4l2 -list_formats all -i /dev/video0
```

### Python Script
```python
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"✓ Camera {i}: {frame.shape}")
        cap.release()
    else:
        print(f"✗ Camera {i}: Not available")
```

## Typical Setup Scenarios

### Scenario A: Two USB Cameras
```bash
# Identify cameras
$ ls /dev/video*
/dev/video0  /dev/video1  /dev/video2  /dev/video3

# Test them
$ python -c "
import cv2
for i in [0,1,2,3]:
    cap = cv2.VideoCapture(i)
    print(f'{i}: {\"OK\" if cap.isOpened() else \"No\"}')"

# Output might be:
# 0: OK
# 1: OK
# 2: No
# 3: No

# Use cameras 0 and 1
python streaming_robot_control.py --config config/streaming_config.yaml --device cpu --webcam 0 1
```

### Scenario B: Laptop Webcam + USB Camera
```bash
# Laptop webcam is typically camera 0
# External USB is typically camera 1

python streaming_robot_control.py --config config/streaming_config.yaml --device cpu --webcam 0 1
```

### Scenario C: USB Cameras with Gaps
```bash
# You might have: /dev/video0 and /dev/video2 (video1 missing)

python streaming_robot_control.py --config config/streaming_config.yaml --device cpu --webcam 0 2
```

## Expected Output

### Single Camera
```
============================================================
STREAMING PIPELINE STARTED (DUAL CAMERA)
============================================================
Webcam 1 opened: 1920x1080 @ 60 Hz
...
Frame 60 | FPS: 58.3 | Single camera | Joint angles: (45.2°, 32.1°, 18.5°) | Gripper: 65.3%
Frame 120 | FPS: 57.9 | Single camera | Joint angles: (48.1°, 30.2°, 19.8°) | Gripper: 62.1%
```

### Dual Cameras
```
============================================================
STREAMING PIPELINE STARTED (DUAL CAMERA)
============================================================
Webcam 1 opened: 1920x1080 @ 60 Hz
Webcam 2 opened: 1920x1080 @ 60 Hz
...
Frame 60 | FPS: 31.2 | Dual cameras | Joint angles: (46.5°, 31.7°, 19.2°) | Gripper: 64.8%
Frame 120 | FPS: 30.8 | Dual cameras | Joint angles: (47.3°, 32.1°, 18.9°) | Gripper: 63.2%
```

### Demo Mode Output
```
======================================================================
DEMO RUNNING (Dual cameras) - Press 'q' to quit
======================================================================
Frame captured
Encoded 3DOF joint angles: [45.2, 31.7, 19.1]
Gripper force: 64.5%
Processing time: 32.4ms
FPS: 30.9
```

## Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| "Failed to open primary camera" | Check if camera ID exists: `ls /dev/video*` |
| "Failed to open secondary camera, using primary only" | Second camera doesn't exist or is in use |
| Frame rate drops below 30 FPS | Use GPU (`--device cuda`) or single camera mode |
| Camera shows but is distorted | Check camera resolution settings match actual hardware |
| One camera captures but other doesn't | Try swapping camera IDs: `--webcam 1 0` |

## Performance Tips

1. **For maximum FPS**: Use single camera (`--webcam 0`)
2. **For best robustness**: Use dual cameras (`--webcam 0 1`)
3. **For AI acceleration**: Use GPU if available (`--device cuda`)
4. **For lower latency**: Reduce target FPS (`--target-fps 30`)

## Hardware Requirements

- **CPU**: Intel i7/Ryzen 7 or better recommended
- **RAM**: 4GB minimum, 8GB recommended for dual cameras
- **Cameras**: 1080p USB webcams, ~30 FPS each
- **Arduino**: Connected via USB serial (115200 baud)

## Configuration Files

**Default**: `config/streaming_config.yaml`

Key settings for cameras:
```yaml
streaming:
  video_resolution: [1080, 1920]  # Height, Width (1080p)
  target_fps: 60
```

Update this file to change video resolution or target frame rate.

## Next Steps

1. **Test single camera**: `python demo_mock_streaming.py --duration 10 --device cpu`
2. **Identify second camera**: Run the Python script above
3. **Test dual cameras**: `python demo_mock_streaming.py --duration 10 --device cpu --webcam 0 1`
4. **Connect real Arduino**: Set port in config, then run production pipeline
5. **Optimize settings**: Adjust FPS/resolution based on your hardware performance
