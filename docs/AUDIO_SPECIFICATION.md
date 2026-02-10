# Audio Input Specification - Voice Commands & Operational Sounds

## Overview
Audio input captures both **voice commands for robot control** and **operational sounds** from the robot's execution, providing rich multimodal context for state understanding.

## Voice Command Categories

### Movement Control
- **Vertical**: "Move up", "Move down"
- **Horizontal**: "Move left", "Move right", "Move forward", "Move backward"
- **Arm-specific**: "Move along arm", "Extend arm", "Retract arm"

### Rotation & Orientation
- "Rotate gripper", "Spin", "Turn", "Roll"

### Gripper Operations
- "Open gripper", "Close gripper"
- "Grip", "Release", "Grasp"

### Navigation & State
- "Return home", "Reset position", "Go to origin"
- "Hold position", "Wait"

### Task Execution
- "Execute task", "Start task", "Stop task", "Pause task"
- "Complete", "Abort", "Redo"

## Audio Capture Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Sample Rate** | 16 kHz | Standard for speech processing |
| **Duration** | 5 seconds | One window per camera frame |
| **Total Samples** | 80,000 | 5s × 16,000 Hz |
| **Channels** | 1 (Mono) | Single microphone input |
| **Bit Depth** | 16-bit | Standard microphone resolution |
| **Audio Content** | Voice + operational | Speech commands + robot response audio |

## Temporal Synchronization

```
Timeline for one 5-second window:
├── Voice Command Capture (0-2s)
│   └── Speaker gives movement instruction
├── Robot Response (2-5s)
│   ├── Motor engagement sounds
│   ├── Joint movement noise
│   └── Gripper/tool operation
└── Synchronized with camera frame at start time
```

## Multimodal Fusion Impact

The audio encoder processes these 80,000 samples to extract:
1. **Semantic intent** from voice commands (which instruction was given)
2. **Execution feedback** from operational audio (did the robot respond correctly)
3. **State indicators** from motor/actuator sounds (position changes, load)

Combined with vision (camera frame), this creates a unified 512-dimensional embedding representing:
- What the robot should do (command intent)
- What the robot is doing (visual state)
- How the robot is executing (operational audio feedback)

## Audio Preprocessing Pipeline

```
Raw Audio (80,000 samples @ 16kHz)
    ↓
Normalization ([-1, 1] range)
    ↓
Learnable 1D CNN Encoder (4 Conv layers)
    ↓
Adaptive Pooling + Projection
    ↓
768-dimensional Embedding
    ↓
Multimodal Fusion → 512-dim unified embedding
```

## Implementation Notes

- **Voice Recognition**: Audio encoder learns patterns without explicit speech recognition
- **Robustness**: Mixed speech + operational audio improves encoder generalization
- **Synchronization**: Each frame timestamped for precise vision-audio alignment
- **Batch Processing**: Multiple voice commands can be processed in parallel
- **Real-time**: 5-second windows allow for low-latency command response
