"""
Real-World Input Specifications for Robotic Multimodal Fusion System
======================================================================

Updated specifications for actual deployment with real sensors
"""

# =============================================================================
# CAMERA INPUTS (Real-World)
# =============================================================================

CAMERA_SPECS = {
    "description": "Dual webcams capturing robot workspace",
    
    # Physical specifications
    "resolution": "1024p or HD",
    "resolution_options": {
        "1024p": (1920, 1024),
        "HD": (1280, 720),
        "HD+": (1600, 900),
    },
    
    # Processing for CLIP model
    "clip_input_size": (224, 224),
    "processing_pipeline": [
        "1. Capture raw video frame from webcam (1920×1024 or 1280×720)",
        "2. Resize to CLIP input size (224×224)",
        "3. Apply CLIP normalization (ImageNet stats)",
        "4. Encode with CLIP ViT-B-32 → 512-dim embedding",
    ],
    
    # Frame sampling strategy
    "frame_sampling": {
        "total_cameras": 2,
        "sampling_rate": "Every Nth frame (e.g., every 5th frame @ 30fps = 6fps)",
        "sync_requirement": "Must capture synchronized frames from both cameras",
        "temporal_alignment": "Frames should be timestamped for audio sync",
    },
    
    # Data shapes
    "input_shapes": {
        "raw_frame": (1920, 1024, 3),  # or (1280, 720, 3) for HD
        "batch_raw": (batch_size, 1920, 1024, 3),
        "clip_input": (224, 224, 3),
        "clip_batch": (batch_size, 3, 224, 224),  # PyTorch format
        "output": (batch_size, 512),  # After CLIP encoding
    },
    
    # Temporal alignment with audio
    "alignment": "Each image frame synchronized with 5-second audio window",
    "notes": [
        "CLIP model automatically handles resizing and normalization",
        "Input to CLIP should be 224×224 RGB",
        "Output is L2-normalized 512-dimensional embedding",
    ]
}


# =============================================================================
# AUDIO INPUTS (Real-World)
# =============================================================================

AUDIO_SPECS = {
    "description": "Microphone capturing robot operational audio",
    
    # Capture specifications
    "sample_rate": 16000,  # Hz (16 kHz)
    "duration": 5.0,  # seconds (changed from 3 to 5)
    "num_channels": 1,  # Mono microphone
    
    # Sample calculation
    "total_samples": 16000 * 5,  # 80,000 samples per 5 seconds
    "calculation": "sample_rate (16kHz) × duration (5s) = 80,000 samples",
    
    # Audio content
    "audio_content": [
        "Voice commands for robot control:",
        "  - 'Move up', 'Move down'",
        "  - 'Move along arm', 'Rotate gripper'",
        "  - 'Open gripper', 'Close gripper'",
        "  - 'Return to home', 'Execute task'",
        "Robot operational sounds:",
        "  - Motor sounds during movement",
        "  - Joint actuator noise",
        "  - Gripper/tool operation sounds",
        "  - Task-specific audio cues",
    ],
    
    # Command vocabulary
    "command_vocabulary": {
        "movement": ["up", "down", "left", "right", "forward", "backward"],
        "rotation": ["rotate", "spin", "turn"],
        "gripper": ["open", "close", "grip", "release"],
        "navigation": ["home", "reset", "return"],
        "task": ["execute", "start", "stop", "pause"],
    },
    
    # Processing pipeline
    "processing_pipeline": [
        "1. Capture 5-second audio clip at 16kHz (80,000 samples)",
        "2. Normalize audio to [-1, 1] range",
        "3. Pass through learnable 1D CNN audio encoder",
        "4. Output: 768-dim embedding",
    ],
    
    # Data shapes
    "input_shapes": {
        "raw_audio": (80000,),  # 5 seconds at 16kHz
        "batch_raw": (batch_size, 80000),
        "output": (batch_size, 768),  # After encoding
    },
    
    # Temporal alignment with vision
    "alignment": {
        "one_image": "Synchronized with one 224×224 frame",
        "time_window": "5 seconds of audio = 1 image frame",
        "sync_requirement": "Audio window must be temporally aligned with camera frame",
    },
    
    # Future fusion considerations (Not changing model yet)
    "future_notes": [
        "Currently: Simple 5-sec audio → single 768-dim vector",
        "Future options:",
        "  - Sliding window approach (overlapping 5-sec windows)",
        "  - Frame-level features (extract features per 100ms window)",
        "  - Temporal pooling strategies (mean, max, attention)",
        "  - Audio + Vision temporal fusion (to be explored)",
    ]
}


# =============================================================================
# SENSOR INPUTS (Unchanged)
# =============================================================================

SENSOR_SPECS = {
    "pressure": {
        "description": "Single pressure sensor or tactile pad",
        "sample_rate": 1000,  # Hz
        "duration": 1.0,  # seconds
        "total_samples": 1000,
        "output_dim": 256,
    },
    
    "emg": {
        "description": "8-channel EMG electrode array",
        "num_channels": 8,
        "sample_rate": 1000,  # Hz
        "duration": 1.0,  # seconds
        "samples_per_channel": 1000,
        "output_dim": 256,
    }
}


# =============================================================================
# COMPLETE SYSTEM INPUT SPECIFICATION
# =============================================================================

COMPLETE_INPUT_SPEC = {
    "raw_inputs": {
        "camera_1": {
            "type": "Webcam frame",
            "shape": (1920, 1024, 3),  # or (1280, 720, 3)
            "description": "Raw RGB frame from webcam 1",
        },
        "camera_2": {
            "type": "Webcam frame",
            "shape": (1920, 1024, 3),  # or (1280, 720, 3)
            "description": "Raw RGB frame from webcam 2",
        },
        "audio": {
            "type": "Audio signal",
            "shape": (80000,),  # 5 seconds at 16kHz
            "description": "5-second audio clip from microphone",
        },
        "pressure": {
            "type": "Sensor signal",
            "shape": (1000,),
            "description": "1000 pressure samples @ 1kHz",
        },
        "emg": {
            "type": "Multi-channel sensor",
            "shape": (8, 1000),
            "description": "8-channel EMG @ 1kHz",
        },
    },
    
    "processed_inputs": {
        "camera_1_encoded": (512,),  # After CLIP
        "camera_2_encoded": (512,),  # After CLIP
        "vision_avg": (512,),  # Average of both cameras
        "audio_encoded": (768,),  # After learnable CNN encoder
        "pressure_encoded": (256,),  # After MLP encoder
        "emg_encoded": (256,),  # After MLP encoder
    },
    
    "fusion_output": {
        "fused_embedding": (512,),  # Unified multimodal representation
    },
    
    "batch_processing": {
        "description": "All inputs processed in batches",
        "batch_size": "Variable (1, 2, 4, 8, ...)",
        "example_batch": 2,
        "camera_1_batch": (2, 1920, 1024, 3),
        "audio_batch": (2, 80000),
        "pressure_batch": (2, 1000),
        "emg_batch": (2, 8, 1000),
    },

    "temporal_synchronization": {
        "requirement": "All modalities synchronized by timestamp",
        "synchronization_strategy": [
            "1. Capture camera frame at t=0",
            "2. Record 5-second audio window [t-2.5, t+2.5] centered on frame",
            "3. Capture pressure/EMG samples during [t-0.5, t+0.5]",
            "4. Align all modalities to frame timestamp",
        ],
        "output": "Single unified 512-dim embedding for this time instant",
    }
}


# =============================================================================
# DIMENSION SUMMARY TABLE
# =============================================================================

DIMENSION_SUMMARY = """
╔════════════════════╦════════════════════╦════════════════════════════════╗
║     MODALITY       ║    INPUT DIMS      ║       OUTPUT DIMS              ║
╠════════════════════╬════════════════════╬════════════════════════════════╣
║ Camera 1           ║ (1920, 1024, 3)    ║ 512-dim (L2-normalized)        ║
║ Camera 2           ║ (1920, 1024, 3)    ║ 512-dim (L2-normalized)        ║
║ Vision (avg)       ║ 2 × 512-dim        ║ 512-dim                        ║
║────────────────────┼────────────────────┼────────────────────────────────║
║ Audio              ║ 80,000 samples     ║ 768-dim (5 sec @ 16kHz)        ║
║                    ║ (5 sec @ 16kHz)    ║                                ║
║────────────────────┼────────────────────┼────────────────────────────────║
║ Pressure           ║ 1,000 samples      ║ 256-dim (1 sec @ 1kHz)         ║
║                    ║ (1 sec @ 1kHz)     ║                                ║
║────────────────────┼────────────────────┼────────────────────────────────║
║ EMG                ║ 8 × 1,000 samples  ║ 256-dim (8-channel @ 1kHz)     ║
║                    ║ (8ch, 1 sec @ 1k) ║                                ║
╠════════════════════╬════════════════════╬════════════════════════════════╣
║ PRE-FUSION         ║         -          ║ 1792-dim (512+768+256+256)     ║
║ FUSED OUTPUT       ║         -          ║ 512-dim (unified embedding)    ║
╚════════════════════╩════════════════════╩════════════════════════════════╝

Compression Ratio: 1792 / 512 = 3.5x
"""


if __name__ == "__main__":
    print("Real-World Input Specifications")
    print("=" * 80)
    print(f"\nCamera Resolution: HD (1280×720) or 1024p (1920×1024)")
    print(f"Audio Duration: 5 seconds @ 16kHz = 80,000 samples")
    print(f"Audio Sync: One 5-second window per camera frame")
    print(f"Pressure: 1000 samples @ 1kHz = 1 second")
    print(f"EMG: 8 channels × 1000 samples @ 1kHz = 1 second")
    print("\n" + DIMENSION_SUMMARY)
