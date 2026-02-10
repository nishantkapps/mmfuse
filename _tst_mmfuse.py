from robotic_feedback_system import RoboticFeedbackSystem
import torch

# Initialize system
system = RoboticFeedbackSystem(
    fusion_dim=512,
    fusion_method="concat_project",
    use_attention=False,
    device="cuda"
)
system.eval()

# Prepare inputs
camera1 = torch.randn(batch_size, 3, 224, 224)  # RGB image
camera2 = torch.randn(batch_size, 3, 224, 224)
audio = torch.randn(batch_size, 48000)           # 16kHz, 3 seconds
pressure = torch.randn(batch_size, 1000)         # Sequence of readings
emg = torch.randn(batch_size, 8, 1000)           # 8 channels

# Process and fuse
with torch.no_grad():
    fused_embedding = system(
        camera_images={'camera1': camera1, 'camera2': camera2},
        audio=audio,
        pressure=pressure,
        emg=emg
    )

print(fused_embedding.shape)  # (batch_size, 512)
