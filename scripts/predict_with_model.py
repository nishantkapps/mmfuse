from run.robotic_feedback_system import RoboticFeedbackSystem
import torch, librosa
from torchvision.io import read_video

# init model
sys = RoboticFeedbackSystem(device='cpu'); sys.eval()

# inputs
videofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p051/p051_c1_part4.mp4'
videofil2 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p051/p051_c2_part4.mp4'
audiofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p051/p051_m1_part4.wav'

# load inputs (replace paths)
v1,_,_ = read_video(videofil1)
v2,_,_ = read_video(videofil2)
frame1 = v1[0].permute(2,0,1).unsqueeze(0).float()/255.0
frame2 = v2[0].permute(2,0,1).unsqueeze(0).float()/255.0
audio, _ = librosa.load(audiofil1, sr=16000); 
audio = torch.tensor(audio).unsqueeze(0)

with torch.no_grad():
    fused = sys(camera_images={'camera1':frame1,'camera2':frame2}, audio=audio, pressure=torch.zeros(1,1000), emg=torch.zeros(1,8,1000))
print('fused shape:', fused.shape)
