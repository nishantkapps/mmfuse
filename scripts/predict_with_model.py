from run.robotic_feedback_system import RoboticFeedbackSystem
import torch, librosa
from torchvision.io import read_video
from PIL import Image
import os, random, numpy as np
import warnings, logging

# suppress Python warnings and noisy loggers
warnings.filterwarnings("ignore")
for _name in ("transformers", "timm", "open_clip", "huggingface_hub", "urllib3", "torch"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Make inference deterministic where possible
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# init model (use GPU when available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# seed CUDA RNGs as well
if device.startswith('cuda'):
    torch.cuda.manual_seed_all(seed)
# instantiate model while silencing stdout/stderr from noisy libs
rfs = RoboticFeedbackSystem(device=device)
rfs.to(device)
rfs.eval()

# inputs
#videofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p051/p051_c1_part4.mp4'
#videofil2 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p051/p051_c2_part4.mp4'
#audiofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p051/p051_m1_part4.wav'

#videofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part3/p037/p037_c1_part3.mp4'
#videofil2 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part3/p037/p037_c2_part3.mp4'
#audiofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part3/p037/p037_m1_part3.wav'

videofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p052/p052_c1_part4.mp4'
videofil2 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p052/p052_c2_part4.mp4'
audiofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p052/p052_m1_part4.wav'

audio_np, _ = librosa.load(audiofil1, sr=16000); 
audio = rfs.audio_preprocessor.preprocess(audio_np, sr=16000).unsqueeze(0).to(device)
# load and preprocess inputs (suppress noisy stdout/stderr from video backend)
v1,_,_ = read_video(videofil1)
v2,_,_ = read_video(videofil2)
# use the model's CLIP preprocess (returns 3,H,W)
frame1 = rfs.vision_encoder.preprocess_image(Image.fromarray(v1[0].numpy())).unsqueeze(0).to(device)
frame2 = rfs.vision_encoder.preprocess_image(Image.fromarray(v2[0].numpy())).unsqueeze(0).to(device)
audio_np, _ = librosa.load(audiofil1, sr=16000)
audio = rfs.audio_preprocessor.preprocess(audio_np, sr=16000).unsqueeze(0).to(device)
with torch.no_grad():
    fused = rfs(camera_images={'camera1':frame1,'camera2':frame2}, audio=audio, pressure=torch.zeros(1,1000).to(device), emg=torch.zeros(1,8,1000).to(device))

# Load trained classifier from checkpoint and predict
from training.train_sdata_attention import ActionClassifier
ckpt_path = 'checkpoints/full_sdata_clip_v2_model.pt'
ckpt = torch.load(ckpt_path, map_location=device)
#print('ckpt keys:', list(ckpt.keys()))
fusion_dim = int(ckpt.get('fusion_dim', fused.shape[-1]))
num_classes = int(ckpt.get('num_classes', 8))
classifier = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes).to(device)

# robustly find a state-dict inside the checkpoint for the classifier
def _find_state_dict(d):
    common = ['model_state', 'state_dict', 'model_state_dict', 'model', 'action_state', 'movement_state']
    for k in common:
        if k in d and isinstance(d[k], dict):
            return k, d[k]
    for k,v in d.items():
        if isinstance(v, dict):
            keys = list(v.keys())
            if any(('weight' in kk or 'bias' in kk) for kk in keys if isinstance(kk, str)):
                return k, v
    for k,v in d.items():
        if isinstance(v, dict):
            for k2,v2 in v.items():
                if isinstance(v2, dict):
                    keys = list(v2.keys())
                    if any(('weight' in kk or 'bias' in kk) for kk in keys if isinstance(kk, str)):
                        return f"{k}.{k2}", v2
    return None, {}

found_key, state_dict_candidate = _find_state_dict(ckpt)
before_sum = sum(p.abs().sum().item() for p in classifier.parameters())
loaded_strict = False
exc = None
if state_dict_candidate:
    try:
        classifier.load_state_dict(state_dict_candidate)
        loaded_strict = True
    except Exception as e:
        exc = e
        classifier.load_state_dict(state_dict_candidate, strict=False)
else:
    # handle case where checkpoint is itself a flat state-dict (param_name -> tensor)
    if all(isinstance(v, (torch.Tensor,)) for v in ckpt.values()):
        # collect top-level prefixes
        prefixes = {}
        for k in ckpt.keys():
            pref = k.split('.', 1)[0]
            prefixes[pref] = prefixes.get(pref, 0) + 1
        # prefer 'action_head' if present
        chosen = None
        for p in ('action_head', 'action', 'classifier', 'model'):
            if p in prefixes:
                chosen = p
                break
        if chosen is None and prefixes:
            # pick most common prefix
            chosen = max(prefixes.items(), key=lambda x: x[1])[0]
        if chosen:
            # build candidate by stripping the prefix + dot
            candidate = {k[len(chosen)+1:]: v for k, v in ckpt.items() if k.startswith(chosen + '.')}
            if candidate:
                found_key = chosen
                state_dict_candidate = candidate
                try:
                    classifier.load_state_dict(state_dict_candidate)
                    loaded_strict = True
                except Exception as e:
                    exc = e
                    classifier.load_state_dict(state_dict_candidate, strict=False)
    if not state_dict_candidate:
        exc = 'no_state_dict_found'

after_sum = sum(p.abs().sum().item() for p in classifier.parameters())
print(f'found_state_key={found_key} classifier weights sum before={before_sum:.6f} after={after_sum:.6f} loaded_strict={loaded_strict} exc={exc}')

if found_key is None:
    # dump a small summary of the checkpoint to help locate the state dict
    def _summarize(obj, depth=0, maxdepth=2):
        indent = '  ' * depth
        if depth > maxdepth:
            print(f"{indent}... (max depth reached)")
            return
        if isinstance(obj, dict):
            print(f"{indent}dict with {len(obj)} keys")
            for k in list(obj.keys())[:10]:
                v = obj[k]
                t = type(v)
                if torch.is_tensor(v):
                    print(f"{indent}  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
                elif isinstance(v, dict):
                    print(f"{indent}  {k}: dict ->")
                    _summarize(v, depth+2, maxdepth)
                else:
                    print(f"{indent}  {k}: {t.__name__}")
        else:
            print(f"{indent}{type(obj).__name__}")

    print('\nCheckpoint summary (top-level, nested up to depth=2):')
    _summarize(ckpt, depth=0, maxdepth=2)

classifier.eval()
with torch.no_grad():
    logits = classifier(fused.to(device))
    #print('logits full:', logits.cpu().numpy())
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())

print(" -------------------------------------------- ")
# map class indices to human-friendly action names (use known mapping, fallback to part#)
predefined = ['Start','Go Here','Move Down','Move Up','Stop','Move Left','Move Right','Perfect']
if num_classes <= len(predefined):
    labels = predefined[:num_classes]
else:
    labels = predefined + [f'part{i}' for i in range(len(predefined), num_classes)]
print('pred:', labels[pred], 'conf:', float(probs[pred]))
# also show all label probabilities for debugging (only for the first/full prediction)
all_probs = [(labels[i], float(probs[i])) for i in range(len(labels))]
#print('all_probs:', ', '.join([f"{lab}:{p:.4f}" for lab, p in all_probs]))

print(" -------------------------------------------- ")
# Quick video-only check: run fusion with silent audio and predict
silence = np.zeros(16000, dtype=np.float32)
audio_sil = rfs.audio_preprocessor.preprocess(silence, sr=16000).unsqueeze(0).to(device)
with torch.no_grad():
    fused_video = rfs(camera_images={'camera1':frame1,'camera2':frame2}, audio=audio_sil, pressure=torch.zeros(1,1000).to(device), emg=torch.zeros(1,8,1000).to(device))
    #print('fused_video mean,std,shape:', float(fused_video.mean().cpu()), float(fused_video.std().cpu()), fused_video.shape)
    logits_v = classifier(fused_video.to(device))
    #print('logits video-only:', logits_v.cpu().numpy())
    probs_v = torch.softmax(logits_v, dim=-1).squeeze(0).cpu().numpy()
    pred_v = int(probs_v.argmax())
print('pred_video_only:', labels[pred_v], 'conf:', float(probs_v[pred_v]))

# show all probs for video-only
all_probs_v = [(labels[i], float(probs_v[i])) for i in range(len(labels))]
#print('all_probs_video_only:', ', '.join([f"{lab}:{p:.4f}" for lab, p in all_probs_v]))

print(" -------------------------------------------- ")
# Quick audio-only check: zero-out frames and use real audio
frames_zero1 = torch.zeros_like(frame1)
frames_zero2 = torch.zeros_like(frame2)
with torch.no_grad():
    fused_audio = rfs(camera_images={'camera1':frames_zero1,'camera2':frames_zero2}, audio=audio, pressure=torch.zeros(1,1000).to(device), emg=torch.zeros(1,8,1000).to(device))
    #print('fused_audio mean,std,shape:', float(fused_audio.mean().cpu()), float(fused_audio.std().cpu()), fused_audio.shape)
    logits_a = classifier(fused_audio.to(device))
    #print('logits audio-only:', logits_a.cpu().numpy())
    probs_a = torch.softmax(logits_a, dim=-1).squeeze(0).cpu().numpy()
    pred_a = int(probs_a.argmax())
print('pred_audio_only:', labels[pred_a], 'conf:', float(probs_a[pred_a]))

# show all probs for audio-only
all_probs_a = [(labels[i], float(probs_a[i])) for i in range(len(labels))]
#print('all_probs_audio_only:', ', '.join([f"{lab}:{p:.4f}" for lab, p in all_probs_a]))
print(" -------------------------------------------- ")
