import torch
import cv2
import numpy as np
import os, random
import warnings, logging
import csv
from pathlib import Path

from mmfuse.preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor
from mmfuse.encoders.vision_encoder import VisionEncoder
from mmfuse.encoders.audio_encoder import Wav2VecPooledEncoder
from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
from config_modality import FUSION_DIM, get_modality_dims, TEXT_DIM, PRESSURE_DIM, EMG_DIM


def setup_environment(seed=0):
    warnings.filterwarnings("ignore")
    for _name in ("transformers", "timm", "open_clip", "huggingface_hub", "urllib3", "torch"):
        logging.getLogger(_name).setLevel(logging.ERROR)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return device


def load_encoders(device):
    """Load the same encoders used during training (CLIP vision, wav2vec audio)."""
    vision = VisionEncoder(device=device).to(device)
    vision.eval()
    for p in vision.parameters():
        p.requires_grad = False

    audio = Wav2VecPooledEncoder(frozen=True, device=device).to(device)
    audio.eval()

    pressure = PressureSensorEncoder(output_dim=PRESSURE_DIM, input_features=2).to(device)
    pressure.eval()

    emg = EMGSensorEncoder(output_dim=EMG_DIM, num_channels=3, input_features=4).to(device)
    emg.eval()

    return {'vision': vision, 'audio': audio, 'pressure': pressure, 'emg': emg}


def _extract_state(ckpt, prefix):
    """Extract sub-dict for a given prefix, stripping the prefix from keys."""
    p = prefix if prefix.endswith('.') else prefix + '.'
    return {k[len(p):]: v for k, v in ckpt.items() if k.startswith(p)}


class MovementHead(torch.nn.Module):
    """Regression head: fused embedding -> (delta_along, delta_lateral, magnitude)."""
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.fc = torch.nn.Linear(embedding_dim, 3)

    def forward(self, x):
        return self.fc(x)


def load_checkpoint(device, encoders, ckpt_path='checkpoints/sdata_clip_v2/ckpt_sdata_epoch_10.pt'):
    """Load trained fusion + classifier + movement head weights. Supports nested (epoch) and flat (exported) checkpoints."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Detect format: nested has 'fusion_state' key, flat has 'fusion.*' prefixed keys
    is_nested = 'fusion_state' in ckpt

    if is_nested:
        fusion_sd = ckpt['fusion_state']
        head_sd = ckpt['model_state']
        fusion_dim = int(ckpt.get('fusion_dim', FUSION_DIM))
        num_classes = int(ckpt.get('num_classes', 8))
        print(f"Loaded nested checkpoint (epoch {ckpt.get('epoch', '?')}): fusion_dim={fusion_dim} num_classes={num_classes}")
    else:
        fusion_sd = _extract_state(ckpt, 'fusion')
        head_sd = _extract_state(ckpt, 'action_head')
        fc_w = head_sd['fc.weight']
        num_classes, fusion_dim = fc_w.shape
        for name in ('vision', 'audio', 'pressure', 'emg'):
            sd = _extract_state(ckpt, name)
            if sd:
                encoders[name].load_state_dict(sd)
                encoders[name].eval()
        print(f"Loaded flat checkpoint: fusion_dim={fusion_dim} num_classes={num_classes}")

    modality_dims = get_modality_dims('clip')
    fusion = MultimodalFusionWithAttention(
        modality_dims=modality_dims, fusion_dim=fusion_dim, num_heads=8, dropout=0.2
    ).to(device)
    fusion.load_state_dict(fusion_sd)
    fusion.eval()
    encoders['fusion'] = fusion

    weight = head_sd['fc.weight'].to(device)
    bias = head_sd['fc.bias'].to(device)

    # Load movement head
    mov_head = MovementHead(embedding_dim=fusion_dim).to(device)
    if 'movement_state' in ckpt:
        mov_head.load_state_dict(ckpt['movement_state'])
        print(f"Loaded movement head from checkpoint")
    mov_head.eval()
    encoders['movement_head'] = mov_head

    # Load per-class movement targets (ground truth deltas)
    if 'movement_targets' in ckpt:
        mov_targets = ckpt['movement_targets'].to(device)
    else:
        mov_targets = torch.zeros(num_classes, 3, device=device)
    encoders['movement_targets'] = mov_targets

    predefined = ['Start', 'Go Here', 'Move Down', 'Move Up', 'Stop', 'Move Left', 'Move Right', 'Perfect']
    labels = predefined[:num_classes] if num_classes <= len(predefined) else \
        predefined + [f'part{i}' for i in range(len(predefined), num_classes)]

    return (weight, bias), labels


def _augment_frame(frame, variation_id=0):
    """Match training's deterministic augmentation (v=0: brightness=0.85, contrast=0.9)."""
    if frame is None or frame.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    out = frame.copy().astype(np.float32)
    if (variation_id // 8) % 2:
        out = np.ascontiguousarray(out[:, ::-1, :])
    brightness = [0.85, 0.95, 1.05, 1.15][(variation_id // 2) % 4]
    contrast = [0.9, 1.1][variation_id % 2]
    out = out * brightness
    out = (out - 127.5) * contrast + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_middle_frame(video_path):
    """Load middle frame from video (same as training dataset). Returns numpy RGB for VisionPreprocessor."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def _build_fused(encoders, c1_path, c2_path, audio_path, device):
    """Preprocess + encode + fuse one sample (mirrors build_embedding from training).
    Pass None for any path to use a zero tensor for that modality."""
    vprep = VisionPreprocessor()
    aprep = AudioPreprocessor(sample_rate=16000, duration=2.5)

    if c1_path:
        f1 = vprep.preprocess(_augment_frame(_load_middle_frame(c1_path))).unsqueeze(0).to(device)
    else:
        f1 = torch.zeros(1, 3, 224, 224, device=device)

    if c2_path:
        f2 = vprep.preprocess(_augment_frame(_load_middle_frame(c2_path))).unsqueeze(0).to(device)
    else:
        f2 = torch.zeros(1, 3, 224, 224, device=device)

    if audio_path and Path(audio_path).exists():
        a = aprep.preprocess(audio_path).unsqueeze(0).to(device)
    else:
        a = torch.zeros(1, int(aprep.duration * aprep.sample_rate), device=device)

    with torch.no_grad():
        v_emb1 = encoders['vision'](f1)
        v_emb2 = encoders['vision'](f2)
        a_emb = encoders['audio'](a)
        p_emb = torch.zeros(1, PRESSURE_DIM, device=device)
        e_emb = torch.zeros(1, EMG_DIM, device=device)

    embeddings = {
        'vision_camera1': v_emb1, 'vision_camera2': v_emb2,
        'audio': a_emb, 'text': torch.zeros(1, TEXT_DIM, device=device),
        'pressure': p_emb, 'emg': e_emb,
    }
    with torch.no_grad():
        fused, _ = encoders['fusion'](embeddings, return_kl=True)
    return fused


def _build_fused_ablation(encoders, c1_path, c2_path, audio_path, device,
                          skip_attention=False, skip_gating=False, skip_mlp=False):
    """Like _build_fused but allows bypassing fusion sub-components for ablation."""
    vprep = VisionPreprocessor()
    aprep = AudioPreprocessor(sample_rate=16000, duration=2.5)

    if c1_path:
        f1 = vprep.preprocess(_augment_frame(_load_middle_frame(c1_path))).unsqueeze(0).to(device)
    else:
        f1 = torch.zeros(1, 3, 224, 224, device=device)
    if c2_path:
        f2 = vprep.preprocess(_augment_frame(_load_middle_frame(c2_path))).unsqueeze(0).to(device)
    else:
        f2 = torch.zeros(1, 3, 224, 224, device=device)
    if audio_path and Path(audio_path).exists():
        a = aprep.preprocess(audio_path).unsqueeze(0).to(device)
    else:
        a = torch.zeros(1, int(aprep.duration * aprep.sample_rate), device=device)

    with torch.no_grad():
        v_emb1 = encoders['vision'](f1)
        v_emb2 = encoders['vision'](f2)
        a_emb = encoders['audio'](a)

    fusion = encoders['fusion']
    embeddings = {
        'vision_camera1': v_emb1, 'vision_camera2': v_emb2,
        'audio': a_emb, 'text': torch.zeros(1, TEXT_DIM, device=device),
        'pressure': torch.zeros(1, PRESSURE_DIM, device=device),
        'emg': torch.zeros(1, EMG_DIM, device=device),
    }

    with torch.no_grad():
        projected = {m: fusion.projections[m](e) for m, e in embeddings.items()}
        core = sorted(k for k in projected if 'camera' in k or 'vision' in k or 'audio' in k)
        stacked = torch.stack([projected[k] for k in core], dim=1)

        if skip_attention:
            attended = stacked
        else:
            attended, _ = fusion.attention(stacked, stacked, stacked)

        name_idx = {k: i for i, k in enumerate(core)}
        h1 = attended[:, name_idx['vision_camera1'], :]
        h2 = attended[:, name_idx['vision_camera2'], :]
        h3 = attended[:, name_idx['audio'], :]

        if skip_gating:
            fused_vec = (h1 + h2 + h3) / 3.0
        else:
            concat_h = torch.cat([h1, h2, h3], dim=-1)
            gates = torch.sigmoid(fusion.gate_net(concat_h))
            fused_vec = gates[:, 0:1] * h1 + gates[:, 1:2] * h2 + gates[:, 2:3] * h3

        if skip_mlp:
            return fused_vec
        return fusion.fusion_mlp(fused_vec)


def _classify(fused, weight, bias):
    """Apply trained linear head: logits = fused @ W^T + b."""
    with torch.no_grad():
        logits = fused @ weight.T + bias
        return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()


def _predict_movement(fused, encoders):
    """Apply movement head to fused embedding. Returns (delta_along, delta_lateral, magnitude)."""
    with torch.no_grad():
        pred = encoders['movement_head'](fused).squeeze(0).cpu().numpy()
    return float(pred[0]), float(pred[1]), float(pred[2])


def predict_sample(encoders, head, c1_path, c2_path, wav_path, device, labels):
    """Predict action class for a single (video1, video2, audio) triplet."""
    fused = _build_fused(encoders, c1_path, c2_path, wav_path, device)
    probs = _classify(fused, *head)
    pred = int(probs.argmax())
    return labels[pred], float(probs[pred]), probs


def predict_video_only(encoders, head, c1_path, c2_path, device, labels):
    """Predict using video frames only (zero audio)."""
    fused = _build_fused(encoders, c1_path, c2_path, None, device)
    probs = _classify(fused, *head)
    pred = int(probs.argmax())
    return labels[pred], float(probs[pred])


def predict_audio_only(encoders, head, wav_path, device, labels):
    """Predict using audio only (zero video frames)."""
    fused = _build_fused(encoders, None, None, wav_path, device)
    probs = _classify(fused, *head)
    pred = int(probs.argmax())
    return labels[pred], float(probs[pred])


def predict_video_frame1_only(encoders, head, c1_path, device, labels):
    """Predict using camera 1 only (zero camera 2 and audio)."""
    fused = _build_fused(encoders, c1_path, None, None, device)
    probs = _classify(fused, *head)
    pred = int(probs.argmax())
    return labels[pred], float(probs[pred])


def predict_video_frame2_only(encoders, head, c2_path, device, labels):
    """Predict using camera 2 only (zero camera 1 and audio)."""
    fused = _build_fused(encoders, None, c2_path, None, device)
    probs = _classify(fused, *head)
    pred = int(probs.argmax())
    return labels[pred], float(probs[pred])


def predict_no_attention(encoders, head, c1_path, c2_path, wav_path, device, labels):
    """Full modalities but skip cross-modal attention."""
    fused = _build_fused_ablation(encoders, c1_path, c2_path, wav_path, device, skip_attention=True)
    probs = _classify(fused, *head)
    pred = int(probs.argmax())
    return labels[pred], float(probs[pred])


def predict_no_gating(encoders, head, c1_path, c2_path, wav_path, device, labels):
    """Full modalities but replace learned gates with equal-weight average."""
    fused = _build_fused_ablation(encoders, c1_path, c2_path, wav_path, device, skip_gating=True)
    probs = _classify(fused, *head)
    pred = int(probs.argmax())
    return labels[pred], float(probs[pred])


def predict_no_mlp(encoders, head, c1_path, c2_path, wav_path, device, labels):
    """Full modalities but skip final fusion MLP."""
    fused = _build_fused_ablation(encoders, c1_path, c2_path, wav_path, device, skip_mlp=True)
    probs = _classify(fused, *head)
    pred = int(probs.argmax())
    return labels[pred], float(probs[pred])


def _ground_truth_label(dir_path, labels):
    """Extract ground truth from directory path (e.g. .../part3/p037/... → index 2 → labels[2])."""
    parts = Path(dir_path).parts
    for p in parts:
        if p.startswith('part') and p[4:].isdigit():
            idx = int(p[4:]) - 1
            if 0 <= idx < len(labels):
                return labels[idx], idx
    return '?', -1


def predict_all(root_dir, encoders, head, device, labels,
                output_csv='predictions.csv', latex_file='ablation_table.tex'):
    """Walk root_dir, run all ablations per sample, write CSV + LaTeX table with classification + movement MSE."""
    from sklearn.metrics import precision_recall_fscore_support

    ablation_keys = [
        'full', 'video_only', 'audio_only', 'cam1_only', 'cam2_only',
        'no_attention', 'no_gating', 'no_mlp',
    ]

    fused_builders = {
        'full':         lambda c1, c2, a: _build_fused(encoders, c1, c2, a, device),
        'video_only':   lambda c1, c2, a: _build_fused(encoders, c1, c2, None, device),
        'audio_only':   lambda c1, c2, a: _build_fused(encoders, None, None, a, device),
        'cam1_only':    lambda c1, c2, a: _build_fused(encoders, c1, None, None, device),
        'cam2_only':    lambda c1, c2, a: _build_fused(encoders, None, c2, None, device),
        'no_attention': lambda c1, c2, a: _build_fused_ablation(encoders, c1, c2, a, device, skip_attention=True),
        'no_gating':    lambda c1, c2, a: _build_fused_ablation(encoders, c1, c2, a, device, skip_gating=True),
        'no_mlp':       lambda c1, c2, a: _build_fused_ablation(encoders, c1, c2, a, device, skip_mlp=True),
    }

    mov_targets = encoders['movement_targets']
    all_gt = []
    all_gt_idx = []
    all_preds = {k: [] for k in ablation_keys}
    all_mov_se = {k: [] for k in ablation_keys}

    csv_header = ['sample', 'ground_truth']
    for k in ablation_keys:
        csv_header += [f'pred_{k}', f'conf_{k}', f'correct_{k}',
                       f'mov_da_{k}', f'mov_dl_{k}', f'mov_mag_{k}', f'mov_mse_{k}']

    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(csv_header)
        for sd, _, files in os.walk(root_dir):
            for wav in sorted(fi for fi in files if fi.endswith('.wav') and '_m1_' in fi):
                c1 = os.path.join(sd, wav.replace('_m1_', '_c1_').replace('.wav', '.mp4'))
                c2 = os.path.join(sd, wav.replace('_m1_', '_c2_').replace('.wav', '.mp4'))
                if not (os.path.exists(c1) and os.path.exists(c2)):
                    continue
                gt, gt_idx = _ground_truth_label(sd, labels)
                audio_path = os.path.join(sd, wav)
                all_gt.append(gt)
                all_gt_idx.append(gt_idx)
                gt_mov = mov_targets[gt_idx].cpu().numpy() if gt_idx >= 0 else [0, 0, 0]

                row = [wav, gt]
                for k in ablation_keys:
                    fused = fused_builders[k](c1, c2, audio_path)
                    probs = _classify(fused, *head)
                    pred_idx = int(probs.argmax())
                    pred_label = labels[pred_idx]
                    conf = float(probs[pred_idx])
                    all_preds[k].append(pred_label)

                    da, dl, mag = _predict_movement(fused, encoders)
                    se = (da - gt_mov[0])**2 + (dl - gt_mov[1])**2 + (mag - gt_mov[2])**2
                    all_mov_se[k].append(se)

                    row += [pred_label, f"{conf:.4f}", pred_label == gt,
                            f"{da:.4f}", f"{dl:.4f}", f"{mag:.4f}", f"{se:.4f}"]
                w.writerow(row)

    print(f"Wrote {output_csv}")

    ablation_display = {
        'full': 'Full model (ours)',
        'video_only': 'Video only (cam1 + cam2)',
        'audio_only': 'Audio only',
        'cam1_only': 'Camera 1 only',
        'cam2_only': 'Camera 2 only',
        'no_attention': 'w/o cross-modal attention',
        'no_gating': 'w/o gated fusion',
        'no_mlp': 'w/o fusion MLP',
    }

    metrics = {}
    for k in ablation_keys:
        n = len(all_gt)
        acc = sum(p == g for p, g in zip(all_preds[k], all_gt)) / n if n else 0
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_gt, all_preds[k], average='macro', zero_division=0)
        mse = sum(all_mov_se[k]) / n if n else 0
        metrics[k] = (acc, prec, rec, f1, mse)
        print(f"  {ablation_display[k]:34s}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} mov_mse={mse:.4f}")

    with open(latex_file, 'w') as f:
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Ablation study on the SData benchmark. Movement MSE measures regression error for predicted displacement deltas.}\n')
        f.write('\\label{tab:ablation}\n')
        f.write('\\begin{tabular}{lccccc}\n')
        f.write('\\toprule\n')
        f.write('Configuration & Accuracy & Precision & Recall & F1 & Mov.~MSE \\\\\n')
        f.write('\\midrule\n')
        for i, k in enumerate(ablation_keys):
            acc, prec, rec, f1, mse = metrics[k]
            name = ablation_display[k]
            if k == 'full':
                line = '\\textbf{' + name + '}'
                line += f' & \\textbf{{{acc:.2%}}} & \\textbf{{{prec:.2%}}} & \\textbf{{{rec:.2%}}} & \\textbf{{{f1:.2%}}} & \\textbf{{{mse:.4f}}} \\\\\n'
            else:
                line = f'{name} & {acc:.2%} & {prec:.2%} & {rec:.2%} & {f1:.2%} & {mse:.4f} \\\\\n'
            if k == 'no_attention':
                f.write('\\midrule\n')
            f.write(line)
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    print(f"Wrote {latex_file}")


if __name__ == '__main__':
    device = setup_environment()
    encoders = load_encoders(device)
    head, labels = load_checkpoint(device, encoders)

    # videofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p052/p052_c1_part4.mp4'
    # videofil2 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p052/p052_c2_part4.mp4'
    # audiofil1 = '/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p052/p052_m1_part4.wav'

    # label, conf, probs = predict_sample(encoders, head, videofil1, videofil2, audiofil1, device, labels)
    # print(" -------------------------------------------- ")
    # print('pred:', label, 'conf:', conf)
    # print(" -------------------------------------------- ")

    # lv, cv = predict_video_only(encoders, head, videofil1, videofil2, device, labels)
    # print('pred_video_only:', lv, 'conf:', cv)
    # print(" -------------------------------------------- ")

    # la, ca = predict_audio_only(encoders, head, audiofil1, device, labels)
    # print('pred_audio_only:', la, 'conf:', ca)
    # print(" -------------------------------------------- ")

    predict_all('/home/theta/nishant/projects/mmfuse/dataset/sdata', encoders, head, device, labels)
