"""
Central config for modality dimensions and fusion.
Used by: precompute_video_text, train_sdata_attention, finetune, run_dataset, run_sdata_baselines, etc.
"""
# Canonical dimensions (single source of truth)
FUSION_DIM = 512
AUDIO_DIM = 768
TEXT_DIM = 768
PRESSURE_DIM = 256
EMG_DIM = 256
VISION_DIM_CLIP = 512
VISION_DIM_VISCOP = 3584


def get_modality_dims(vision_encoder: str = "clip"):
    """Return modality_dims dict for 6 modalities. vision_encoder: 'clip' | 'viscop'."""
    vision_dim = VISION_DIM_VISCOP if vision_encoder == "viscop" else VISION_DIM_CLIP
    return {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": AUDIO_DIM,
        "text": TEXT_DIM,
        "pressure": PRESSURE_DIM,
        "emg": EMG_DIM,
    }


def get_embedding_config(vision_encoder: str = "clip"):
    """Return dict for embeddings config.json (vision_dim, audio_dim, text_dim, etc.)."""
    vision_dim = VISION_DIM_VISCOP if vision_encoder == "viscop" else VISION_DIM_CLIP
    return {
        "vision_dim": vision_dim,
        "audio_dim": AUDIO_DIM,
        "text_dim": TEXT_DIM,
    }
