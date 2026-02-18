"""
Vision Encoder using VisCoP pretrained model (egocentric/robot control).
Uses visual_probes from viscop_qwen2.5_7b_viscop-lora_egocentric-expert.
Output: (batch, 3584) - Qwen2.5 hidden size.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import sys


def _ensure_viscop_path():
    """Add VisCoP to path if cloned alongside mmfuse (e.g. projects/VisCoP)."""
    # Check common locations: same parent as mmfuse, or mmfuse/VisCoP
    this_file = Path(__file__).resolve()
    candidates = [
        this_file.parent.parent.parent / "VisCoP",  # projects/VisCoP
        this_file.parent.parent / "VisCoP",         # mmfuse/VisCoP
    ]
    for p in candidates:
        if p.exists() and (p / "viscop").exists():
            viscop_path = str(p)
            if viscop_path not in sys.path:
                sys.path.insert(0, viscop_path)
            return True
    return False


def _patch_transformers_video_input():
    """Patch transformers.image_utils with VideoInput if missing (removed in recent transformers)."""
    import transformers.image_utils as iu
    if not hasattr(iu, "VideoInput"):
        # VideoInput was removed; use ImageInput as alias (video = list of frames)
        iu.VideoInput = iu.ImageInput


def _patch_transformers_processing_kwargs():
    """Patch transformers.processing_utils with ProcessingKwargs and Unpack if missing (added in newer transformers)."""
    import sys
    from typing import TypedDict
    import transformers.processing_utils as pu
    if not hasattr(pu, "ProcessingKwargs"):
        # ProcessingKwargs added in transformers 4.46+; provide minimal TypedDict for older versions
        class ProcessingKwargs(TypedDict, total=False):
            _defaults: dict
            text_kwargs: dict
            images_kwargs: dict
            videos_kwargs: dict
            audio_kwargs: dict

        pu.ProcessingKwargs = ProcessingKwargs
    if not hasattr(pu, "Unpack"):
        # Unpack for TypedDict **kwargs; use typing (3.11+) or typing_extensions
        if sys.version_info >= (3, 11):
            import typing
            pu.Unpack = typing.Unpack
        else:
            import typing_extensions
            pu.Unpack = typing_extensions.Unpack


_ensure_viscop_path()
_patch_transformers_video_input()
_patch_transformers_processing_kwargs()
try:
    from viscop import model_init
    VISCOP_AVAILABLE = True
    _VISCOP_IMPORT_ERROR = None
except Exception as e:
    VISCOP_AVAILABLE = False
    _VISCOP_IMPORT_ERROR = e


HF_VISCOP_REPO = "dreilly/viscop-models"
HF_VISCOP_SUBFOLDER = "viscop_qwen2.5_7b_viscop-lora_egocentric-expert"
HF_VISCOP_BASE_MODEL = "DAMO-NLP-SG/VideoLLaMA3-7B-Image"  # Base VLM for LoRA adapter


def _is_valid_viscop_dir(p: Path) -> bool:
    """Check if path is a valid VisCoP model dir (has config.json)."""
    return p.exists() and p.is_dir() and (p / "config.json").exists()


def _resolve_viscop_model_path(model_path: str) -> str:
    """Resolve model path: use local if valid (has config.json), else download from HuggingFace."""
    p = Path(model_path)
    if _is_valid_viscop_dir(p):
        return str(p.resolve())
    # Try relative to cwd
    cwd_path = Path.cwd() / model_path
    if _is_valid_viscop_dir(cwd_path):
        return str(cwd_path.resolve())
    # Fallback: download from HuggingFace
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(HF_VISCOP_REPO)
        resolved = Path(cache_dir) / HF_VISCOP_SUBFOLDER
        if _is_valid_viscop_dir(resolved):
            return str(resolved)
    except Exception:
        pass
    raise FileNotFoundError(
        f"VisCoP model not found at '{model_path}'. "
        f"Download: python -m huggingface_hub.cli.hf download {HF_VISCOP_REPO} --local-dir viscop_trained_models\n"
        f"Then: --viscop-model-path viscop_trained_models/{HF_VISCOP_SUBFOLDER}\n"
        f"If gated: python -m huggingface_hub.cli.hf login"
    )


def _patch_viscop_config_contains():
    """Fix VisCoP assert 'num_visual_probes' in config - config is object not dict in some transformers versions."""
    try:
        from viscop.model.viscop_qwen2 import ViSCoP_Qwen2Config

        def _config_contains(self, key):
            return hasattr(self, key)

        ViSCoP_Qwen2Config.__contains__ = _config_contains
    except Exception:
        pass


def _patch_viscop_base_model():
    """Use HF base model when config's _name_or_path is a local path that doesn't exist."""
    if getattr(_patch_viscop_base_model, "_done", False):
        return
    _patch_viscop_base_model._done = True

    import viscop
    from transformers import PretrainedConfig

    # Patch viscop.load_pretrained_model (model_init imports from .model, so patch the re-export)
    _orig_load = viscop.load_pretrained_model

    def _patched_load(model_path, model_base, model_name, **kwargs):
        if model_base is None and ("lora" in model_name.lower() or "qlora" in model_name.lower()):
            try:
                cfg = PretrainedConfig.from_pretrained(model_path, token=kwargs.get("token"))
                base_path = getattr(cfg, "_name_or_path", None)
                if base_path and isinstance(base_path, str):
                    p = Path(base_path)
                    if not p.is_absolute():
                        p = Path.cwd() / base_path
                    if not (p.exists() and p.is_dir() and (p / "config.json").exists()):
                        model_base = HF_VISCOP_BASE_MODEL
            except Exception:
                pass
        return _orig_load(model_path, model_base, model_name, **kwargs)

    viscop.load_pretrained_model = _patched_load


class VisCoPVisionEncoder(nn.Module):
    """
    VisCoP vision encoder using pretrained egocentric/robot-control model.
    Extracts visual_probes (domain-adapted) as embeddings.
    Requires: pip install git+https://github.com/dominickrei/VisCoP.git
    Model: huggingface-cli download dreilly/viscop-models --local-dir ./viscop_trained_models
    """
    EMBEDDING_DIM = 3584  # Qwen2.5 hidden size (projected visual probes)

    def __init__(
        self,
        model_path: str = "viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert",
        frozen: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        if not VISCOP_AVAILABLE:
            msg = (
                "VisCoP not found. Clone it and add to PYTHONPATH:\n"
                "  git clone https://github.com/dominickrei/VisCoP.git\n"
                "  export PYTHONPATH=/path/to/VisCoP:$PYTHONPATH\n"
                "Or clone into projects/VisCoP (sibling of mmfuse) for auto-detection."
            )
            if _VISCOP_IMPORT_ERROR is not None:
                msg += f"\n\nOriginal error: {_VISCOP_IMPORT_ERROR}"
            raise ImportError(msg)
        self.device = device
        resolved_path = _resolve_viscop_model_path(model_path)
        self.model_path = resolved_path

        # Patch VisCoP config: 'x in config' fails when config is object not dict (older transformers)
        _patch_viscop_config_contains()
        # Patch: use HF base model when config points to non-existent local path
        _patch_viscop_base_model()

        device_map = "auto" if device == "auto" else {"": device}
        model, processor = model_init(model_path=resolved_path, device_map=device_map)
        self.model = model
        self.processor = processor
        self._num_visual_probes = getattr(model.config, 'num_visual_probes', 32)
        self._model_dtype = next(model.parameters()).dtype

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.EMBEDDING_DIM

    def _to_pil_list(self, images: Union[torch.Tensor, np.ndarray, List]) -> List[Image.Image]:
        """Convert batch of images to list of PIL Images for VisCoP processor."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [images]
            pil_list = []
            for i in range(images.shape[0]):
                arr = images[i]
                if arr.shape[0] == 3:  # CHW -> HWC
                    arr = arr.transpose(1, 2, 0)
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                pil_list.append(Image.fromarray(arr))
            return pil_list
        if isinstance(images, list):
            out = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if isinstance(img, np.ndarray):
                    if img.shape[0] == 3:
                        img = img.transpose(1, 2, 0)
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                out.append(img)
            return out
        raise TypeError(f"Unsupported image type: {type(images)}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings (visual_probes, mean-pooled).

        Args:
            images: (batch_size, 3, H, W) or (batch_size, H, W, 3), values [0,1] or [0,255]

        Returns:
            (batch_size, 3584) embeddings
        """
        pil_list = self._to_pil_list(images)
        # Single image per item: use merge_size=1
        image_inputs = self.processor.process_images(images=pil_list, merge_size=1, return_tensors="pt")
        inp_device = next(self.model.parameters()).device if self.device == "auto" else self.device
        pixel_values = image_inputs["pixel_values"].to(device=inp_device, dtype=self._model_dtype)
        grid_sizes = image_inputs["grid_sizes"]
        merge_sizes = image_inputs["merge_sizes"]
        if isinstance(grid_sizes, list):
            grid_sizes = torch.stack([g if isinstance(g, torch.Tensor) else torch.tensor(g) for g in grid_sizes]).to(inp_device)
        else:
            grid_sizes = grid_sizes.to(inp_device)
        if isinstance(merge_sizes, list):
            merge_sizes = torch.stack([m if isinstance(m, torch.Tensor) else torch.tensor(m) for m in merge_sizes]).to(inp_device)
        else:
            merge_sizes = merge_sizes.to(inp_device) if hasattr(merge_sizes, 'to') else torch.tensor(merge_sizes, device=inp_device)

        with torch.no_grad():
            mm_features, visual_probes = self.model.encode_images(
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
            )
        # visual_probes: (batch * num_probes, 3584)
        batch_size = len(pil_list)
        probes = visual_probes.view(batch_size, self._num_visual_probes, -1)
        return probes.mean(dim=1).float()  # (batch, 3584), float32 for downstream
