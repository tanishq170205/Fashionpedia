"""
CLIP text (and image) encoder for the retriever — dual-backend.

Supports both OpenAI's clip package and open_clip (for fashion-domain checkpoints
such as hf-hub:Marqo/marqo-fashionCLIP).

Backend selection is automatic based on model name format:
  "ViT-B/32", "ViT-L/14"              → clip (OpenAI package)
  "hf-hub:Marqo/marqo-fashionCLIP"   → open_clip package

The vectors produced here are directly comparable to those stored in ChromaDB
only if the same model was used during indexing. The model mismatch guard in
app/main.py enforces this at startup.
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image


_model = None
_preprocess = None
_tokenizer = None   # used by open_clip only
_device: str = "cpu"
_loaded_model_name: str = ""
_backend: str = ""  # "clip" or "open_clip"


_OPENAI_CLIP_NAMES = {
    "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
    "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px",
}


def _is_open_clip_model(model_name: str) -> bool:
    if model_name in _OPENAI_CLIP_NAMES:
        return False
    return True  # hf-hub: prefix or any unknown format


def load_model(model_name: str = "ViT-B/32", device: str = "cpu") -> None:
    global _model, _preprocess, _tokenizer, _device, _loaded_model_name, _backend

    if _model is not None and _loaded_model_name == model_name:
        return

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    _device = device

    if _is_open_clip_model(model_name):
        try:
            import open_clip  # type: ignore
        except ImportError as e:
            raise ImportError(
                "open_clip_torch is required for fashion-domain checkpoints. "
                "Install with: pip install open_clip_torch"
            ) from e
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            model_name, device=device
        )
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(model_name)
        _backend = "open_clip"
    else:
        import clip  # type: ignore
        _model, _preprocess = clip.load(model_name, device=device)
        _model.eval()
        _tokenizer = None
        _backend = "clip"

    _loaded_model_name = model_name


def encode_text(text: str) -> np.ndarray:
    """Return an L2-normalized float32 embedding for a text string."""
    if _model is None:
        raise RuntimeError("Call load_model() before encode_text().")

    if _backend == "open_clip":
        tokens = _tokenizer([text]).to(_device)
    else:
        import clip  # type: ignore
        tokens = clip.tokenize([text], truncate=True).to(_device)

    with torch.no_grad():
        embedding = _model.encode_text(tokens)

    vec = embedding.squeeze(0).float().cpu().numpy()
    return _l2_normalize(vec)


def encode_image(image: Image.Image) -> np.ndarray:
    """Return an L2-normalized float32 embedding for a PIL image."""
    if _model is None:
        raise RuntimeError("Call load_model() before encode_image().")

    tensor = _preprocess(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        embedding = _model.encode_image(tensor)

    vec = embedding.squeeze(0).float().cpu().numpy()
    return _l2_normalize(vec)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm
