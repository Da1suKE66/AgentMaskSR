"""Optional multimodal metrics for Meissonic-SR candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image


@dataclass
class CLIPMetricResult:
    text_similarity: Optional[float]
    image_similarity: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "clip_text_similarity": self.text_similarity,
            "clip_image_similarity": self.image_similarity,
        }


class CLIPScorer:
    """Small wrapper around a cached CLIPModel for image-text/image-image checks."""

    def __init__(
        self,
        model_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()

    def _image_features(self, image: Image.Image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _text_features(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def image_text_similarity(self, image: Image.Image, text: str) -> Optional[float]:
        if not text.strip():
            return None
        image_features = self._image_features(image)
        text_features = self._text_features(text)
        return float((image_features * text_features).sum(dim=-1).detach().cpu().item())

    def image_image_similarity(self, candidate: Image.Image, reference: Image.Image) -> float:
        candidate_features = self._image_features(candidate)
        reference_features = self._image_features(reference)
        return float((candidate_features * reference_features).sum(dim=-1).detach().cpu().item())

    def score(
        self,
        candidate: Image.Image,
        prompt: str = "",
        reference: Optional[Image.Image] = None,
    ) -> CLIPMetricResult:
        return CLIPMetricResult(
            text_similarity=self.image_text_similarity(candidate, prompt),
            image_similarity=self.image_image_similarity(candidate, reference) if reference is not None else None,
        )


def naturalness_proxy(image: Image.Image) -> Dict[str, float]:
    """Cheap deterministic no-reference proxies used when VLM/CLIP is unavailable."""

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        "naturalness_luma_mean": float(gray.mean()),
        "naturalness_luma_std": float(gray.std()),
        "naturalness_rgb_mean": mean,
        "naturalness_rgb_std": std,
        "naturalness_grad_mean": float(grad.mean()),
    }
