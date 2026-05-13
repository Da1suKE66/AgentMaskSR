"""Meissonic VQ-token editor wrapper.

This module keeps Meissonic frozen and exposes the token artifacts the SR
controller needs: encoded base tokens, token masks, sampled candidate tokens,
and decoded candidate images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .token_masks import token_mask_to_pixel_mask


@dataclass
class TokenRefineResult:
    image: Image.Image
    tokens: np.ndarray
    seed: Optional[int]


class MeissonicTokenEditor:
    """Thin token-level wrapper around the existing Meissonic inpaint pipeline."""

    def __init__(self, pipe):
        self.pipe = pipe

    @property
    def vae_scale_factor(self) -> int:
        return int(self.pipe.vae_scale_factor)

    @property
    def device(self):
        return self.pipe._execution_device

    @property
    def mask_token_id(self) -> int:
        return int(self.pipe.scheduler.config.mask_token_id)

    @property
    def codebook_size(self) -> int:
        embedding = getattr(getattr(self.pipe.vqvae, "quantize", None), "embedding", None)
        if embedding is not None and hasattr(embedding, "weight"):
            return int(embedding.weight.shape[0])
        value = getattr(self.pipe.vqvae.config, "num_vq_embeddings", None)
        if value is not None:
            return int(value)
        raise AttributeError("could not infer VQ codebook size")

    def encode_image_tokens(self, image: Image.Image) -> np.ndarray:
        import torch

        with torch.no_grad():
            tensor = self.pipe.image_processor.preprocess(image)
            tensor = tensor.to(dtype=self.pipe.vqvae.dtype, device=self.device)
            latents = self.pipe.vqvae.encode(tensor).latents
            batch, _channels, height, width = latents.shape
            tokens = self.pipe.vqvae.quantize(latents)[2][2].reshape(batch, height, width)
        return tokens.detach().cpu().numpy().astype(np.int64)

    def decode_tokens(self, tokens: np.ndarray, target_size: Tuple[int, int]) -> Image.Image:
        import torch

        token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        if token_tensor.ndim == 2:
            token_tensor = token_tensor.unsqueeze(0)
        batch, height, width = token_tensor.shape
        with torch.no_grad():
            decoded = self.pipe.vqvae.decode(
                token_tensor,
                force_not_quantize=True,
                shape=(batch, height, width, self.pipe.vqvae.config.latent_channels),
            ).sample.clip(0, 1)
            images = self.pipe.image_processor.postprocess(decoded, output_type="pil")
        image = images[0].convert("RGB")
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.BICUBIC)
        return image

    def refine(
        self,
        image: Image.Image,
        active_token_mask: np.ndarray,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        temperature: Tuple[float, float],
        strength: float = 1.0,
        seed: Optional[int] = None,
        preserve_known_tokens: bool = True,
        preserve_known_pixels: bool = True,
    ) -> TokenRefineResult:
        import torch

        generator = None
        if seed is not None:
            generator_device = self.device if str(self.device).startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        active_mask = np.asarray(active_token_mask, dtype=bool)
        base_tokens = self.encode_image_tokens(image) if preserve_known_tokens else None
        mask_image = token_mask_to_pixel_mask(active_mask, image.size)
        effective_steps = max(1, int(num_inference_steps))
        effective_strength = max(float(strength), 1.0 / float(effective_steps))
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask_image,
                strength=effective_strength,
                guidance_scale=float(guidance_scale),
                num_inference_steps=effective_steps,
                generator=generator,
                temperature=temperature,
                output_type="latent",
            )
        tokens = result.images
        if hasattr(tokens, "detach"):
            tokens_np = tokens.detach().cpu().numpy().astype(np.int64)
        else:
            tokens_np = np.asarray(tokens, dtype=np.int64)
        if preserve_known_tokens and base_tokens is not None:
            generated = tokens_np[0] if tokens_np.ndim == 3 else tokens_np
            base = base_tokens[0] if base_tokens.ndim == 3 else base_tokens
            if generated.shape[-2:] != active_mask.shape:
                raise ValueError(
                    f"active token mask shape {active_mask.shape} does not match generated tokens {generated.shape[-2:]}"
                )
            invalid = np.logical_or(generated == self.mask_token_id, generated < 0)
            invalid = np.logical_or(invalid, generated >= self.codebook_size)
            generated = np.where(invalid, base, generated)
            merged = np.where(active_mask, generated, base).astype(np.int64)
            tokens_np = merged[None, ...] if tokens_np.ndim == 3 else merged
        decoded = self.decode_tokens(tokens_np, image.size)
        if preserve_known_pixels:
            pixel_mask = token_mask_to_pixel_mask(active_mask, image.size)
            decoded = Image.composite(decoded, image.convert("RGB"), pixel_mask)
        return TokenRefineResult(image=decoded, tokens=tokens_np, seed=seed)


def seed_for_candidate(base_seed: Optional[int], round_id: int, candidate_id: int) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed) + int(round_id) * 1000 + int(candidate_id)
