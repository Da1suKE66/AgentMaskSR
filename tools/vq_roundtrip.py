#!/usr/bin/env python
"""Measure Meissonic VQ encode/decode round-trip drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentsr.reranker import lr_grad_l1, lr_l1  # noqa: E402
from agentsr.token_editor import MeissonicTokenEditor  # noqa: E402
from agentsr.controller import downsample_consistency_metrics, parse_resolution  # noqa: E402
from tools.agent_mask_sr import load_meissonic_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Meissonic VQ round-trip test.")
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--output_dir", default="outputs/vq_roundtrip")
    parser.add_argument("--target_resolution", default=None, help="Optional WIDTHxHEIGHT resize before VQ encode.")
    parser.add_argument("--lr_image", default=None, help="Optional LR observation for downsample consistency.")
    parser.add_argument("--model_path", default="MeissonFlow/Meissonic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float32")
    return parser.parse_args()


def ensure_repo_local(path: Path) -> Path:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if repo not in (resolved, *resolved.parents):
        raise ValueError(f"output path must stay inside repository: {repo}")
    return resolved


def _rgb(image: Image.Image, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _image_metrics(reference: Image.Image, candidate: Image.Image) -> Dict[str, float]:
    ref = _rgb(reference)
    cand = _rgb(candidate, reference.size)
    diff = cand - ref
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 1e-12 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    return {
        "l1": float(np.mean(np.abs(diff))),
        "mse": mse,
        "psnr": psnr,
        "mean_delta_r": float(cand[..., 0].mean() - ref[..., 0].mean()),
        "mean_delta_g": float(cand[..., 1].mean() - ref[..., 1].mean()),
        "mean_delta_b": float(cand[..., 2].mean() - ref[..., 2].mean()),
    }


def main() -> int:
    args = parse_args()
    output_dir = ensure_repo_local(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.input_image).convert("RGB")
    if args.target_resolution:
        image = image.resize(parse_resolution(args.target_resolution), Image.Resampling.BICUBIC)

    input_path = output_dir / "vq_input.png"
    recon_path = output_dir / "vq_recon.png"
    image.save(input_path)

    pipe = load_meissonic_pipeline(args.model_path, args.device, dtype=args.dtype)
    editor = MeissonicTokenEditor(pipe)
    tokens = editor.encode_image_tokens(image)
    recon = editor.decode_tokens(tokens, image.size)
    recon.save(recon_path)

    metrics: Dict[str, object] = {
        "input_image": str(args.input_image),
        "input_size": list(image.size),
        "token_shape": list(tokens.shape[-2:]),
        "vq_input": str(input_path),
        "vq_recon": str(recon_path),
        "roundtrip": _image_metrics(image, recon),
    }
    if args.lr_image:
        lr = Image.open(args.lr_image).convert("RGB")
        metrics["downsample_input_vs_lr"] = downsample_consistency_metrics(image, lr)
        metrics["downsample_recon_vs_lr"] = downsample_consistency_metrics(recon, lr)
        metrics["recon_lr_l1"] = lr_l1(recon, lr)
        metrics["recon_lr_grad_l1"] = lr_grad_l1(recon, lr)

    metrics_path = output_dir / "roundtrip_error_report.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
