#!/usr/bin/env python
"""Agent-guided Meissonic SR/detail/outpaint entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentsr.controller import (  # noqa: E402
    DEFAULT_NEGATIVE_PROMPT,
    build_refinement_assets,
    derive_agent_plan,
    downsample_consistency_metrics,
    load_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-free agent-guided masked token refinement for Meissonic."
    )
    parser.add_argument("--input_image", required=True, help="Low-resolution observation image.")
    parser.add_argument("--output_dir", default="outputs/agent_mask_sr", help="Repository-local output directory.")
    parser.add_argument("--prompt", default="", help="User instruction or image editing prompt.")
    parser.add_argument("--plan_json", default=None, help="Optional existing AgentPlan JSON.")
    parser.add_argument("--mode", choices=["sr", "detail", "outpaint", "sr_outpaint"], default=None)
    parser.add_argument("--target_resolution", default="1024x1024", help="WIDTHxHEIGHT target resolution.")
    parser.add_argument("--alpha", type=float, default=None, help="Detail/outpaint strength in [0, 1].")
    parser.add_argument("--outpaint_direction", nargs="*", default=None, choices=["left", "right", "top", "bottom"])
    parser.add_argument("--outpaint_margin_ratio", type=float, default=0.18)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--tile_overlap", type=int, default=128)
    parser.add_argument("--dry_run", action="store_true", help="Only write plan, mask, init image, and metrics.")
    parser.add_argument("--run_meissonic", action="store_true", help="Run the Meissonic inpaint backend.")
    parser.add_argument("--model_path", default="MeissonFlow/Meissonic")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=9.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args()


def ensure_repo_local(path: Path) -> Path:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if repo not in (resolved, *resolved.parents):
        raise ValueError(f"output path must stay inside repository: {repo}")
    return resolved


def load_meissonic_pipeline(model_path: str, device: str, dtype: str = "float32"):
    import torch
    from diffusers import VQModel
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    from src.pipeline_inpaint import InpaintPipeline
    from src.scheduler import Scheduler
    from src.transformer import Transformer2DModel

    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer")
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae")
    text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")

    dtype_map = {
        "auto": torch.float16 if device.startswith("cuda") else torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map[dtype]
    model = model.to(dtype=target_dtype)
    vq_model = vq_model.to(dtype=target_dtype)
    text_encoder = text_encoder.to(dtype=target_dtype)

    pipe = InpaintPipeline(
        vq_model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=model,
        scheduler=scheduler,
    )
    return pipe.to(device)


def run_meissonic(args: argparse.Namespace, assets: dict, output_dir: Path) -> Path:
    import torch

    plan = assets["plan"]
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device if args.device.startswith("cuda") else "cpu").manual_seed(args.seed)

    pipe = load_meissonic_pipeline(args.model_path, args.device, dtype=args.dtype)
    result = pipe(
        prompt=plan.prompt,
        negative_prompt=args.negative_prompt,
        image=assets["init_image"],
        mask_image=assets["mask_image"],
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
        temperature=(max(0.01, plan.temperature), 0.0),
    ).images[0]

    output_path = output_dir / "meissonic_refined.png"
    result.save(output_path)

    metrics = downsample_consistency_metrics(result, Image.open(args.input_image).convert("RGB"))
    metrics_path = output_dir / "meissonic_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_path


def main() -> int:
    args = parse_args()
    output_dir = ensure_repo_local(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(args.input_image).convert("RGB")
    if args.plan_json:
        plan = load_plan(Path(args.plan_json))
    else:
        plan = derive_agent_plan(
            args.prompt,
            target_resolution=args.target_resolution,
            mode=args.mode,
            alpha=args.alpha,
            outpaint_direction=args.outpaint_direction,
        )

    assets = build_refinement_assets(
        input_image,
        plan,
        output_dir,
        outpaint_margin_ratio=args.outpaint_margin_ratio,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    summary = {
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "run_meissonic": args.run_meissonic,
        "assets": assets["paths"],
        "diagnostics": assets["diagnostics"],
    }

    if args.run_meissonic and not args.dry_run:
        summary["refined_image"] = str(run_meissonic(args, assets, output_dir))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
