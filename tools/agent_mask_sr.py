#!/usr/bin/env python
# Agent-guided Meissonic SR/detail/outpaint entry point.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentsr.controller import (  # noqa: E402
    AgentPlan,
    DEFAULT_NEGATIVE_PROMPT,
    MASK_STRATEGIES,
    build_refinement_assets,
    derive_agent_plan,
    downsample_consistency_metrics,
    load_plan,
    observation_consistency_project,
)
from agentsr.reranker import ScoreWeights, lr_grad_l1, lr_l1, score_candidates, write_scores  # noqa: E402
from agentsr.semantic_metrics import CLIPScorer, naturalness_proxy  # noqa: E402
from agentsr.token_editor import MeissonicTokenEditor, seed_for_candidate  # noqa: E402
from agentsr.token_masks import (  # noqa: E402
    TokenMaskSet,
    build_initial_token_masks,
    mask_metadata,
    save_token_masks_npz,
    token_mask_to_pixel_mask,
    update_masks_after_round,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Training-free agent-guided masked token refinement for Meissonic.'
    )
    parser.add_argument('--input_image', required=True, help='Low-resolution observation image.')
    parser.add_argument('--output_dir', default='outputs/agent_mask_sr', help='Repository-local output directory.')
    parser.add_argument('--prompt', default='', help='User instruction or image editing prompt.')
    parser.add_argument('--plan_json', default=None, help='Optional existing AgentPlan JSON.')
    parser.add_argument(
        '--mode',
        choices=[
            'sr',
            'detail',
            'outpaint',
            'sr_outpaint',
            'token_rerank_sr',
            'progressive_token_rerank_sr',
            'cascaded_token_rerank_sr',
        ],
        default=None,
    )
    parser.add_argument('--target_resolution', default='1024x1024', help='WIDTHxHEIGHT target resolution.')
    parser.add_argument('--alpha', type=float, default=None, help='Detail/outpaint strength in [0, 1].')
    parser.add_argument('--outpaint_direction', nargs='*', default=None, choices=['left', 'right', 'top', 'bottom'])
    parser.add_argument('--outpaint_margin_ratio', type=float, default=0.18)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--tile_overlap', type=int, default=128)
    parser.add_argument('--dry_run', action='store_true', help='Only write controller assets and token metadata.')
    parser.add_argument('--run_meissonic', action='store_true', help='Run the Meissonic backend.')
    parser.add_argument('--model_path', default='MeissonFlow/Meissonic')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--guidance_scale', type=float, default=9.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--negative_prompt', default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        '--mask_strategy',
        choices=list(MASK_STRATEGIES),
        default='uncage',
        help='Pixel-to-token mask policy for partial refinement.',
    )

    parser.add_argument('--rounds', type=int, default=2, help='Token-rerank refinement rounds.')
    parser.add_argument('--candidate_k', type=int, default=2, help='Candidates sampled per token-rerank round.')
    parser.add_argument('--progressive_scale_factor', type=int, default=2, help='Scale factor per progressive SR stage.')
    parser.add_argument('--refine_strength', type=float, default=1.0, help='Meissonic inpaint strength for active tokens.')
    parser.add_argument('--token_vae_scale_factor', type=int, default=16, help='Dry-run token grid scale factor.')
    parser.add_argument('--token_mask_threshold', type=float, default=0.25, help='Pixel-mask occupancy required to activate a token cell.')
    parser.add_argument('--lr_worse_margin', type=float, default=1.0, help='Local LR-error margin before remasking.')
    parser.add_argument('--reject_lr_l1_margin', type=float, default=1.0, help='Reject a candidate when LR L1 is worse than current by this margin.')
    parser.add_argument('--reject_lr_worse_ratio', type=float, default=0.50, help='Reject when this fraction of token cells get worse and LR L1 regresses.')
    parser.add_argument('--reject_active_lr_worse_ratio', type=float, default=0.50, help='Reject when this fraction of active token cells get locally worse and LR L1 regresses.')
    parser.add_argument('--disable_candidate_reject', action='store_true', help='Disable controller-level reject/rollback safety checks.')
    parser.add_argument('--enable_clip_score', action='store_true', help='Compute optional CLIP text/image similarities for candidates.')
    parser.add_argument('--clip_model_path', default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    parser.add_argument('--clip_score_device', default='cpu')
    parser.add_argument('--clip_text_weight', type=float, default=0.0, help='Subtract this weight times CLIP text similarity from candidate score.')
    parser.add_argument('--clip_image_weight', type=float, default=0.0, help='Subtract this weight times CLIP image-reference similarity from candidate score.')
    parser.add_argument('--semantic_refine_prompts', default=None, help='Semicolon-separated prompts that should increase active-mask priority.')
    parser.add_argument('--semantic_protect_prompts', default=None, help='Semicolon-separated prompts that should decrease active-mask priority.')
    parser.add_argument('--semantic_clip_model_path', default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    parser.add_argument('--semantic_clip_device', default='cpu')
    parser.add_argument('--semantic_grid_size', type=int, default=8)
    parser.add_argument('--semantic_batch_size', type=int, default=16)

    parser.add_argument('--run_projection_ablation', action='store_true', help='Write LR projection as ablation only.')
    parser.add_argument('--skip_consistency_projection', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--consistency_steps', type=int, default=2)
    parser.add_argument('--consistency_strength', type=float, default=None)
    parser.add_argument('--edit_strength', type=float, default=None)
    parser.add_argument('--mask_blur_radius', type=float, default=6.0)
    return parser.parse_args()


def ensure_repo_local(path: Path) -> Path:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if repo not in (resolved, *resolved.parents):
        raise ValueError(f'output path must stay inside repository: {repo}')
    return resolved


def load_meissonic_pipeline(model_path: str, device: str, dtype: str = 'float32'):
    import torch
    from diffusers import VQModel
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    from src.pipeline_inpaint import InpaintPipeline
    from src.scheduler import Scheduler
    from src.transformer import Transformer2DModel

    model = Transformer2DModel.from_pretrained(model_path, subfolder='transformer')
    vq_model = VQModel.from_pretrained(model_path, subfolder='vqvae')
    text_encoder = CLIPTextModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer')
    scheduler = Scheduler.from_pretrained(model_path, subfolder='scheduler')

    dtype_map = {
        'auto': torch.float16 if device.startswith('cuda') else torch.float32,
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
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


def _projection_ablation(args: argparse.Namespace, result: Image.Image, assets: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    if args.skip_consistency_projection or not args.run_projection_ablation:
        return {}

    observation = Image.open(args.input_image).convert('RGB')
    plan = assets['plan']
    lr_weight = args.consistency_strength
    if lr_weight is None:
        lr_weight = min(plan.lr_consistency_weight, 0.50) if plan.mode == 'sr' else plan.lr_consistency_weight
    edit_strength = args.edit_strength
    if edit_strength is None:
        if plan.mode == 'sr':
            edit_strength = 0.80 + 0.20 * plan.alpha
        elif plan.mode == 'detail':
            edit_strength = 0.75 + 0.20 * plan.alpha
        else:
            edit_strength = 0.60 + 0.35 * plan.alpha

    consistent, projection_metrics = observation_consistency_project(
        result,
        observation=observation,
        init_image=assets['init_image'],
        mask_image=assets['mask_image'],
        lr_weight=lr_weight,
        edit_strength=edit_strength,
        num_steps=args.consistency_steps,
        mask_blur_radius=args.mask_blur_radius,
    )
    consistent_path = output_dir / 'projection_ablation.png'
    consistent.save(consistent_path)
    projection_path = output_dir / 'projection_ablation_metrics.json'
    with projection_path.open('w', encoding='utf-8') as handle:
        json.dump(projection_metrics, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return {
        'projection_ablation_image': str(consistent_path),
        'projection_ablation_metrics': str(projection_path),
    }


def run_meissonic(args: argparse.Namespace, assets: dict, output_dir: Path) -> dict:
    import torch

    plan = assets['plan']
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device if args.device.startswith('cuda') else 'cpu').manual_seed(args.seed)

    pipe = load_meissonic_pipeline(args.model_path, args.device, dtype=args.dtype)
    result = pipe(
        prompt=plan.prompt,
        negative_prompt=args.negative_prompt,
        image=assets['init_image'],
        mask_image=assets['mask_image'],
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
        temperature=(max(0.01, plan.temperature), 0.0),
    ).images[0]

    output_path = output_dir / 'meissonic_refined.png'
    result.save(output_path)
    outputs = {'refined_image': str(output_path)}

    metrics = downsample_consistency_metrics(result, Image.open(args.input_image).convert('RGB'))
    metrics_path = output_dir / 'meissonic_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write('\n')

    outputs.update(_projection_ablation(args, result, assets, output_dir))
    return outputs


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def _clone_plan_for_stage(plan: AgentPlan, target_resolution: Tuple[int, int]) -> AgentPlan:
    stage_plan = AgentPlan.from_mapping(plan.to_dict())
    stage_plan.target_resolution = tuple(target_resolution)
    stage_plan.mode = 'sr'
    return stage_plan


def _progressive_stage_sizes(
    input_size: Tuple[int, int],
    target_size: Tuple[int, int],
    scale_factor: int,
) -> List[Tuple[int, int]]:
    scale = max(2, int(scale_factor))
    current_w, current_h = int(input_size[0]), int(input_size[1])
    target_w, target_h = int(target_size[0]), int(target_size[1])
    stages: List[Tuple[int, int]] = []
    while (current_w, current_h) != (target_w, target_h):
        if current_w < target_w:
            next_w = min(target_w, current_w * scale)
        else:
            next_w = target_w
        if current_h < target_h:
            next_h = min(target_h, current_h * scale)
        else:
            next_h = target_h
        if (next_w, next_h) == (current_w, current_h):
            break
        stages.append((next_w, next_h))
        current_w, current_h = next_w, next_h
    if not stages:
        stages.append((target_w, target_h))
    return stages


def _mask_after_rejected_candidate(masks: TokenMaskSet, next_masks: TokenMaskSet) -> TokenMaskSet:
    proposed_new_commit = next_masks.commit & ~masks.commit
    return TokenMaskSet(
        known=masks.known,
        active=next_masks.active | proposed_new_commit,
        commit=masks.commit,
        outpaint=masks.outpaint,
    )


def run_token_rerank_sr(
    args: argparse.Namespace,
    input_image: Image.Image,
    plan,
    output_dir: Path,
) -> Dict[str, Any]:
    assets = build_refinement_assets(
        input_image,
        plan,
        output_dir,
        outpaint_margin_ratio=args.outpaint_margin_ratio,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        mask_strategy=args.mask_strategy,
        semantic_refine_prompts=args.semantic_refine_prompts,
        semantic_protect_prompts=args.semantic_protect_prompts,
        semantic_clip_model_path=args.semantic_clip_model_path,
        semantic_clip_device=args.semantic_clip_device,
        semantic_grid_size=args.semantic_grid_size,
        semantic_batch_size=args.semantic_batch_size,
    )
    x_base_path = output_dir / 'x_base_hr.png'
    assets['init_image'].save(x_base_path)

    masks = build_initial_token_masks(
        assets['mask_image'],
        plan.target_resolution,
        vae_scale_factor=args.token_vae_scale_factor,
        token_threshold=args.token_mask_threshold,
    )
    round0_dir = output_dir / 'round_00'
    save_token_masks_npz(round0_dir / 'token_masks.npz', masks)
    token_mask_to_pixel_mask(masks.active, plan.target_resolution).save(round0_dir / 'active_token_mask.png')

    summary: Dict[str, Any] = {
        'mode': 'token_rerank_sr',
        'output_dir': str(output_dir),
        'dry_run': bool(args.dry_run or not args.run_meissonic),
        'run_meissonic': bool(args.run_meissonic),
        'x_base_hr': str(x_base_path),
        'assets': assets['paths'],
        'controller_diagnostics': assets['diagnostics'],
        'initial_token_masks': mask_metadata(masks),
        'score_weights': ScoreWeights(
            clip_text=args.clip_text_weight,
            clip_image=args.clip_image_weight,
        ).__dict__,
        'rounds': [],
    }

    if args.dry_run or not args.run_meissonic:
        _write_summary(output_dir / 'run_summary.json', summary)
        return summary

    pipe = load_meissonic_pipeline(args.model_path, args.device, dtype=args.dtype)
    editor = MeissonicTokenEditor(pipe)
    clip_scorer = None
    if args.enable_clip_score:
        clip_scorer = CLIPScorer(args.clip_model_path, device=args.clip_score_device)
    current_image = assets['init_image']
    z_base = editor.encode_image_tokens(current_image)
    z_current = z_base
    generated_candidate_used = False

    actual_shape = tuple(z_base.shape[-2:])
    if actual_shape != masks.shape:
        masks = build_initial_token_masks(
            assets['mask_image'],
            plan.target_resolution,
            vae_scale_factor=editor.vae_scale_factor,
            token_threshold=args.token_mask_threshold,
        )
        summary['initial_token_masks'] = mask_metadata(masks)

    for round_id in range(max(0, int(args.rounds))):
        if not bool(masks.active.any()):
            summary['rounds'].append({'round': round_id + 1, 'stopped': 'no_active_tokens'})
            break

        round_dir = output_dir / f'round_{round_id + 1:02d}'
        round_dir.mkdir(parents=True, exist_ok=True)
        save_token_masks_npz(round_dir / 'token_masks.npz', masks, z_base=z_base, z_current=z_current)
        token_mask_to_pixel_mask(masks.active, current_image.size).save(round_dir / 'active_token_mask.png')

        candidates: List[Image.Image] = []
        candidate_tokens: List[Any] = []
        candidate_multimodal: List[Dict[str, Any]] = []
        seeds: List[Optional[int]] = []
        for candidate_id in range(max(1, int(args.candidate_k))):
            seed = seed_for_candidate(args.seed, round_id, candidate_id)
            result = editor.refine(
                current_image,
                masks.active,
                prompt=plan.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                temperature=(max(0.01, plan.temperature), 0.0),
                strength=args.refine_strength,
                seed=seed,
            )
            candidate_path = round_dir / f'candidate_{candidate_id:02d}.png'
            result.image.save(candidate_path)
            candidates.append(result.image)
            candidate_tokens.append(result.tokens)
            metrics = naturalness_proxy(result.image)
            if clip_scorer is not None:
                metrics.update(clip_scorer.score(result.image, prompt=plan.prompt, reference=current_image).to_dict())
            candidate_multimodal.append(metrics)
            seeds.append(seed)

        score_weights = ScoreWeights(
            clip_text=args.clip_text_weight,
            clip_image=args.clip_image_weight,
        )
        best_id, scores = score_candidates(
            candidates,
            current_image,
            input_image,
            masks,
            seeds,
            weights=score_weights,
            multimodal_metrics=candidate_multimodal,
        )
        write_scores(round_dir / 'candidate_scores.json', scores, weights=score_weights)
        with (round_dir / 'candidate_multimodal_metrics.json').open('w', encoding='utf-8') as handle:
            json.dump(
                {
                    'enabled_clip_score': bool(args.enable_clip_score),
                    'clip_model_path': args.clip_model_path if args.enable_clip_score else None,
                    'metrics': candidate_multimodal,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write('\n')

        best_image = candidates[best_id]
        best_tokens = candidate_tokens[best_id]
        current_lr_l1 = lr_l1(current_image, input_image)
        current_lr_grad_l1 = lr_grad_l1(current_image, input_image)
        next_masks, mask_update = update_masks_after_round(
            masks,
            previous_image=current_image,
            best_image=best_image,
            observation=input_image,
            candidate_tokens=candidate_tokens,
            best_tokens=best_tokens,
            lr_worse_margin=args.lr_worse_margin,
        )

        best_score = scores[best_id]
        rejected = False
        reject_reason = None
        if not args.disable_candidate_reject:
            lr_regression = best_score.lr_l1 - current_lr_l1
            if lr_regression > args.reject_lr_l1_margin:
                rejected = True
                reject_reason = f'lr_l1_regression:{lr_regression:.6f}>{args.reject_lr_l1_margin:.6f}'
            elif mask_update['lr_worse_ratio'] > args.reject_lr_worse_ratio and best_score.lr_l1 > current_lr_l1:
                rejected = True
                reject_reason = (
                    f'lr_worse_ratio:{mask_update["lr_worse_ratio"]:.6f}'
                    f'>{args.reject_lr_worse_ratio:.6f}'
                )
            elif (
                mask_update.get('active_lr_worse_ratio', 0.0) > args.reject_active_lr_worse_ratio
                and best_score.lr_l1 > current_lr_l1
            ):
                rejected = True
                reject_reason = (
                    f'active_lr_worse_ratio:{mask_update["active_lr_worse_ratio"]:.6f}'
                    f'>{args.reject_active_lr_worse_ratio:.6f}'
                )

        round_summary = {
            'round': round_id + 1,
            'best_candidate': int(best_id),
            'best_seed': seeds[best_id],
            'best_score': best_score.total,
            'current_lr_l1': current_lr_l1,
            'current_lr_grad_l1': current_lr_grad_l1,
            'candidate_lr_l1': best_score.lr_l1,
            'candidate_lr_grad_l1': best_score.lr_grad_l1,
            'accepted': not rejected,
            'reject_reason': reject_reason,
            'used_as_current': True,
            'committed': not rejected,
            'mask_update': mask_update,
        }
        summary['rounds'].append(round_summary)

        current_image = best_image
        z_current = best_tokens
        generated_candidate_used = True
        if rejected:
            # Rejection means "do not commit these tokens", not "use bicubic as
            # the main result". Keep the generated image on the SR path while
            # remasking the proposed stable tokens for the next round.
            masks = _mask_after_rejected_candidate(masks, next_masks)
        else:
            masks = next_masks

    final_path = output_dir / 'final_hr.png'
    current_image.save(final_path)
    final_metrics = downsample_consistency_metrics(current_image, input_image)
    summary['final_hr'] = str(final_path)
    summary['final_source'] = 'best_generated_candidate' if generated_candidate_used else 'x_base_hr_no_candidate'
    summary['final_metrics'] = final_metrics
    summary['final_token_masks'] = mask_metadata(masks)

    if args.run_projection_ablation and not args.skip_consistency_projection:
        summary.update(_projection_ablation(args, current_image, assets, output_dir))

    _write_summary(output_dir / 'run_summary.json', summary)
    return summary


def run_progressive_token_rerank_sr(
    args: argparse.Namespace,
    input_image: Image.Image,
    plan,
    output_dir: Path,
) -> Dict[str, Any]:
    stage_sizes = _progressive_stage_sizes(
        input_image.size,
        plan.target_resolution,
        scale_factor=args.progressive_scale_factor,
    )
    summary: Dict[str, Any] = {
        'mode': 'progressive_token_rerank_sr',
        'output_dir': str(output_dir),
        'dry_run': bool(args.dry_run or not args.run_meissonic),
        'run_meissonic': bool(args.run_meissonic),
        'input_size': list(input_image.size),
        'target_resolution': list(plan.target_resolution),
        'progressive_scale_factor': int(args.progressive_scale_factor),
        'stage_sizes': [list(size) for size in stage_sizes],
        'score_reference': 'previous_stage_image',
        'original_lr_reference': 'input_image',
        'score_weights': ScoreWeights(
            clip_text=args.clip_text_weight,
            clip_image=args.clip_image_weight,
        ).__dict__,
        'stages': [],
    }

    pipe = None
    editor = None
    clip_scorer = None
    if args.run_meissonic and not args.dry_run:
        pipe = load_meissonic_pipeline(args.model_path, args.device, dtype=args.dtype)
        editor = MeissonicTokenEditor(pipe)
        if args.enable_clip_score:
            clip_scorer = CLIPScorer(args.clip_model_path, device=args.clip_score_device)

    current_stage_input = input_image.convert('RGB')
    generated_candidate_used = False
    final_token_masks = None

    for stage_idx, stage_size in enumerate(stage_sizes, start=1):
        stage_dir = output_dir / f'stage_{stage_idx:02d}_{stage_size[0]}x{stage_size[1]}'
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_plan = _clone_plan_for_stage(plan, stage_size)
        stage_plan.mask_policy = args.mask_strategy

        assets = build_refinement_assets(
            current_stage_input,
            stage_plan,
            stage_dir,
            outpaint_margin_ratio=args.outpaint_margin_ratio,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            mask_strategy=args.mask_strategy,
            semantic_refine_prompts=args.semantic_refine_prompts,
            semantic_protect_prompts=args.semantic_protect_prompts,
            semantic_clip_model_path=args.semantic_clip_model_path,
            semantic_clip_device=args.semantic_clip_device,
            semantic_grid_size=args.semantic_grid_size,
            semantic_batch_size=args.semantic_batch_size,
        )
        stage_base_path = stage_dir / 'x_base_stage.png'
        assets['init_image'].save(stage_base_path)

        masks = build_initial_token_masks(
            assets['mask_image'],
            stage_plan.target_resolution,
            vae_scale_factor=args.token_vae_scale_factor,
            token_threshold=args.token_mask_threshold,
        )
        round0_dir = stage_dir / 'round_00'
        save_token_masks_npz(round0_dir / 'token_masks.npz', masks)
        token_mask_to_pixel_mask(masks.active, stage_plan.target_resolution).save(round0_dir / 'active_token_mask.png')

        stage_summary: Dict[str, Any] = {
            'stage': stage_idx,
            'input_size': list(current_stage_input.size),
            'target_resolution': list(stage_plan.target_resolution),
            'stage_input': str(stage_dir / 'stage_input.png'),
            'x_base_stage': str(stage_base_path),
            'assets': assets['paths'],
            'controller_diagnostics': assets['diagnostics'],
            'initial_token_masks': mask_metadata(masks),
            'rounds': [],
        }
        current_stage_input.save(stage_dir / 'stage_input.png')

        if args.dry_run or not args.run_meissonic:
            stage_output = assets['init_image']
            stage_output_path = stage_dir / 'stage_final.png'
            stage_output.save(stage_output_path)
            stage_summary['stage_final'] = str(stage_output_path)
            stage_summary['stage_final_source'] = 'x_base_stage_dry_run'
            stage_summary['stage_consistency_metrics'] = downsample_consistency_metrics(
                stage_output,
                current_stage_input,
            )
            stage_summary['original_lr_metrics'] = downsample_consistency_metrics(stage_output, input_image)
            stage_summary['final_token_masks'] = mask_metadata(masks)
            summary['stages'].append(stage_summary)
            current_stage_input = stage_output
            final_token_masks = masks
            continue

        assert editor is not None
        current_image = assets['init_image']
        z_base = editor.encode_image_tokens(current_image)
        z_current = z_base
        actual_shape = tuple(z_base.shape[-2:])
        if actual_shape != masks.shape:
            masks = build_initial_token_masks(
                assets['mask_image'],
                stage_plan.target_resolution,
                vae_scale_factor=editor.vae_scale_factor,
                token_threshold=args.token_mask_threshold,
            )
            stage_summary['initial_token_masks'] = mask_metadata(masks)

        for round_id in range(max(0, int(args.rounds))):
            if not bool(masks.active.any()):
                stage_summary['rounds'].append({'round': round_id + 1, 'stopped': 'no_active_tokens'})
                break

            round_dir = stage_dir / f'round_{round_id + 1:02d}'
            round_dir.mkdir(parents=True, exist_ok=True)
            save_token_masks_npz(round_dir / 'token_masks.npz', masks, z_base=z_base, z_current=z_current)
            token_mask_to_pixel_mask(masks.active, current_image.size).save(round_dir / 'active_token_mask.png')

            candidates: List[Image.Image] = []
            candidate_tokens: List[Any] = []
            candidate_multimodal: List[Dict[str, Any]] = []
            seeds: List[Optional[int]] = []
            for candidate_id in range(max(1, int(args.candidate_k))):
                seed = seed_for_candidate(args.seed, stage_idx * 100 + round_id, candidate_id)
                result = editor.refine(
                    current_image,
                    masks.active,
                    prompt=stage_plan.prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    temperature=(max(0.01, stage_plan.temperature), 0.0),
                    strength=args.refine_strength,
                    seed=seed,
                )
                candidate_path = round_dir / f'candidate_{candidate_id:02d}.png'
                result.image.save(candidate_path)
                candidates.append(result.image)
                candidate_tokens.append(result.tokens)
                metrics = naturalness_proxy(result.image)
                if clip_scorer is not None:
                    metrics.update(clip_scorer.score(result.image, prompt=stage_plan.prompt, reference=current_image).to_dict())
                candidate_multimodal.append(metrics)
                seeds.append(seed)

            score_weights = ScoreWeights(
                clip_text=args.clip_text_weight,
                clip_image=args.clip_image_weight,
            )
            best_id, scores = score_candidates(
                candidates,
                current_image,
                current_stage_input,
                masks,
                seeds,
                weights=score_weights,
                multimodal_metrics=candidate_multimodal,
            )
            write_scores(round_dir / 'candidate_scores.json', scores, weights=score_weights)
            with (round_dir / 'candidate_multimodal_metrics.json').open('w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'enabled_clip_score': bool(args.enable_clip_score),
                        'clip_model_path': args.clip_model_path if args.enable_clip_score else None,
                        'metrics': candidate_multimodal,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
                handle.write('\n')

            best_image = candidates[best_id]
            best_tokens = candidate_tokens[best_id]
            current_ref_l1 = lr_l1(current_image, current_stage_input)
            current_ref_grad_l1 = lr_grad_l1(current_image, current_stage_input)
            current_original_l1 = lr_l1(current_image, input_image)
            next_masks, mask_update = update_masks_after_round(
                masks,
                previous_image=current_image,
                best_image=best_image,
                observation=current_stage_input,
                candidate_tokens=candidate_tokens,
                best_tokens=best_tokens,
                lr_worse_margin=args.lr_worse_margin,
            )

            best_score = scores[best_id]
            rejected = False
            reject_reason = None
            if not args.disable_candidate_reject:
                ref_regression = best_score.lr_l1 - current_ref_l1
                if ref_regression > args.reject_lr_l1_margin:
                    rejected = True
                    reject_reason = f'stage_ref_l1_regression:{ref_regression:.6f}>{args.reject_lr_l1_margin:.6f}'
                elif mask_update['lr_worse_ratio'] > args.reject_lr_worse_ratio and best_score.lr_l1 > current_ref_l1:
                    rejected = True
                    reject_reason = (
                        f'stage_ref_worse_ratio:{mask_update["lr_worse_ratio"]:.6f}'
                        f'>{args.reject_lr_worse_ratio:.6f}'
                    )
                elif (
                    mask_update.get('active_lr_worse_ratio', 0.0) > args.reject_active_lr_worse_ratio
                    and best_score.lr_l1 > current_ref_l1
                ):
                    rejected = True
                    reject_reason = (
                        f'active_stage_ref_worse_ratio:{mask_update["active_lr_worse_ratio"]:.6f}'
                        f'>{args.reject_active_lr_worse_ratio:.6f}'
                    )

            round_summary = {
                'round': round_id + 1,
                'best_candidate': int(best_id),
                'best_seed': seeds[best_id],
                'best_score': best_score.total,
                'current_stage_ref_l1': current_ref_l1,
                'current_stage_ref_grad_l1': current_ref_grad_l1,
                'candidate_stage_ref_l1': best_score.lr_l1,
                'candidate_stage_ref_grad_l1': best_score.lr_grad_l1,
                'current_original_lr_l1': current_original_l1,
                'candidate_original_lr_l1': lr_l1(best_image, input_image),
                'accepted': not rejected,
                'reject_reason': reject_reason,
                'used_as_current': True,
                'committed': not rejected,
                'mask_update': mask_update,
            }
            stage_summary['rounds'].append(round_summary)

            current_image = best_image
            z_current = best_tokens
            generated_candidate_used = True
            if rejected:
                masks = _mask_after_rejected_candidate(masks, next_masks)
            else:
                masks = next_masks

        stage_output_path = stage_dir / 'stage_final.png'
        current_image.save(stage_output_path)
        stage_summary['stage_final'] = str(stage_output_path)
        stage_summary['stage_final_source'] = 'best_generated_candidate'
        stage_summary['stage_consistency_metrics'] = downsample_consistency_metrics(current_image, current_stage_input)
        stage_summary['original_lr_metrics'] = downsample_consistency_metrics(current_image, input_image)
        stage_summary['final_token_masks'] = mask_metadata(masks)
        summary['stages'].append(stage_summary)
        current_stage_input = current_image
        final_token_masks = masks

    final_path = output_dir / 'final_hr.png'
    current_stage_input.save(final_path)
    summary['final_hr'] = str(final_path)
    summary['final_source'] = 'best_generated_candidate' if generated_candidate_used else 'x_base_stage_no_candidate'
    summary['final_metrics'] = downsample_consistency_metrics(current_stage_input, input_image)
    if final_token_masks is not None:
        summary['final_token_masks'] = mask_metadata(final_token_masks)

    if args.run_projection_ablation and not args.skip_consistency_projection:
        last_assets = summary['stages'][-1]['assets'] if summary.get('stages') else {}
        summary['projection_ablation_skipped'] = {
            'reason': 'progressive mode keeps projection disabled; run single-stage ablation for projection outputs',
            'last_stage_assets': last_assets,
        }

    _write_summary(output_dir / 'run_summary.json', summary)
    return summary


def main() -> int:
    args = parse_args()
    output_dir = ensure_repo_local(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(args.input_image).convert('RGB')
    requested_mode = args.mode
    token_modes = {'token_rerank_sr', 'progressive_token_rerank_sr', 'cascaded_token_rerank_sr'}
    controller_mode = 'sr' if requested_mode in token_modes else requested_mode
    if args.plan_json:
        plan = load_plan(Path(args.plan_json))
    else:
        plan = derive_agent_plan(
            args.prompt,
            target_resolution=args.target_resolution,
            mode=controller_mode,
            alpha=args.alpha,
            outpaint_direction=args.outpaint_direction,
        )
    plan.mask_policy = args.mask_strategy

    if requested_mode == 'token_rerank_sr':
        summary = run_token_rerank_sr(args, input_image, plan, output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if requested_mode in {'progressive_token_rerank_sr', 'cascaded_token_rerank_sr'}:
        summary = run_progressive_token_rerank_sr(args, input_image, plan, output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    assets = build_refinement_assets(
        input_image,
        plan,
        output_dir,
        outpaint_margin_ratio=args.outpaint_margin_ratio,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        mask_strategy=args.mask_strategy,
        semantic_refine_prompts=args.semantic_refine_prompts,
        semantic_protect_prompts=args.semantic_protect_prompts,
        semantic_clip_model_path=args.semantic_clip_model_path,
        semantic_clip_device=args.semantic_clip_device,
        semantic_grid_size=args.semantic_grid_size,
        semantic_batch_size=args.semantic_batch_size,
    )

    summary = {
        'output_dir': str(output_dir),
        'dry_run': args.dry_run,
        'run_meissonic': args.run_meissonic,
        'projection_default': 'disabled; use --run_projection_ablation',
        'assets': assets['paths'],
        'diagnostics': assets['diagnostics'],
    }

    if args.run_meissonic and not args.dry_run:
        summary.update(run_meissonic(args, assets, output_dir))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
