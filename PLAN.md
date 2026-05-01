# Plan

This document tracks the AgentMaskSR research plan and the current state of the Meissonic-based implementation.

## Remote Safety Policy

All remote project writes must stay inside:

```text
/home/ma-user/workspace/llc/AgentMaskSR
```

The conda runtime requested for this project is the only allowed repository-external write location:

```text
/cache/llc/SR
```

Do not modify files outside `/home/ma-user/workspace/llc/AgentMaskSR` except for creating or updating that conda environment. Keep caches under repository-local paths or `/cache/llc/SR-*`.

## Current Repository Status

The repository is forked from official Meissonic:

```text
https://github.com/viiika/Meissonic
```

The initial upstream code has been pushed to:

```text
https://github.com/Da1suKE66/AgentMaskSR
```

Current additions:

- `agentsr/controller.py`: training-free agent plan, adaptive mask generation, outpaint canvas, consistency metrics, and tile grid.
- `tools/agent_mask_sr.py`: CLI that writes controller assets and optionally calls Meissonic inpaint refinement.
- `envs/agentsr_cache_env.sh`: reproducible activation script for `/cache/llc/SR`.

## Research Framing

AgentMaskSR turns a masked generative model from "generate an image" into:

> an observation-constrained image editor that selectively fills unknown high-frequency or expanded-region tokens while preserving the low-resolution observation.

Meissonic is used as the frozen visual masked generative backbone. The project contribution is the training-free controller around it:

1. Agent-guided plan generation.
2. Observation-consistent masked refinement.
3. Adaptive spatial-frequency unmasking.
4. Detail-outpaint mode control.
5. Tile/global sparse decoding plan.
6. Context reuse and early commit hooks for later implementation.

## Phase 1: Controller Dry Run

Objective:

Build the assets Meissonic needs before running expensive model inference.

Implemented behavior:

1. Parse user instruction into an `AgentPlan`.
2. Upsample LR input to target resolution.
3. For outpainting, place the upsampled observation into a larger target canvas.
4. Compute a deterministic gradient/variance detail map.
5. Create a binary mask where white means Meissonic may repaint and black means preserve.
6. Record LR downsample consistency and tile grid metadata.

Example:

```bash
python tools/agent_mask_sr.py \
  --input_image assets/inpaint/0eKR4M2uuL8.jpg \
  --output_dir outputs/agent_mask_sr_dryrun \
  --prompt "faithful super-resolution with clean texture detail" \
  --mode sr \
  --target_resolution 1024x1024 \
  --alpha 0.40 \
  --dry_run
```

## Phase 2: Meissonic Refinement

Objective:

Run the existing Meissonic `InpaintPipeline` on the controller assets:

```text
init_observation.png + mask_refine.png + prompt -> meissonic_refined.png
```

The first implementation reuses the repository's image-editing interface instead of changing Meissonic weights.

## Phase 3: Token-Level Control

Objective:

Move from pixel mask control to explicit token control:

- encode the target initialization with Meissonic VQ-VAE;
- freeze protected token positions;
- mask only high-frequency/outpaint token positions;
- rescore or reject updates that violate LR downsample consistency.

This is the key step for the paper story because it makes the controller explicitly token-level.

## Phase 4: High-Resolution Sparse Decoding

Objective:

Support 1K/2K outputs without full global decoding every step.

Planned modules:

- overlap tile grid;
- global anchors for low-frequency structure;
- early commit for stable tokens;
- boundary consistency for outpaint seams;
- per-tile metadata in `controller_metrics.json`.

## Current Next Step

After dependency installation is complete in `/cache/llc/SR`, run:

```bash
source envs/agentsr_cache_env.sh
python -m py_compile agentsr/controller.py tools/agent_mask_sr.py
python tools/agent_mask_sr.py --input_image assets/inpaint/0eKR4M2uuL8.jpg --output_dir outputs/smoke_controller --prompt "faithful super-resolution" --mode sr --target_resolution 512x512 --dry_run
```
