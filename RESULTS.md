# Results

This file records command results, artifacts, metrics, and sync status for AgentMaskSR.

## Remote Safety Policy

All remote project outputs must stay under:

```text
/home/ma-user/workspace/llc/AgentMaskSR
```

The project conda runtime is:

```text
/cache/llc/SR
```

## 2026-05-01 GitHub Sync

Status:

```text
initial upstream Meissonic baseline pushed
```

Target repository:

```text
https://github.com/Da1suKE66/AgentMaskSR
```

Notes:

- Remote `lsh-temp` clone succeeded.
- Remote HTTPS push failed because GitHub credentials were not available in the non-interactive SSH session.
- Local temporary clone push succeeded.

## 2026-05-01 Environment Setup

Requested environment path:

```text
/cache/llc/SR
```

Activation helper:

```bash
source envs/agentsr_cache_env.sh
```

The helper sets:

- `CONDA_PKGS_DIRS=/cache/llc/SR-pkgs`
- `PIP_CACHE_DIR=/cache/llc/SR-pip-cache`
- `HF_HOME=/home/ma-user/workspace/llc/AgentMaskSR/.hf_cache`
- `MPLCONFIGDIR=/home/ma-user/workspace/llc/AgentMaskSR/.mplconfig`

## 2026-05-01 Controller Dry-Run Outputs

Smoke-test command:

```bash
python tools/agent_mask_sr.py \
  --input_image assets/inpaint/0eKR4M2uuL8.jpg \
  --output_dir outputs/smoke_controller \
  --prompt "faithful super-resolution" \
  --mode sr \
  --target_resolution 512x512 \
  --dry_run
```

Outputs:

```text
outputs/smoke_controller/init_observation.png
outputs/smoke_controller/mask_refine.png
outputs/smoke_controller/agent_plan.json
outputs/smoke_controller/controller_metrics.json
```

Result:

```text
py_compile passed for agentsr/controller.py and tools/agent_mask_sr.py
dry-run controller generation passed
```

Recorded metrics:

```text
mse_downsample_vs_lr = 70.5918197631836
psnr_downsample_vs_lr = 29.643259832940526
masked_pixel_ratio = 0.315521240234375
target_resolution = 512x512
tile_grid = [{"x0": 0, "y0": 0, "x1": 512, "y1": 512}]
```

The dry run validates the controller and mask policy without downloading Meissonic checkpoints.

## 2026-05-01 Full Meissonic Pipeline Smoke Run

Status:

```text
completed
```

Environment:

```text
/cache/llc/SR
python 3.10.20
torch 2.8.0+cu128
torchvision 0.23.0
transformers 4.56.2
diffusers 0.35.1
accelerate 1.10.1
```

CUDA validation:

```text
torch.cuda.is_available() = True
GPU = NVIDIA A100-SXM4-80GB
```

Downloaded Hugging Face weights into the repository-local cache:

```text
/home/ma-user/workspace/llc/AgentMaskSR/.hf_cache
```

Models loaded:

```text
MeissonFlow/Meissonic
laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

The first attempted 512x512 run failed in Meissonic's rotary-position path:

```text
RuntimeError: The size of tensor a (333) must match the size of tensor b (1101)
```

The first 1024x1024 run completed with fp16, but the decoded image was all black. The wrapper was then changed to default to `float32`, matching the official Meissonic scripts.

Successful command:

```bash
python tools/agent_mask_sr.py \
  --input_image assets/inpaint/0eKR4M2uuL8.jpg \
  --output_dir outputs/first_pipeline_run_1024_fp32 \
  --prompt "faithful super-resolution with clean texture detail" \
  --mode sr \
  --target_resolution 1024x1024 \
  --alpha 0.35 \
  --run_meissonic \
  --steps 4 \
  --guidance_scale 7.0 \
  --seed 66 \
  --dtype float32
```

Outputs:

```text
outputs/first_pipeline_run_1024_fp32/init_observation.png
outputs/first_pipeline_run_1024_fp32/mask_refine.png
outputs/first_pipeline_run_1024_fp32/agent_plan.json
outputs/first_pipeline_run_1024_fp32/controller_metrics.json
outputs/first_pipeline_run_1024_fp32/meissonic_refined.png
outputs/first_pipeline_run_1024_fp32/meissonic_metrics.json
```

Controller metrics:

```text
mode = sr
alpha = 0.35
target_resolution = 1024x1024
masked_pixel_ratio = 0.3082866668701172
mse_downsample_vs_lr = 48.346221923828125
psnr_downsample_vs_lr = 31.287178196100704
```

Refined output check:

```text
size = 1024x1024
RGB mean = [115.91, 110.99, 96.93]
RGB extrema = [(0, 255), (0, 249), (0, 255)]
```

Downsample consistency after Meissonic refinement:

```text
mse_downsample_vs_lr = 1477.1201171875
psnr_downsample_vs_lr = 16.436645479455237
```

Interpretation:

The first complete pipeline is now operational and produces a non-empty image. The low PSNR after Meissonic refinement shows the next research step clearly: add stronger observation-consistency scoring or token-level rejection so Meissonic's visual prior cannot drift too far from the LR observation.

## 2026-05-01 Observation Consistency Projection

Status:

```text
completed
```

Added a deterministic post-refinement projection in `agentsr/controller.py`:

1. Downsample the HR Meissonic candidate to the original LR size.
2. Compute the low-frequency residual against the LR observation.
3. Upsample the residual to HR and subtract it from the candidate.
4. Re-apply the controller mask so protected black-mask regions move back toward `init_observation.png`.

New output files:

```text
outputs/first_pipeline_run_1024_consistency/meissonic_consistent.png
outputs/first_pipeline_run_1024_consistency/consistency_projection_metrics.json
```

Successful integrated command:

```bash
python tools/agent_mask_sr.py \
  --input_image assets/inpaint/0eKR4M2uuL8.jpg \
  --output_dir outputs/first_pipeline_run_1024_consistency \
  --prompt "faithful super-resolution with clean texture detail" \
  --mode sr \
  --target_resolution 1024x1024 \
  --alpha 0.35 \
  --run_meissonic \
  --steps 4 \
  --guidance_scale 7.0 \
  --seed 66 \
  --dtype float32
```

Projection parameters:

```text
projection_steps = 2
projection_lr_weight = 0.5
projection_edit_strength = 0.87
projection_mask_blur_radius = 6.0
```

Metrics before projection:

```text
mse_downsample_vs_lr = 1477.1201171875
psnr_downsample_vs_lr = 16.436645479455237
```

Metrics after projection:

```text
mse_downsample_vs_lr = 120.35453796386719
psnr_downsample_vs_lr = 27.32617890889089
psnr_gain_db = 10.889533429435652
```

Non-copy check:

```text
mean_abs(meissonic_refined - init_observation) = [18.69, 18.47, 19.02]
mean_abs(meissonic_consistent - init_observation) = [3.42, 3.41, 3.49]
mean_abs(meissonic_consistent - meissonic_refined) = [15.27, 15.07, 15.53]
```

Interpretation:

The projection strongly improves LR consistency while retaining a controlled amount of Meissonic edit signal. It is still pixel-space, not token-level rejection, so the next method step is to move this residual score into candidate token acceptance or scheduler callbacks.
