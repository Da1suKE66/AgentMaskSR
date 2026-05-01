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
