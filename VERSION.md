# Version Log

## v0.1.0-upstream-meissonic - 2026-05-01

### Scope

Import the official Meissonic codebase as the visual masked generative backbone for AgentMaskSR.

### Paths

Remote repository:

```text
/home/ma-user/workspace/llc/AgentMaskSR
```

Conda environment:

```text
/cache/llc/SR
```

GitHub repository:

```text
https://github.com/Da1suKE66/AgentMaskSR
```

### Baseline

Upstream source:

```text
https://github.com/viiika/Meissonic
```

Upstream commit:

```text
982de37 Update README with new badges and paper links
```

## v0.2.0-agent-controller-scaffold - 2026-05-01

### Scope

Add a training-free AgentMaskSR controller around the frozen Meissonic backbone.

### Added

- `agentsr/controller.py`
- `tools/agent_mask_sr.py`
- `envs/agentsr_cache_env.sh`
- project tracking Markdown files

### Capabilities

- Structured agent plan generation.
- Adaptive spatial-frequency mask generation.
- SR/detail/outpaint mode control through `alpha`.
- LR downsample consistency metrics.
- Tile-grid metadata for later high-resolution sparse decoding.
- Optional call into Meissonic `InpaintPipeline`.

### Limitations

- The current agent layer is a deterministic local fallback, not a VLM/API planner.
- Token-level freeze/re-mask hooks are planned but not yet wired directly into the scheduler.
- Full Meissonic inference requires model dependencies and checkpoint availability in `/cache/llc/SR`.

### Validation

Validated on `lsh-temp` with the conda environment at `/cache/llc/SR`:

```bash
python -m py_compile agentsr/controller.py tools/agent_mask_sr.py
python tools/agent_mask_sr.py --input_image assets/inpaint/0eKR4M2uuL8.jpg --output_dir outputs/smoke_controller --prompt "faithful super-resolution" --mode sr --target_resolution 512x512 --dry_run
```

Dry-run result:

```text
masked_pixel_ratio = 0.315521240234375
psnr_downsample_vs_lr = 29.643259832940526
```

## v0.3.0-full-pipeline-smoke - 2026-05-01

### Scope

Install Meissonic inference dependencies, download Hugging Face weights, and run the complete AgentMaskSR pipeline once.

### Runtime

Conda environment:

```text
/cache/llc/SR
```

Key packages:

```text
torch 2.8.0+cu128
torchvision 0.23.0
transformers 4.56.2
diffusers 0.35.1
accelerate 1.10.1
```

Weights cache:

```text
/home/ma-user/workspace/llc/AgentMaskSR/.hf_cache
```

### Code Fixes

- Explicit `--mode` now has priority over instruction keyword inference.
- `tools/agent_mask_sr.py` defaults to `--dtype float32`.
- `--dtype auto|float32|float16|bfloat16` is available for later speed/quality testing.

### First Complete Output

Command:

```bash
python tools/agent_mask_sr.py --input_image assets/inpaint/0eKR4M2uuL8.jpg --output_dir outputs/first_pipeline_run_1024_fp32 --prompt "faithful super-resolution with clean texture detail" --mode sr --target_resolution 1024x1024 --alpha 0.35 --run_meissonic --steps 4 --guidance_scale 7.0 --seed 66 --dtype float32
```

Generated image:

```text
outputs/first_pipeline_run_1024_fp32/meissonic_refined.png
```

Validation:

```text
image size = 1024x1024
RGB mean = [115.91, 110.99, 96.93]
RGB extrema = [(0, 255), (0, 249), (0, 255)]
```

Current limitation:

```text
psnr_downsample_vs_lr after refinement = 16.436645479455237 dB
```

The pipeline runs end to end, but stronger observation consistency is needed next.
