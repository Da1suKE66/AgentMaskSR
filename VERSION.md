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
