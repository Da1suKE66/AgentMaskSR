# History

This document records concrete AgentMaskSR file changes.

## Remote Safety Policy

Remote write operations must stay inside:

```text
/home/ma-user/workspace/llc/AgentMaskSR
```

The requested conda environment is installed at:

```text
/cache/llc/SR
```

Do not write code, logs, checkpoints, or experiment outputs outside the repository. Environment package/cache writes should use `/cache/llc/SR`, `/cache/llc/SR-pkgs`, or `/cache/llc/SR-pip-cache`.

## 2026-05-01 Upstream Import

Cloned official Meissonic into:

```text
/home/ma-user/workspace/llc/AgentMaskSR
```

Source:

```text
https://github.com/viiika/Meissonic
```

The remote machine could not push to GitHub through HTTPS because no interactive credentials were available. The same upstream commit was pushed from a local temporary clone to:

```text
https://github.com/Da1suKE66/AgentMaskSR
```

Initial upstream commit:

```text
982de37 Update README with new badges and paper links
```

## 2026-05-01 AgentMaskSR Controller Scaffold

Added:

- `agentsr/__init__.py`
- `agentsr/controller.py`
- `tools/agent_mask_sr.py`
- `envs/agentsr_cache_env.sh`
- `PLAN.md`
- `HISTORY.md`
- `RESULTS.md`
- `VERSION.md`
- `CHAT.md`

Updated:

- `README.md`
- `.gitignore`

Main implementation details:

1. `AgentPlan` stores the structured agent/controller policy.
2. `derive_agent_plan` maps user intent to mode, alpha, consistency weights, temperature, and outpaint directions.
3. `build_refinement_assets` creates:
   - `init_observation.png`;
   - `mask_refine.png`;
   - `agent_plan.json`;
   - `controller_metrics.json`.
4. `tools/agent_mask_sr.py` runs dry-run asset generation and can optionally call Meissonic `InpaintPipeline`.
5. `.gitignore` excludes runtime outputs, local model directories, and repository-local Hugging Face cache paths.

The implementation is training-free and leaves Meissonic weights unchanged.

## 2026-05-01 Remote Environment and Validation

Created the requested conda environment:

```text
/cache/llc/SR
```

Installed minimal controller dry-run dependencies:

```text
numpy
pillow
```

Full Meissonic inference dependencies were not installed from the upstream `requirements.txt` in this checkpoint because that file points to CUDA 12.4 wheels, while the known remote driver setup is safer with a CUDA 12.1 PyTorch stack.

Validation commands completed on `lsh-temp`:

```bash
python -m py_compile agentsr/controller.py tools/agent_mask_sr.py
python tools/agent_mask_sr.py --input_image assets/inpaint/0eKR4M2uuL8.jpg --output_dir outputs/smoke_controller --prompt "faithful super-resolution" --mode sr --target_resolution 512x512 --dry_run
```
