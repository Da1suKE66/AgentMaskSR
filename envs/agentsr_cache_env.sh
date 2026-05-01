#!/usr/bin/env bash
set -euo pipefail

export AGENTMASKSR_ROOT="/home/ma-user/workspace/llc/AgentMaskSR"
export AGENTMASKSR_CONDA_PREFIX="/cache/llc/SR"
export CONDA_PKGS_DIRS="/cache/llc/SR-pkgs"
export PIP_CACHE_DIR="/cache/llc/SR-pip-cache"
export HF_HOME="/home/ma-user/workspace/llc/AgentMaskSR/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="/home/ma-user/workspace/llc/AgentMaskSR/.mplconfig"
export PYTHONNOUSERSITE=1

cd "${AGENTMASKSR_ROOT}"
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate "${AGENTMASKSR_CONDA_PREFIX}"
