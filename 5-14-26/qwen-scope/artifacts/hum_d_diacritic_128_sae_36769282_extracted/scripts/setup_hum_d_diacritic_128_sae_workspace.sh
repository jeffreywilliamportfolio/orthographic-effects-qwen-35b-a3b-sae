#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/qwen-scope/5-14-26
INSTANCE_ID=36769282
MODEL_REPO=Qwen/Qwen3.5-35B-A3B-Base
SAE_REPO=Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50
MODEL_PATH="$ROOT/models/Qwen3.5-35B-A3B-Base"
SAE_PATH="$ROOT/saes/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
PROV_DIR="$ROOT/provenance"
MANIFEST_DIR="$ROOT/manifests"
ENV_FILE="$ROOT/.env.hf"

mkdir -p "$ROOT"/{models,saes,prompts,scripts,outputs,sae_outputs,provenance,manifests,logs,.hf_home}
chmod 700 "$ROOT/.hf_home"
chmod 600 "$ENV_FILE"

export HF_TOKEN="$(cat "$ENV_FILE")"
export HF_HOME="$ROOT/.hf_home"
export HF_HUB_ENABLE_HF_TRANSFER=1

{
  date -u +"timestamp_utc=%Y-%m-%dT%H:%M:%SZ"
  printf 'instance_id=%s\n' "$INSTANCE_ID"
  printf 'workspace_root=%s\n' "$ROOT"
  printf 'purpose=%s\n' '5-14 hum d-diacritic 128-token SAE trajectory setup'
  printf 'model_repo=%s\n' "$MODEL_REPO"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'sae_repo=%s\n' "$SAE_REPO"
  printf 'sae_path=%s\n' "$SAE_PATH"
  printf 'confirmation=%s\n' 'setup only; no experiment run in this script'
} > "$PROV_DIR/setup_hum_d_diacritic_128_sae_workspace_20260514.txt"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip git rsync

python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch transformers accelerate safetensors "huggingface_hub[cli]" hf_transfer hf_xet numpy pandas einops tqdm

python - <<'PY'
import json, platform, subprocess, sys
import torch
snapshot = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
try:
    snapshot["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
except Exception as exc:
    snapshot["nvidia_smi_error"] = repr(exc)
print(json.dumps(snapshot, indent=2, sort_keys=True))
PY

python -m pip freeze > "$PROV_DIR/pip_freeze_hum_d_diacritic_128_sae_20260514.txt"
python - <<'PY' > "$PROV_DIR/python_environment_snapshot_hum_d_diacritic_128_sae_20260514.txt"
import json, platform, subprocess, sys
import torch
snapshot = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
try:
    snapshot["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
except Exception as exc:
    snapshot["nvidia_smi_error"] = repr(exc)
print(json.dumps(snapshot, indent=2, sort_keys=True))
PY

python - <<'PY'
from huggingface_hub import snapshot_download
import os
root = "/workspace/qwen-scope/5-14-26"
token = os.environ["HF_TOKEN"]
snapshot_download(
    repo_id="Qwen/Qwen3.5-35B-A3B-Base",
    local_dir=f"{root}/models/Qwen3.5-35B-A3B-Base",
    token=token,
)
snapshot_download(
    repo_id="Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50",
    local_dir=f"{root}/saes/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50",
    token=token,
)
PY

find "$MODEL_PATH" -type f -printf '%P\t%s\n' | sort > "$MANIFEST_DIR/base_model_file_manifest_hum_d_diacritic_128_sae.tsv"
find "$SAE_PATH" -type f -printf '%P\t%s\n' | sort > "$MANIFEST_DIR/sae_file_manifest_hum_d_diacritic_128_sae.tsv"
find "$SAE_PATH" -maxdepth 2 -type f | sort > "$MANIFEST_DIR/sae_find_layout_hum_d_diacritic_128_sae.txt"

{
  printf '%s\n\n' '# 5-14-26 Qwen-Scope Hum D-Diacritic 128 SAE Workspace'
  printf '%s\n\n' 'Fresh remote workspace for hum-prompt d-diacritic 128-token generation trajectory capture.'
  printf '%s\n\n' 'This is observational SAE trajectory capture only. It is not residual steering and not SAE feature steering. It uses Transformers/PyTorch selected-layer capture plus Qwen-Scope TopK-50 SAE encoding.'
  printf '%s\n' "- instance_id: \`$INSTANCE_ID\`"
  printf '%s\n' "- model_repo: \`$MODEL_REPO\`"
  printf '%s\n' "- model_path: \`$MODEL_PATH\`"
  printf '%s\n' "- sae_repo: \`$SAE_REPO\`"
  printf '%s\n' "- sae_path: \`$SAE_PATH\`"
  printf '%s\n' '- restrictions: no residual steering, no SAE feature steering, no Hauhau, no llama.cpp, no all-layer expansion, no semantic labels'
} > "$MANIFEST_DIR/workspace_manifest_hum_d_diacritic_128_sae.md"

{
  date -u +"timestamp_utc=%Y-%m-%dT%H:%M:%SZ"
  printf 'instance_id=%s\n' "$INSTANCE_ID"
  printf 'workspace_root=%s\n' "$ROOT"
  printf 'model_repo=%s\n' "$MODEL_REPO"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'model_size=%s\n' "$(du -sh "$MODEL_PATH" | awk '{print $1}')"
  printf 'model_file_count=%s\n' "$(find "$MODEL_PATH" -type f | wc -l)"
  printf 'model_safetensor_shard_count=%s\n' "$(find "$MODEL_PATH" -type f -name '*.safetensors' | wc -l)"
  printf 'sae_repo=%s\n' "$SAE_REPO"
  printf 'sae_path=%s\n' "$SAE_PATH"
  printf 'sae_size=%s\n' "$(du -sh "$SAE_PATH" | awk '{print $1}')"
  printf 'sae_file_count=%s\n' "$(find "$SAE_PATH" -type f | wc -l)"
  printf 'sae_layer_file_count=%s\n' "$(find "$SAE_PATH" -type f -name 'layer*.sae.pt' | wc -l)"
  printf 'confirmation=%s\n' 'No residual steering, no SAE feature steering, no Hauhau, no llama.cpp, no GGUF, and no experiment run was performed during setup.'
} > "$PROV_DIR/setup_hum_d_diacritic_128_sae_download_verification_20260514.txt"

df -h /workspace > "$PROV_DIR/disk_after_hum_d_diacritic_128_sae_downloads_20260514.txt"
echo "HUM_D_DIACRITIC_128_SAE_SETUP_COMPLETE"
