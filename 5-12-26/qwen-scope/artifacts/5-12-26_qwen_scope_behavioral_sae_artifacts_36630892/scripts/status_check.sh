#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/qwen-scope/5-12-26
MODEL="$ROOT/models/Qwen3.5-35B-A3B-Base"
SAE="$ROOT/saes/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
. "$ROOT/.venv/bin/activate"
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "python_version=$(python --version 2>&1)"
python - <<'PY'
import torch
print(f"torch_version={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"gpu_{i}_name={torch.cuda.get_device_name(i)}")
PY
echo "model_directory_size=$(du -sh "$MODEL" | cut -f1)"
echo "sae_directory_size=$(du -sh "$SAE" | cut -f1)"
echo "safetensor_shard_count=$(find "$MODEL" -name '*.safetensors' -type f | wc -l)"
echo "sae_layer_file_count=$(find "$SAE" -name 'layer*.sae.pt' -type f | wc -l)"
echo "model_file_count=$(find "$MODEL" -type f ! -path '*/.cache/*' | wc -l)"
echo "sae_file_count=$(find "$SAE" -type f ! -path '*/.cache/*' | wc -l)"
echo "disk_usage_workspace=$(du -sh "$ROOT" | cut -f1)"
echo "disk_free=$(df -h /workspace | awk 'NR==2{print $4}')"
echo "disk_line=$(df -h /workspace | tail -n 1)"
