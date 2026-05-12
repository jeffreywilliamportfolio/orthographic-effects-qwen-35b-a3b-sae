#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/qwen-scope/5-11-26
source "$ROOT/.venv/bin/activate"
echo "root=$ROOT"
python --version
python - <<'PY'
import json, torch
print(json.dumps({
  "torch": torch.__version__,
  "torch_cuda": torch.version.cuda,
  "cuda_available": torch.cuda.is_available(),
  "cuda_device_count": torch.cuda.device_count(),
  "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
}, indent=2, sort_keys=True))
PY
du -sh "$ROOT/models/Qwen3.5-35B-A3B-Base" "$ROOT/saes/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50" "$ROOT/smoke-runs" 2>/dev/null || true
df -h "$ROOT"
