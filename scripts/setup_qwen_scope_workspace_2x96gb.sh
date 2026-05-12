#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/qwen-scope/5-11-26
MODEL_REPO=Qwen/Qwen3.5-35B-A3B-Base
SAE_REPO=Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50
MODEL_DIR="$ROOT/models/Qwen3.5-35B-A3B-Base"
SAE_DIR="$ROOT/saes/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
HF_HOME_DIR="$ROOT/.hf_home"
VENV="$ROOT/.venv"
PROV="$ROOT/provenance"
MAN="$ROOT/manifests"
LOGS="$ROOT/logs"
SMOKE_ROOT="$ROOT/smoke-runs"

mkdir -p \
  "$ROOT/models" "$ROOT/saes" "$ROOT/scripts" "$ROOT/prompts" \
  "$ROOT/hidden_states" "$ROOT/sae_outputs" "$LOGS" "$PROV" "$MAN" \
  "$SMOKE_ROOT/pre_migration_2x5090" \
  "$SMOKE_ROOT/migration_validation_2x96gb" \
  "$HF_HOME_DIR"

LOG="$LOGS/setup_qwen_scope_workspace_2x96gb.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== QWEN-SCOPE 2x96GB SETUP START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat > "$PROV/instance_migration_2x96gb_20260511.txt" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
workspace_root=$ROOT
source_instance_id=36555651
source_instance_label=qwen-scope-5-11-26-2x5090-1tb
failed_target_instance_id=36562807
failed_target_offer=4x RTX 5090 offer 35676693
failed_target_outcome=created but remained stopped/loading; destroyed before setup
target_instance_id=36563002
target_instance_label=qwen-scope-5-11-26-2x96gb-smoke-segregated
target_hardware=2 x RTX PRO 6000 WS / Blackwell Max-Q, 97887 MiB each
target_reason=higher VRAM headroom to avoid OOM; fallback after 4x5090 provider did not start
model_repo=$MODEL_REPO
model_dir=$MODEL_DIR
sae_repo=$SAE_REPO
sae_dir=$SAE_DIR
smoke_run_root=$SMOKE_ROOT
phase=Transformers/PyTorch residual-stream hidden states + Qwen-Scope SAE feature analysis; not GGUF, llama.cpp, Hauhau, or router capture
EOF

cat > "$MAN/workspace_manifest.md" <<EOF
# Qwen-Scope 2026-05-11 Workspace

This workspace is for Hugging Face Transformers/PyTorch residual-stream hidden-state capture and Qwen-Scope SAE feature analysis.

It is explicitly not the old GGUF / llama.cpp / router-capture phase.

- Root: \`$ROOT\`
- Instance: \`36563002\`
- Hardware: 2 x RTX PRO 6000 WS / Blackwell Max-Q, 97887 MiB each
- Source instance retained until migration verification: \`36555651\`
- Failed unavailable 4x5090 reservation cleaned up: \`36562807\`
- Model repo: \`$MODEL_REPO\`
- Model path: \`$MODEL_DIR\`
- SAE repo: \`$SAE_REPO\`
- SAE path: \`$SAE_DIR\`
- Smoke-run organizer: \`$SMOKE_ROOT\`

Smoke and pre-full-experiment artifacts must stay under \`smoke-runs/\` or be mirrored there with provenance.
EOF

if [ -f "$ROOT/.env.hf" ]; then
  chmod 600 "$ROOT/.env.hf"
  set -a
  # shellcheck disable=SC1090
  source "$ROOT/.env.hf"
  set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN missing from $ROOT/.env.hf" >&2
  exit 2
fi

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Installing python venv support before creating workspace venv"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv python3-venv

rm -rf "$VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch transformers accelerate safetensors huggingface_hub numpy pandas einops tqdm hf_transfer hf_xet

python - <<'PY' | tee /workspace/qwen-scope/5-11-26/provenance/python_environment_snapshot_initial_2x96gb.txt
import json, platform, sys
pkgs = ["torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy", "pandas", "einops", "tqdm"]
versions = {}
for name in pkgs:
    mod = __import__(name)
    versions[name] = getattr(mod, "__version__", "unknown")
import torch
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "package_versions": versions,
    "cuda": {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    },
}, indent=2, sort_keys=True))
PY

if ! python - <<'PY'
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "Initial PyTorch wheel did not detect CUDA; installing CUDA 12.8 wheel." | tee "$PROV/torch_cuda_fix_2x96gb_20260511.txt"
  python -m pip freeze > "$PROV/pip_freeze_before_torch_fix_2x96gb.txt"
  python -m pip uninstall -y \
    torch cuda-bindings cuda-toolkit \
    nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
    nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
    nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx || true
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
fi

python -m pip check
python -m pip freeze > "$PROV/pip_freeze_2x96gb.txt"

python - <<'PY' | tee /workspace/qwen-scope/5-11-26/provenance/python_environment_snapshot_final_2x96gb.txt
import json, platform, sys
pkgs = ["torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy", "pandas", "einops", "tqdm"]
versions = {}
for name in pkgs:
    mod = __import__(name)
    versions[name] = getattr(mod, "__version__", "unknown")
import torch
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "package_versions": versions,
    "cuda": {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    },
}, indent=2, sort_keys=True))
PY

echo "--- listing Hugging Face repo files ---"
python - <<PY
from huggingface_hub import list_repo_files
for name, repo in [("base_model", "$MODEL_REPO"), ("sae", "$SAE_REPO")]:
    files = list_repo_files(repo, token="${HF_TOKEN}")
    out = "$MAN/" + name + "_hf_repo_files.txt"
    open(out, "w").write("\\n".join(files) + "\\n")
    print(f"{name}: {repo}: {len(files)} files listed -> {out}")
PY

echo "--- downloading full base model snapshot ---"
python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="$MODEL_REPO",
    local_dir="$MODEL_DIR",
    token="${HF_TOKEN}",
    local_dir_use_symlinks=False,
)
print(path)
PY

echo "--- downloading full Qwen-Scope SAE snapshot ---"
python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="$SAE_REPO",
    local_dir="$SAE_DIR",
    token="${HF_TOKEN}",
    local_dir_use_symlinks=False,
)
print(path)
PY

echo "--- writing manifests ---"
find "$MODEL_DIR" -type f -printf '%P\t%s bytes\n' | sort > "$MAN/base_model_file_manifest.tsv"
find "$SAE_DIR" -type f -printf '%P\t%s bytes\n' | sort > "$MAN/sae_file_manifest.tsv"
find "$SAE_DIR" -maxdepth 2 -printf '%y\t%P\t%s bytes\n' | sort > "$MAN/sae_find_layout.txt"
find "$MODEL_DIR" -type f -size -100M -print0 | sort -z | xargs -0 sha256sum > "$MAN/base_model_sha256_small_files.txt"
find "$SAE_DIR" -type f -size -100M -print0 | sort -z | xargs -0 sha256sum > "$MAN/sae_sha256_small_files.txt"

{
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model_repo=$MODEL_REPO"
  echo "model_dir=$MODEL_DIR"
  echo "model_du=$(du -sh "$MODEL_DIR" | awk '{print $1}')"
  echo "model_file_count=$(find "$MODEL_DIR" -type f | wc -l)"
  echo "model_safetensor_shards=$(find "$MODEL_DIR" -maxdepth 1 -name 'model.safetensors-*.safetensors' | wc -l)"
  echo "sae_repo=$SAE_REPO"
  echo "sae_dir=$SAE_DIR"
  echo "sae_du=$(du -sh "$SAE_DIR" | awk '{print $1}')"
  echo "sae_file_count=$(find "$SAE_DIR" -type f | wc -l)"
  echo "sae_layer_files=$(find "$SAE_DIR" -maxdepth 1 -name 'layer*.sae.pt' | wc -l)"
  echo "workspace_du=$(du -sh "$ROOT" | awk '{print $1}')"
  df -h "$ROOT"
} | tee "$PROV/download_completion_manifest_2x96gb.txt"

cat > "$ROOT/scripts/status_check.sh" <<'EOF'
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
EOF
chmod +x "$ROOT/scripts/status_check.sh"
"$ROOT/scripts/status_check.sh" | tee "$PROV/status_check_after_setup_2x96gb.txt"

echo "=== QWEN-SCOPE 2x96GB SETUP DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
