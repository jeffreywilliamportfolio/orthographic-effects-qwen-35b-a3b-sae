#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/qwen-scope/5-14-26
PROV_DIR="$ROOT/provenance"
MANIFEST_DIR="$ROOT/manifests"
MODEL_REPO=Qwen/Qwen3.5-35B-A3B-Base
SAE_REPO=Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50
MODEL_PATH="$ROOT/models/Qwen3.5-35B-A3B-Base"
SAE_PATH="$ROOT/saes/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"

mkdir -p "$PROV_DIR" "$MANIFEST_DIR"

find "$MODEL_PATH" -type f -printf '%P\t%s\n' | sort > "$MANIFEST_DIR/base_model_file_manifest.tsv"
find "$SAE_PATH" -type f -printf '%P\t%s\n' | sort > "$MANIFEST_DIR/sae_file_manifest.tsv"
find "$SAE_PATH" -maxdepth 2 -type f | sort > "$MANIFEST_DIR/sae_find_layout.txt"

{
  printf '%s\n\n' '# 5-14-26 Qwen-Scope Workspace'
  printf '%s\n\n' 'This is the fresh 5-14 Qwen-Scope workspace for Transformers/PyTorch residual-stream trajectory capture and Qwen-Scope SAE analysis.'
  printf '%s\n\n' 'It is observational only for this run: no steering, no Hauhau, no llama.cpp, no GGUF routing work, no all-layer expansion, and no semantic SAE feature labels.'
  printf '%s\n' "- model_repo: \`$MODEL_REPO\`"
  printf '%s\n' "- model_path: \`$MODEL_PATH\`"
  printf '%s\n' "- sae_repo: \`$SAE_REPO\`"
  printf '%s\n' "- sae_path: \`$SAE_PATH\`"
} > "$MANIFEST_DIR/workspace_manifest.md"

{
  date -u +"timestamp_utc=%Y-%m-%dT%H:%M:%SZ"
  printf 'instance_id=%s\n' "36760754"
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
  printf 'torch_snapshot=%s\n' "$PROV_DIR/python_environment_snapshot_20260514.txt"
  printf 'pip_freeze=%s\n' "$PROV_DIR/pip_freeze_20260514.txt"
  printf 'confirmation=%s\n' 'No Hauhau, no llama.cpp, no GGUF, no steering, and no experiment run was performed during setup.'
} > "$PROV_DIR/setup_download_verification_20260514.txt"

df -h /workspace > "$PROV_DIR/disk_after_downloads_20260514.txt"
echo "POST_DOWNLOAD_VERIFY_COMPLETE"
