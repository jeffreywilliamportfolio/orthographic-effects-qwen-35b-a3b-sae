#!/usr/bin/env python3
"""Download Qwen3.5-35B-A3B-Base and selected Qwen-Scope SAE layers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(os.environ.get("RUN_ROOT", "/workspace/qwen_sae_replication_extension_20260617"))
MODEL_REPO = "Qwen/Qwen3.5-35B-A3B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
SAE_LAYERS = [14, 15, 16, 24, 25, 26]


def main() -> None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "saes").mkdir(exist_ok=True)
    (ROOT / "metadata").mkdir(exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    model_dir = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
    sae_dir = ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"

    model_path = snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(model_dir),
        resume_download=True,
    )
    sae_patterns = ["*.json", "*.md"] + [f"layer{layer}.sae.pt" for layer in SAE_LAYERS]
    sae_path = snapshot_download(
        repo_id=SAE_REPO,
        local_dir=str(sae_dir),
        allow_patterns=sae_patterns,
        resume_download=True,
    )

    metadata = {
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "model_repo": MODEL_REPO,
        "model_path": model_path,
        "sae_repo": SAE_REPO,
        "sae_path": sae_path,
        "sae_layers": SAE_LAYERS,
        "hf_xet_high_performance": os.environ.get("HF_XET_HIGH_PERFORMANCE"),
    }
    (ROOT / "metadata" / "download_assets_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
