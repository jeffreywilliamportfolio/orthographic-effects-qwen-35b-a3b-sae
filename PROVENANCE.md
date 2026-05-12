# Provenance

This package was prepared from the active Qwen-Scope workspace:

`/workspace/qwen-scope/5-11-26`

Active instance during the run:

`36563002`

Workspace phase:

Transformers/PyTorch residual-stream capture plus Qwen-Scope SAE TopK-50 encoding.

## Source Repositories

Base model:

`Qwen/Qwen3.5-35B-A3B-Base`

SAE:

`Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`

## Teardown Archive Receipt

The local teardown archive was verified before this publish folder was prepared.

Archive SHA256:

`3feea5cf1ab667c5cf0403b6f22c2323f520cab474c4ed44e3c4e7964b354ef7`

The archive itself is not committed. This repo contains the extracted small artifacts needed for audit and review.

## Excluded Artifacts

Excluded from git:

- model weights
- SAE checkpoint weights
- residual `.pt` tensors
- virtualenvs and caches
- remote `.env.hf`
- tar archives

## Timestamped Provenance

Run-specific provenance files are preserved under:

`provenance/`

The most relevant files are:

- `full_controlled_perturbation_matrix_20260511.txt`
- `full_controlled_perturbation_matrix_aggregates_20260511.txt`
- `active_2x96gb_teardown_archive_manifest_20260512.tsv`

