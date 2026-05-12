# Orthographic Effects in Qwen3.5-35B-A3B SAE Features

This repository contains the 2026-05-11/2026-05-12 Qwen-Scope sparse-autoencoder work for controlled orthographic perturbation probes on `Qwen/Qwen3.5-35B-A3B-Base`.

The work here is **Transformers/PyTorch residual-stream + Qwen-Scope SAE analysis**. It is not a `llama.cpp`, GGUF, router-capture, Hauhau, or steering run.

## Scope

Target base model:

`Qwen/Qwen3.5-35B-A3B-Base`

Target SAE:

`Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`

Analyzed layers (previously determined to be where Expert 114 had the largest effect):

- Layer 26, primary.
- Layer 14, comparison.

Main matrix:

- 5 matched prompt families.
- 6 perturbation variants.
- 5 prompt-boundary token positions.
- 2 layers.
- 300 prompt-position-layer residual captures.
- Official Qwen-Scope TopK-50 SAE encoding.

No semantic feature labels are claimed in this repo. Feature IDs are reported as evidence candidates only.

## Main Results

Start with:

- [`RESULTS.md`](RESULTS.md)
- [`5-12-26/qwen-scope/README.md`](5-12-26/qwen-scope/README.md)
- [`results/full_controlled_perturbation_matrix/aggregates/results_memo.md`](results/full_controlled_perturbation_matrix/aggregates/results_memo.md)
- [`results/full_controlled_perturbation_matrix/full_controlled_perturbation_matrix_summary.md`](results/full_controlled_perturbation_matrix/full_controlled_perturbation_matrix_summary.md)

The 5-12 behavioral-SAE alignment run found that `all_diacritics` maximized token inflation and SAE displacement, while `s_to_ṡ` produced the strongest auto-classified behavioral movement. In that Qwen run, the largest SAE displacement did not match the strongest behavioral movement.

The aggregate tables and plots are under:

`results/full_controlled_perturbation_matrix/aggregates/`

## Directory Layout

- `prompts/`: prompt matrices and seed banks.
- `scripts/`: setup, smoke tests, scans, matrix run, and aggregate postprocessing scripts.
- `results/`: SAE TopK outputs, generated text snippets, delta/Jaccard tables, summaries, aggregate tables, and plots.
- `manifests/`: model/SAE file manifests and workspace manifest.
- `provenance/`: timestamped run provenance.
- `logs/`: remote run logs.
- `smoke-validation/`: active migration-validation two-prompt outputs, excluding raw `.pt` tensors.
- `archive/`: teardown archive receipt only; the archive itself is not committed.
- `5-12-26/`: dated behavioral-SAE alignment run, including small verified artifacts and teardown provenance.

## Large Artifacts Not Included

This repository intentionally excludes:

- Hugging Face model weights.
- Qwen-Scope SAE checkpoint weights.
- residual `.pt` tensors.
- virtualenvs, caches, offload folders, and remote env files.

The model and SAE repo IDs plus file manifests are recorded under `manifests/` and `provenance/`.

## Reproducibility Notes

The 5-11 scripts assume the remote workspace layout used during that run:

`/workspace/qwen-scope/5-11-26`

The 5-12 scripts assume:

`/workspace/qwen-scope/5-12-26`

The scripts were archived to preserve the exact run logic. They may need path edits or a recreated workspace before rerun.

## Safety

No Hugging Face token, Vast API key, `.env`, private key, model weights, or SAE weights are included.
