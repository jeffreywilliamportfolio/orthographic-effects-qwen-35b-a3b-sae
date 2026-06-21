# 2026-06-17 Standardized Replication Package

This folder holds the standardized June 2026 evidence package backing the preprint
*Orthographic Perturbations in Qwen: A Replicated SAE Case Study with a
Tokenizer-Equivalence Audit Protocol*. It supersedes the earlier May exploratory
runs (`../5-12-26`, `../5-14-26`, `../5-15-26`) with a controlled four-family,
twelve-variant matrix and deterministic replication.

## Contents

| Path | What it is |
|---|---|
| `qwen_sae_replication_extension_20260617/` | Primary Qwen-Scope SAE run: prompt manifests, generated text, residual/SAE TopK metrics, capture stats, replication-stability table, run scripts, logs, environment metadata, model/SAE file hashes, `SHA256SUMS`. |
| `qwen_sae_standardized_20260617/` | Companion standardized Qwen-Scope run (earlier pass reproduced by the replication extension). |
| `provider_behavior_secondary/` | Secondary behavioral context only — an OpenAI standardized run and the OpenRouter Anthropic content-filter probe. These are **not** Qwen mechanistic evidence; they are kept separate by design. |
| `manifests/` | Package-level provenance: `data_provenance_manifest.tsv`, `experiment_index.tsv`, `file_inventory.tsv`, `preservation_actions.tsv`, `MANIFEST.sha256`. |
| `docs/` | Findings, confound checklist, provenance/manifest schema, and reruns-needed notes. |
| `code/` | Original analysis and figure scripts. |
| `figures/` | Generated figures and figure support tables. |

## Study design

- Model: `Qwen/Qwen3.5-35B-A3B-Base`
- SAE: `Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50` (Qwen-Scope residual, TopK-50)
- Four prompt families x twelve canonical variants (48 canonical prompts), plus
  84 targeted deterministic control rows.
- Captures at layers 14, 15, 16, 24, 25, 26; final prompt token and generated
  tokens 1, 8, 16, 32, 64.
- Decoding: greedy deterministic, `max_new_tokens=160`, across three prompt-order
  replication passes. A small low-temperature variance probe used seeds 1001,
  1002, 1003 (temperature 0.2, top-p 0.95, top-k 50).
- Environment: Python 3.10.12, Torch 2.12.0+cu130, CUDA 13.0.

## Notes on this release

- **Weights excluded.** Model and SAE weights are not included; repository
  identifiers and file hashes are retained in each run's `metadata/` and
  `SHA256SUMS` for verification.
- **`sae_topk_rows.tsv` is stored gzipped** in the replication-extension run
  (the uncompressed file exceeds GitHub's 100 MB limit). Restore with
  `gunzip sae_topk_rows.tsv.gz`; the entry in `SHA256SUMS` hashes the
  uncompressed file.
- **Blinding key withheld.** `blinded_adjudication/key.tsv` (the unblinding map)
  is deliberately not published so the blinded adjudication stays valid until
  human scoring is complete. `outputs_blinded.tsv` and `rubric.md` are included.
