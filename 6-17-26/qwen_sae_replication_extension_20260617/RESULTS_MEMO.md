# Qwen SAE Replication Extension Results Memo

Run completed on 2026-06-17 using `Qwen/Qwen3.5-35B-A3B-Base` with `Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`.

## Execution

- Vast instance: `41345768`, RTX PRO 6000 Blackwell Server Edition, stopped after artifact pull.
- Model/SAE weights were downloaded remotely and excluded from the evidence package; file hashes are retained under `metadata/`.
- Environment: Python 3.10.12, `torch 2.12.0+cu130`, CUDA 13.0, `transformers 5.12.1`.
- One failed pre-model-load attempt is preserved in `logs/attempt1_failed_before_model_load.log`; it was caused by stale `torchvision/torchaudio` packages incompatible with the installed Torch build. Those packages were removed before the successful run.

## Artifact Counts

- Scheduled prompt executions: 300.
- Canonical deterministic replications: 144 rows, covering 48 canonical prompts across 3 deterministic orderings.
- Targeted deterministic controls: 84 rows.
- Low-temperature seed-variance rows: 72 rows, using seeds 1001, 1002, and 1003 at temperature 0.2, top_p 0.95, top_k 50.
- SAE TopK rows: 540,000.
- Capture-stat rows: 10,800.
- Residual/SAE metric rows: 9,792.
- Skipped capture positions: 0.
- Blinded adjudication export rows: 300.

## Validation Summary

- All TSV artifacts parse successfully with stable column counts.
- The 48 canonical prompt hashes match the original `qwen_sae_standardized_20260617` prompt manifest exactly.
- The original-order deterministic pass reproduces the original primary run:
  - Generated text exact matches: 48/48.
  - Rule-label matches: 48/48.
  - Residual L2 metric exact matches: 1,584/1,584.
  - SAE TopK Jaccard exact matches: 1,584/1,584.
  - Residual cosine max absolute difference: approximately 3.3e-7, consistent with floating-point/platform noise.
- Across the three deterministic canonical passes, generated text was stable for 48/48 canonical prompts.
- Deterministic metric stability across the three passes was effectively exact:
  - Residual L2 population std max: 0.
  - SAE TopK Jaccard population std max: approximately 1.1e-16.
- Token-count matched controls are near-exact: 40 rows, max absolute token-count delta 2, with 11 nonzero deltas.

## Bounded Interpretation

This extension strengthens the Qwen case-study skeleton by showing that the existing deterministic Qwen SAE/tokenization/control result is reproducible across repeated deterministic passes and minor prompt-order changes. It also adds manuscript-useful targeted controls and a low-temperature variance layer.

The behavioral labels remain rule-based and secondary. The blinded adjudication package is prepared but not yet scored, so behavioral claims should remain descriptive until adjudication is complete.
