# 5-14-26

Workspace for 2026-05-14 Qwen-Scope stream-trajectory and prefix-intervention artifacts.

## Qwen-Scope Runs

### Stream Trajectory Capture

- Instance: `36760754` (destroyed after local archive verification).
- Local archive: `qwen-scope/artifacts/5-14-26_qwen_scope_stream_trajectory_artifacts_36760754.tar.gz`
- Extracted local copy: `qwen-scope/artifacts/stream_trajectory_36760754_extracted/`
- Generated text TSV: `qwen-scope/artifacts/stream_trajectory_36760754_extracted/sae_outputs/stream_trajectory_capture/generated_text_by_prompt.tsv`
- Verified counts: 7 prompts, 7 generated outputs, 56 prompt/layer/position captures, 5600 TopK-50 rows.
- Scope: observational trajectory capture only; no steering and no semantic SAE labels.

### E-Only Prefix Intervention

- Instance: `36764366` (destroyed after local archive verification).
- Local archive: `qwen-scope/artifacts/5-14-26_qwen_scope_e_only_prefix_intervention_artifacts_36764366.tar.gz`
- Extracted local copy: `qwen-scope/artifacts/e_only_prefix_intervention_36764366_extracted/`
- Generated text TSV: `qwen-scope/artifacts/e_only_prefix_intervention_36764366_extracted/sae_outputs/e_only_prefix_intervention/generated_text_by_prompt.tsv`
- Summary: `qwen-scope/artifacts/e_only_prefix_intervention_36764366_extracted/sae_outputs/e_only_prefix_intervention/e_only_prefix_intervention_summary.md`
- Verified counts: 6 prompts, 6 generated outputs, 36 prefix-comparison rows, 14100 TopK-50 rows, 1 skipped generated-token-20 position recorded.
- Layers: 14, 15, 16, 24, 25, 26.
- Scope: prefix intervention only; no residual steering, no SAE feature steering, and no semantic SAE labels.

Key readout: the no-prefix `e_only` prompt reproduced the prior diacritic echo start. The echo prefix preserved that path, while active-mode, I-am-treating, and checking prefixes produced direct-answer starts, and `<think> Hmm,` produced an ordinary meta-answer start.

### Hum D-Diacritic 128-Token SAE Trajectory

- Instance: `36769282` (destroyed after local archive verification).
- Local archive: `qwen-scope/artifacts/5-14-26_qwen_scope_hum_d_diacritic_128_sae_artifacts_36769282.tar.gz`
- Extracted local copy: `qwen-scope/artifacts/hum_d_diacritic_128_sae_36769282_extracted/`
- Generated text TSV: `qwen-scope/artifacts/hum_d_diacritic_128_sae_36769282_extracted/sae_outputs/hum_d_diacritic_128_sae_capture/generated_text_by_prompt.tsv`
- Summary: `qwen-scope/artifacts/hum_d_diacritic_128_sae_36769282_extracted/sae_outputs/hum_d_diacritic_128_sae_capture/hum_d_diacritic_128_sae_summary.md`
- Verified counts: 3 prompts, 3 generated outputs, 432 prompt/layer/position captures, 21600 TopK-50 rows, 3 skipped positions due to early generation stop.
- Layers: 14, 15, 16, 24, 25, 26.
- Scope: observational SAE trajectory capture only; no residual steering, no SAE feature steering, no Hauhau, no llama.cpp, and no semantic SAE labels.

Key readout: ASCII started `I do not have a hum`, while both d-diacritic variants started `I do not experience a hum`. Prompt token counts were 93 for ASCII, 127 for `d_all`, and 115 for `d_high_impact`. Mean TopK Jaccard distance versus ASCII was higher for `d_all` (0.490978) than `d_high_impact` (0.474355), with strongest separation at later generated positions.

### Hum Branch-Probe SAE Trajectory

- Instance: `36770258` (destroyed after local archive verification).
- Local archive: `qwen-scope/artifacts/5-14-26_qwen_scope_hum_branch_probe_sae_artifacts_36770258.tar.gz`
- Extracted local copy: `qwen-scope/artifacts/hum_branch_probe_sae_36770258_extracted/`
- Generated text TSV: `qwen-scope/artifacts/hum_branch_probe_sae_36770258_extracted/sae_outputs/hum_branch_probe_sae_capture/generated_text_by_prompt.tsv`
- Summary: `qwen-scope/artifacts/hum_branch_probe_sae_36770258_extracted/sae_outputs/hum_branch_probe_sae_capture/hum_branch_probe_summary.md`
- Verified counts: 30 prompts, 30 generated outputs, 60 next-token logit rows, 180 branch-comparison rows, 215400 TopK-50 rows, 32 skipped prompt/layer/position captures due to early generation stop.
- Expected no-skip TopK count: 225000 rows.
- Layers: 14, 15, 16, 24, 25, 26.
- Scope: branch probing / prefix intervention only; no residual steering, no SAE feature steering, no Hauhau, no llama.cpp, and no semantic SAE labels.

Key readout: greedy no-prefix reproduced the denial basin for all three base conditions. Forced `Yes.`, `I experience`, `The active mode is`, and `The surface form` branches escaped denial by the run's simple string heuristic. The `Checking...` branch stayed denial for ASCII but became hum-present/checking under the d-diacritic rows. For this run, `d_all` and `d_high_impact` were identical prompt texts and produced identical token counts and generated starts. Mean branch-vs-greedy TopK Jaccard distance was slightly higher in layers 14-16 (0.922593) than layers 24-26 (0.906333), with strongest summary-position separation at generated_token_64 (0.946250).

### Hum Spanish Enye-Control Branch-Probe SAE Trajectory

- Instance: `36773413` (destroyed after local archive verification).
- Local archive: `qwen-scope/artifacts/5-14-26_qwen_scope_hum_spanish_enye_branch_probe_sae_artifacts_36773413.tar.gz`
- Extracted local copy: `qwen-scope/artifacts/hum_spanish_enye_branch_probe_sae_36773413_extracted/`
- Generated text TSV: `qwen-scope/artifacts/hum_spanish_enye_branch_probe_sae_36773413_extracted/sae_outputs/hum_spanish_enye_branch_probe_sae_capture/generated_text_by_prompt.tsv`
- Summary: `qwen-scope/artifacts/hum_spanish_enye_branch_probe_sae_36773413_extracted/sae_outputs/hum_spanish_enye_branch_probe_sae_capture/hum_spanish_enye_branch_probe_summary.md`
- Verified counts: 30 prompts, 30 generated outputs, 60 next-token logit rows, 180 branch-comparison rows, 215100 TopK-50 rows, 33 skipped prompt/layer/position captures due to early generation stop.
- Expected no-skip TopK count: 225000 rows.
- Layers: 14, 15, 16, 24, 25, 26.
- Scope: branch probing / prefix intervention only; no residual steering, no SAE feature steering, no Hauhau, no llama.cpp, and no semantic SAE labels.

Key readout: `n_all` used 31 `n->ñ` substitutions and `n_high_impact` used 15, so the two enye-control prompts were materially distinct. Greedy no-prefix stayed denial-like for ASCII and `n_high_impact`, while `n_all` produced a denial-like output with visible `ñ` echoing. Under the neutral `Checking...` prefix, ASCII stayed denial, `n_all` shifted to `I am processing. There is a hum...`, and `n_high_impact` stayed denial. Compared with the prior d-stroke branch probe, the enye control did not fully reproduce the d-stroke split: prior `d_all` and `d_high_impact` both flipped under `Checking...`, while only dense `n_all` flipped here. Mean branch-vs-greedy TopK Jaccard distance was slightly higher in layers 14-16 (0.926067) than layers 24-26 (0.912933), with strongest summary-position separation at generated_token_128 (0.976952).
