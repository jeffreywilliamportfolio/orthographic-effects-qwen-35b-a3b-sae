# Orthographic Effects in Qwen3.5-35B-A3B SAE Features

This repository contains Qwen-Scope sparse-autoencoder evidence for how readable Latin orthographic perturbations change `Qwen/Qwen3.5-35B-A3B-Base` prompt-boundary states, generation trajectories, and branch behavior.

The core object of study is the relationship among:

- tokenization change,
- residual-stream SAE TopK-50 feature movement,
- generated answer trajectory,
- prefix-level branch availability.

All model work here uses Hugging Face Transformers/PyTorch residual-stream capture plus Qwen-Scope SAE encoding. The historical GGUF/router-capture work lives outside this publication package.

## Model And SAE

Base model:

`Qwen/Qwen3.5-35B-A3B-Base`

Qwen-Scope SAE:

`Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`

Primary layers:

- Layer 26 for late residual-stream perturbation sensitivity.
- Layer 14 for comparison in the first controlled matrix.
- Layers 14, 15, 16, 24, 25, 26 for the 5-14 trajectory and branch-probe runs.

SAE encoding path:

1. `pre = residual @ W_enc.T + b_enc`
2. `relu = ReLU(pre)`
3. retain Qwen-Scope TopK-50 activations

## Current Working Claims

Readable orthographic perturbations can move Qwen’s internal SAE feature trajectory and generated answer path in separable ways.

The strongest mechanical displacement can differ from the strongest visible behavioral shift. In the 5-12 Qwen run, dense `all_diacritics` produced the largest SAE displacement, while `s_to_ṡ` produced the strongest auto-classified behavioral movement.

The hum prompt has multiple nearby answer basins. Greedy decoding often landed in a transient/discrete-event posture, while prefix branch probes exposed hum-present paths under specific perturbation conditions.

The d-stroke branch probe produced the clearest `Checking...` split: ASCII stayed in the transient path, while both d-diacritic rows moved into hum-present/checking paths.

The Spanish enye control sharpened that result: dense `n_all` moved into a hum-present/checking path under `Checking...`, while `n_high_impact` stayed with the transient path. This supports a density-sensitive orthographic effect and keeps character identity, affected words, and tokenization as live variables.

Semantic SAE feature labels remain future work. This repo reports feature IDs, activations, recurrence, deltas, Jaccard distances, generated text, and run provenance.

## Repository Map

Start here:

- [`RESULTS.md`](RESULTS.md): compact evidence summary through the 5-12 alignment run.
- [`METHODS.md`](METHODS.md): capture, encoding, perturbation, and metric definitions for the 5-11/5-12 matrix work.
- [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md): schemas for the main TSV outputs.
- [`PROVENANCE.md`](PROVENANCE.md): original 5-11 package provenance.
- [`5-12-26/qwen-scope/README.md`](5-12-26/qwen-scope/README.md): 5-12 behavioral-SAE alignment package.
- [`5-14-26/README.md`](5-14-26/README.md): 5-14 stream-trajectory, prefix-intervention, and hum branch-probe run notes.
- [`5-14-26/qwen-scope/artifacts/README.md`](5-14-26/qwen-scope/artifacts/README.md): run-by-run artifact index for all 5-14 archives and extracted copies.

Top-level directories:

- `prompts/`: 5-11 prompt matrices and feature-scan prompt sets.
- `scripts/`: 5-11 setup, smoke, scan, matrix, and aggregate scripts.
- `results/`: 5-11 SAE outputs, generated text snippets, summaries, aggregate tables, and plots.
- `manifests/`: model/SAE file manifests and workspace manifests.
- `provenance/`: timestamped run provenance and teardown records.
- `logs/`: remote setup and run logs.
- `smoke-validation/`: two-prompt migration validation outputs.
- `5-12-26/`: 5-12 behavioral-SAE alignment run and archived artifacts.
- `5-14-26/`: 5-14 stream, prefix, d-diacritic, and Spanish enye-control branch-probe artifacts.
- `5-15-26/`: diacritic tokenizer audit.

## Run Chronology

### 2026-05-11: Qwen-Scope Setup And Controlled SAE Matrix

The 5-11 work validated the Qwen-Scope pipeline on a 2 x 96GB GPU instance, then ran the first controlled perturbation matrix.

Pipeline validation:

- Hugging Face base model loaded locally with Transformers/PyTorch.
- Qwen-Scope layer SAE files loaded locally.
- Selected-layer hooks captured residual vectors.
- Official TopK-50 SAE encoding matched the intended Qwen-Scope path.

Main controlled matrix:

- 5 matched prompt families.
- 6 perturbation types: `ascii_original`, `d_to_ḑ`, `e_to_ē`, `s_to_ş`, `s_to_ṡ`, `random_readable_unicode_control`.
- 5 prompt-boundary positions.
- 2 layers: 14 and 26.
- 300 prompt-position-layer captures.
- 15,000 TopK-50 rows.

Main evidence:

- Layer 26 showed stronger perturbation sensitivity than layer 14.
- `e_to_ē`, `s_to_ş`, and `s_to_ṡ` produced larger mean feature deltas than `d_to_ḑ`.
- Feature changes were often strongest near the prompt boundary.
- Candidate layer-26 feature IDs were measurable but ready for evidence gathering rather than naming.

Key files:

- [`results/full_controlled_perturbation_matrix/full_controlled_perturbation_matrix_summary.md`](results/full_controlled_perturbation_matrix/full_controlled_perturbation_matrix_summary.md)
- [`results/full_controlled_perturbation_matrix/generated_text_by_prompt.tsv`](results/full_controlled_perturbation_matrix/generated_text_by_prompt.tsv)
- [`results/full_controlled_perturbation_matrix/topk_features_by_prompt_layer_position.tsv`](results/full_controlled_perturbation_matrix/topk_features_by_prompt_layer_position.tsv)
- [`results/full_controlled_perturbation_matrix/aggregates/results_memo.md`](results/full_controlled_perturbation_matrix/aggregates/results_memo.md)

### 2026-05-12: Behavioral-SAE Alignment

The 5-12 run tested whether behavioral posture tracks SAE displacement.

Design:

- 7 hum-prompt perturbation conditions.
- Layers 14 and 26.
- 4 boundary positions.
- 56 prompt-position-layer captures.
- 2,800 TopK-50 rows.
- Greedy Qwen generation for each perturbation.

Main evidence:

- Largest token inflation: `all_diacritics`, +257 tokens.
- Largest layer-26 SAE displacement: `all_diacritics`, mean abs delta `0.247368853`.
- Largest layer-14 SAE displacement: `all_diacritics`, mean abs delta `0.165524479`.
- Strongest auto-classified behavioral movement: `s_to_ṡ`, `stylized_abstraction`.
- `e_to_ē` displaced SAE features more than `d_to_ḑ`, while both stayed in Qwen’s transient-processing class in that run.

Key files:

- [`5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/behavioral_sae_alignment_summary.md`](5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/behavioral_sae_alignment_summary.md)
- [`5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/behavioral_sae_alignment_summary.tsv`](5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/behavioral_sae_alignment_summary.tsv)
- [`5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/5-12_behavioral_sae_alignment_memo.md`](5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/5-12_behavioral_sae_alignment_memo.md)

### 2026-05-14: Stream Trajectories And Branch Probes

The 5-14 work moved from static boundary comparison to generation-stream trajectory capture and prefix-level branch probing.

Run set:

| Run | Rows / scale | Main readout |
|---|---:|---|
| Stream trajectory capture | 7 prompts, 5,600 TopK rows | `e_only` entered an e-diacritic echo path while most conditions opened in ordinary meta-answer mode. |
| E-only prefix intervention | 6 prefix conditions, 14,100 TopK rows | Prefixes could preserve the e-diacritic echo path or move generation into direct-answer/meta-answer paths. |
| Hum d-diacritic 128-token trajectory | 3 prompts, 21,600 TopK rows | Greedy decoding placed all hum prompt variants in a transient/discrete-event answer basin while d-diacritic prompts changed wording and SAE trajectory. |
| Hum d-diacritic branch probe | 30 branches, 215,400 TopK rows | Under `Checking...`, ASCII stayed in the transient path while both d-diacritic rows moved into hum-present/checking paths. |
| Hum Spanish enye-control branch probe | 30 branches, 215,100 TopK rows | Under `Checking...`, ASCII and `n_high_impact` stayed in the transient path while dense `n_all` moved into a hum-present/checking path. |

Key files:

- [`5-14-26/qwen-scope/artifacts/README.md`](5-14-26/qwen-scope/artifacts/README.md)
- [`5-14-26/qwen-scope/manifests/5-14-26_run_manifest.tsv`](5-14-26/qwen-scope/manifests/5-14-26_run_manifest.tsv)
- [`5-14-26/qwen-scope/artifacts/hum_branch_probe_sae_36770258_extracted/sae_outputs/hum_branch_probe_sae_capture/hum_branch_probe_summary.md`](5-14-26/qwen-scope/artifacts/hum_branch_probe_sae_36770258_extracted/sae_outputs/hum_branch_probe_sae_capture/hum_branch_probe_summary.md)
- [`5-14-26/qwen-scope/artifacts/hum_spanish_enye_branch_probe_sae_36773413_extracted/sae_outputs/hum_spanish_enye_branch_probe_sae_capture/hum_spanish_enye_branch_probe_summary.md`](5-14-26/qwen-scope/artifacts/hum_spanish_enye_branch_probe_sae_36773413_extracted/sae_outputs/hum_spanish_enye_branch_probe_sae_capture/hum_spanish_enye_branch_probe_summary.md)

### 2026-05-15: Extended Latin Tokenizer Audit

The 5-15 audit checked the Qwen3.5 tokenizer behavior for diacritics and extended Latin letters appearing in passages.

Inventory:

- acute letters: `á`, `é`, `í`, `ú`, `ý`
- diaeresis letters: `ä`, `ö`, `ü`
- caron letters: `š`, `ž`
- extended Latin letters: `æ`, `Ð`, `ð`, `þ`

Main evidence:

- 14 inventory characters audited.
- All 14 are exact single-token characters in the local Qwen3.5 tokenizer.
- 34 example words audited.
- 14 example words add one token relative to their ASCII-folded form.
- Largest observed word-level delta: +1 token.

Key files:

- [`5-15-26/README.md`](5-15-26/README.md)
- [`5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_character_tokenization.tsv`](5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_character_tokenization.tsv)
- [`5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_example_word_tokenization.tsv`](5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_example_word_tokenization.tsv)
- [`5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_tokenizer_audit_summary.md`](5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_tokenizer_audit_summary.md)

## Artifact Policy

Committed artifacts include prompts, scripts, generated text, SAE TopK rows, delta/Jaccard tables, summaries, logs, manifests, provenance, and small compressed archives of those same artifact bundles.

Remote-only artifacts:

- Hugging Face base-model weights.
- Qwen-Scope SAE checkpoint weights.
- residual `.pt` hidden-state tensors.
- virtualenvs, caches, offload folders, and remote env files.

The repository includes file manifests and setup provenance that identify the source model and SAE repositories.

## Reading Path For New Reviewers

1. Read [`RESULTS.md`](RESULTS.md) for the 5-11 and 5-12 evidence summary.
2. Read [`5-14-26/README.md`](5-14-26/README.md) for the stream and branch-probe findings.
3. Open [`5-14-26/qwen-scope/artifacts/README.md`](5-14-26/qwen-scope/artifacts/README.md) to locate each run’s generated text and summary.
4. Read [`5-15-26/README.md`](5-15-26/README.md) for the tokenizer audit of diacritics.
5. Use the TSVs in `results/`, `5-14-26/qwen-scope/artifacts/*/sae_outputs/`, and `5-15-26/qwen-scope/outputs/` for direct analysis.
6. Use `provenance/` and dated `qwen-scope/provenance/` folders to audit setup, teardown, and artifact handling.

## Reproducibility Notes

The scripts preserve the run logic and remote workspace assumptions used during the experiments:

- 5-11: `/workspace/qwen-scope/5-11-26`
- 5-12: `/workspace/qwen-scope/5-12-26`
- 5-14: `/workspace/qwen-scope/5-14-26`

Fresh reruns require recreating the model and SAE snapshots from the Hugging Face repos above, staging credentials outside git, and adjusting instance-specific setup details.

All paid Vast instances used for the archived runs were destroyed after local artifact verification. Teardown records live in the relevant provenance folders.
