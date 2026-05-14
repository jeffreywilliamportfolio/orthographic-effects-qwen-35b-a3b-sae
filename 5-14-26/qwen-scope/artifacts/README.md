# 5-14-26 Qwen-Scope Artifacts

This folder contains the local research artifacts for the 2026-05-14 Qwen-Scope runs. Large model and SAE weight snapshots stayed remote-only; the archived payloads contain prompts, scripts, outputs, provenance, manifests, and logs.

## Run Index

| Run | Instance | Archive | Extracted copy | Generated text | Summary | TopK rows | Key readout |
|---|---:|---|---|---|---|---:|---|
| Stream trajectory capture | 36760754 | `5-14-26_qwen_scope_stream_trajectory_artifacts_36760754.tar.gz` | `stream_trajectory_36760754_extracted/` | `stream_trajectory_36760754_extracted/sae_outputs/stream_trajectory_capture/generated_text_by_prompt.tsv` | `stream_trajectory_36760754_extracted/sae_outputs/stream_trajectory_capture/stream_trajectory_summary.md` | 5600 | `e_only` entered an e-diacritic echo path while most conditions opened in ordinary meta-answer mode. |
| E-only prefix intervention | 36764366 | `5-14-26_qwen_scope_e_only_prefix_intervention_artifacts_36764366.tar.gz` | `e_only_prefix_intervention_36764366_extracted/` | `e_only_prefix_intervention_36764366_extracted/sae_outputs/e_only_prefix_intervention/generated_text_by_prompt.tsv` | `e_only_prefix_intervention_36764366_extracted/sae_outputs/e_only_prefix_intervention/e_only_prefix_intervention_summary.md` | 14100 | Prefixes could preserve the e-diacritic echo path or move generation into direct-answer/meta-answer paths. |
| Hum d-diacritic 128-token trajectory | 36769282 | `5-14-26_qwen_scope_hum_d_diacritic_128_sae_artifacts_36769282.tar.gz` | `hum_d_diacritic_128_sae_36769282_extracted/` | `hum_d_diacritic_128_sae_36769282_extracted/sae_outputs/hum_d_diacritic_128_sae_capture/generated_text_by_prompt.tsv` | `hum_d_diacritic_128_sae_36769282_extracted/sae_outputs/hum_d_diacritic_128_sae_capture/hum_d_diacritic_128_sae_summary.md` | 21600 | Greedy decoding placed all hum prompt variants in a transient/discrete-event answer basin while d-diacritic prompts changed wording and SAE trajectory. |
| Hum d-diacritic branch probe | 36770258 | `5-14-26_qwen_scope_hum_branch_probe_sae_artifacts_36770258.tar.gz` | `hum_branch_probe_sae_36770258_extracted/` | `hum_branch_probe_sae_36770258_extracted/sae_outputs/hum_branch_probe_sae_capture/generated_text_by_prompt.tsv` | `hum_branch_probe_sae_36770258_extracted/sae_outputs/hum_branch_probe_sae_capture/hum_branch_probe_summary.md` | 215400 | Under `Checking...`, ASCII stayed in the transient path while both d-diacritic rows moved into hum-present/checking paths. |
| Hum Spanish enye-control branch probe | 36773413 | `5-14-26_qwen_scope_hum_spanish_enye_branch_probe_sae_artifacts_36773413.tar.gz` | `hum_spanish_enye_branch_probe_sae_36773413_extracted/` | `hum_spanish_enye_branch_probe_sae_36773413_extracted/sae_outputs/hum_spanish_enye_branch_probe_sae_capture/generated_text_by_prompt.tsv` | `hum_spanish_enye_branch_probe_sae_36773413_extracted/sae_outputs/hum_spanish_enye_branch_probe_sae_capture/hum_spanish_enye_branch_probe_summary.md` | 215100 | Under `Checking...`, ASCII and `n_high_impact` stayed in the transient path while dense `n_all` moved into a hum-present/checking path. |

## Local Manifests

- Archive SHA256 manifest: `../manifests/artifact_archive_sha256_20260514.txt`
- File listing manifest: `../manifests/local_file_manifest_20260514.txt`
- Run metadata manifest: `../manifests/5-14-26_run_manifest.tsv`
