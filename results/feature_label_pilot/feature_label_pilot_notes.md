# Feature Label Pilot Evidence Notes

Generated UTC: 2026-05-11T20:18:10.605853Z

Scope: evidence-only pilot for selected changed sparse features from the active 2x96GB two-prompt TopK-50 validation run.

No semantic labels are assigned here. Notes describe only TopK-50 presence, activation direction, and rank movement in the existing normal-hum vs d-stroke-hum comparison.

Delta source: `/workspace/qwen-scope/5-11-26/smoke-runs/migration_validation_2x96gb/sae_outputs/feature_delta_topk50.tsv`
Top-feature source: `/workspace/qwen-scope/5-11-26/smoke-runs/migration_validation_2x96gb/sae_outputs/top_features_by_condition.tsv`
Evidence TSV: `/workspace/qwen-scope/5-11-26/sae_outputs/feature_label_pilot/feature_label_pilot_evidence.tsv`

Selected pilot features:
- Layer 14: 9030, 30172, 12433, 3291, 28025
- Layer 26: 23977, 7108, 2722, 9745, 31784

Interpretation rules:
- Activation `0` with blank rank means the feature was not present in that condition's TopK-50 output for this validation run.
- `delta_from_normal_to_dstroke` is d-stroke activation minus normal activation.
- Notes are provisional evidence statements only, not labels.

| layer | feature_id | normal activation/rank | d-stroke activation/rank | delta | abs_delta | evidence note |
|---:|---:|---:|---:|---:|---:|---|
| 14 | 9030 | 0.137331 / 29 | 0.109591 / 47 | -0.0277399 | 0.0277399 | feature appears in both conditions with higher activation in normal; feature shifts rank between normal rank 29 and d-stroke rank 47 |
| 14 | 30172 | 0.127116 / 39 | 0 / not TopK-50 | -0.127116 | 0.127116 | feature appears only in normal TopK-50 |
| 14 | 12433 | 0.119017 / 46 | 0 / not TopK-50 | -0.119017 | 0.119017 | feature appears only in normal TopK-50 |
| 14 | 3291 | 0 / not TopK-50 | 0.118389 / 41 | 0.118389 | 0.118389 | feature appears only in d-stroke TopK-50 |
| 14 | 28025 | 0.115953 / 47 | 0 / not TopK-50 | -0.115953 | 0.115953 | feature appears only in normal TopK-50 |
| 26 | 23977 | 0.192175 / 30 | 0 / not TopK-50 | -0.192175 | 0.192175 | feature appears only in normal TopK-50 |
| 26 | 7108 | 0.17277 / 43 | 0 / not TopK-50 | -0.17277 | 0.17277 | feature appears only in normal TopK-50 |
| 26 | 2722 | 0.18328 / 35 | 0 / not TopK-50 | -0.18328 | 0.18328 | feature appears only in normal TopK-50 |
| 26 | 9745 | 0.173812 / 41 | 0 / not TopK-50 | -0.173812 | 0.173812 | feature appears only in normal TopK-50 |
| 26 | 31784 | 0 / not TopK-50 | 0.170044 / 40 | 0.170044 | 0.170044 | feature appears only in d-stroke TopK-50 |

Stop condition honored: no semantic labels, no steering, no full experiment expansion, no Hauhau, and no llama.cpp.
