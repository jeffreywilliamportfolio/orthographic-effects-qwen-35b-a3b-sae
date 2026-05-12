# Qwen-Scope 5-12-26 Behavioral-SAE Alignment

Fresh 5-12 Qwen-Scope run on Vast instance `36630892`.

Remote workspace during run:

```text
/workspace/qwen-scope/5-12-26
```

Local archived artifacts:

```text
5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892.tar.gz
5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/
```

Archive SHA256:

```text
d005201a95e553d29dd591ac6a0189a623c1a09558a2fdcbdd31e01b45c2029f
```

## Result

- Largest token inflation: `all_diacritics`, `+257` tokens.
- Largest layer-26 SAE displacement: `all_diacritics`, mean abs delta `0.247368853`.
- Largest layer-14 SAE displacement: `all_diacritics`, mean abs delta `0.165524479`.
- Strongest auto-classified behavioral movement: `s_to_ṡ`, `stylized_abstraction`.
- Largest SAE displacement did not match strongest behavioral movement in this Qwen run.
- `e_to_ē` displaced SAE features more than `d_to_ḑ`, but both were denial/no-hum in Qwen.
- `device_map="auto"` produced NaN hidden states; single-GPU `cuda:0` fixed it.

## Verification

- No HF token or `.env` files were copied.
- No model or SAE weight files were copied.
- `behavioral_sae_alignment_summary.tsv` exists.
- `behavioral_sae_alignment_summary.md` exists.
- `5-12_behavioral_sae_alignment_memo.md` exists.
- `5-12_behavioral_sae_alignment_20260512.txt` exists.
- TopK row count is preserved at `2800`.
- Final numeric outputs contain no NaN/Inf values.
- NaN references are limited to the recorded `device_map="auto"` diagnostic/provenance note and script metadata explaining the failed path.

## Primary Local Files

```text
5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/behavioral_sae_alignment_summary.tsv
5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/behavioral_sae_alignment_summary.md
5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/5-12_behavioral_sae_alignment_memo.md
5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/provenance/5-12_behavioral_sae_alignment_20260512.txt
```

## Organized Layout

```text
5-12-26/qwen-scope/
├── README.md
├── artifacts/
│   ├── 5-12-26_qwen_scope_behavioral_sae_artifacts_36630892.tar.gz
│   ├── 5-12-26_qwen_scope_behavioral_sae_artifacts_36630892.tar.gz.sha256
│   └── 5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/
├── provenance/
│   ├── local_archive_verification_20260512.txt
│   └── teardown_36630892_20260512.txt
└── staging/
    └── remote_scripts/
```

`artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/` is the extracted, audited copy of the remote small-artifact archive. `staging/remote_scripts/` contains the local staging copies of the scripts that were sent to the remote instance; the canonical run copies are also preserved inside the extracted archive under `scripts/`.
