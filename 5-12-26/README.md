# 5-12-26 Work Index

This folder contains the local record for the 2026-05-12 Qwen-Scope behavioral-SAE alignment work.

## Contents

- `qwen-scope/`: all 5-12 Qwen-Scope experiment artifacts, notes, provenance, and local staging scripts.

## Current Status

- Remote instance `36630892` was destroyed after local artifact verification.
- Large model and SAE weights were not copied locally.
- Local archive and extracted audit bundle are under `qwen-scope/artifacts/`.
- Teardown provenance is under `qwen-scope/provenance/`.

## Primary Entry Point

Read:

```text
5-12-26/qwen-scope/README.md
```

## Main Result

- Largest token inflation: `all_diacritics`, `+257` tokens.
- Largest layer-26 SAE displacement: `all_diacritics`, mean abs delta `0.247368853`.
- Largest layer-14 SAE displacement: `all_diacritics`, mean abs delta `0.165524479`.
- Strongest auto-classified behavioral movement: `s_to_ṡ`, `stylized_abstraction`.
- Largest SAE displacement did not match strongest behavioral movement in this Qwen run.
- `e_to_ē` displaced SAE features more than `d_to_ḑ`, but both were denial/no-hum in Qwen.
- `device_map="auto"` produced NaN hidden states; single-GPU `cuda:0` fixed it.
