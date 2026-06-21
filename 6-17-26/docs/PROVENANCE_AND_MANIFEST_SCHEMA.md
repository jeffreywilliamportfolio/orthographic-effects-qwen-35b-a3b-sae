# Provenance And Manifest Schema

## Required Provenance Fields

Every manuscript-grade run should have:

- `experiment_id`: stable identifier used in filenames and tables.
- `evidence_tier`: primary, structured_exploratory, or legacy_exploratory.
- `source_path`: original path before packaging.
- `package_path`: path inside this standalone package.
- `artifact_type`: prompts, generations, tokenizer_audit, sae_capture, activation_metric, scoring, logs, script, figure, or metadata.
- `model_or_tool`: model/checkpoint/API/tool name.
- `model_version_detail`: exact checkpoint, provider model string, or unavailable marker.
- `prompt_set`: canonical_sweep, tine_battery, standardized_qwen_20260617, legacy_branch_probe, etc.
- `variant_set`: perturbation/control family.
- `decoding_or_capture`: temperature, seed, max tokens, layers, positions, TopK settings, or unavailable marker.
- `records`: row count or file count.
- `hashes`: SHA-256 source, package, or internal prompt hashes.
- `provenance_status`: clean, partial, mixed, or deprecated.
- `known_limitations`: missing tokenizer, hidden provider settings, rule labels only, omitted tensors, etc.
- `recommended_use`: primary evidence, supporting evidence, exploratory context, or preserve only.

## Manifest Files In This Package

### `manifests/file_inventory.tsv`

Machine-generated inventory of package files. Columns:

- `path`
- `bytes`
- `sha256`
- `extension`
- `artifact_guess`

### `manifests/data_provenance_manifest.tsv`

Curated artifact-family table. Columns:

- `package_path`
- `source_path`
- `family`
- `artifact_type`
- `evidence_tier`
- `model_or_tool`
- `prompt_set`
- `variant_set`
- `decoding_or_capture`
- `records`
- `provenance_status`
- `known_limitations`
- `recommended_use`

### `manifests/experiment_index.tsv`

Experiment-family table. Columns:

- `experiment_id`
- `date_or_label`
- `evidence_tier`
- `model_or_platform`
- `prompt_family`
- `perturbations_or_controls`
- `primary_artifacts`
- `clean_evidence`
- `main_result`
- `publication_blocker`

### `manifests/preservation_actions.tsv`

File and directory disposition table. Columns:

- `path_or_pattern`
- `action`
- `canonical_target`
- `reason`

## Promotion Rule

An exploratory run can be promoted to manuscript-grade evidence only if it has exact prompt text, hashes, model/checkpoint details, decoding settings, raw outputs, scoring rubric, logs, and either native tokenizer counts or an explicit unavailable-tokenizer marker.

