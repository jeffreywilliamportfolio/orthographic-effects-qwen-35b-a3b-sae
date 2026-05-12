# Results

This is an evidence summary, not a semantic-label report.

## Pipeline Validation

The Qwen-Scope pipeline was validated on a 2 x 96GB GPU instance:

- Hugging Face base model loaded locally with Transformers/PyTorch.
- Qwen-Scope layer SAE files loaded locally.
- Selected-layer hooks worked without requesting all hidden states.
- Official Qwen-Scope TopK-50 encoding was used.
- Initial normal/d-stroke two-prompt delta was reproduced after migration.

## Feature-Label Pilot

The first priority summary selected layer-26 feature IDs:

- `23977`
- `2722`
- `9745`
- `7108`
- `31784`

A 30-prompt seed scan produced zero tracked TopK-50 hits, suggesting these were not broad category features in that seed set.

## Locality Check

The 10-prompt locality check scanned the original normal/d-stroke pair and near-neighbor variants at five boundary positions.

Key readout:

- Original normal and d-stroke prompts reproduced prior tracked final-token hit sets.
- Tracked features appeared at nearby boundary positions.
- Near-neighbor variants also had hits.
- `2722` and `7108` appeared only in normal-family prompts in that locality run.
- `23977`, `9745`, and `31784` crossed both normal-family and d-stroke-family prompts.

## Matched Perturbation Control

The matched perturbation control compared:

- `ascii_original`
- `d_to_ḑ`
- `e_to_ē`
- `s_to_ş`
- `s_to_ṡ`

Key readout:

- `2722` appeared only in `ascii_original` prompts in that control.
- `7108` appeared in `ascii_original` and once in `e_to_ē`.
- `31784` appeared in `d_to_ḑ`, but also in `ascii_original`, `e_to_ē`, and `s_to_ṡ`.
- Hits were concentrated most at `final_prompt_token`, with some at `minus_1` and `minus_5`.

## Full Controlled Matrix

The full matrix used:

- 5 base prompt families.
- 6 perturbation types.
- 5 boundary positions.
- 2 layers.

Verified counts:

- Prompt matrix rows: `30`
- TopK rows: `15000`
- Tracked layer-26 rows: `750`
- Generated text rows: `30`
- Prompt-position-layer captures: `300`
- Skipped positions: `0`

Aggregate readout:

- Largest mean feature deltas versus ASCII came from `e_to_ē`, `s_to_ş`, then `s_to_ṡ`.
- `d_to_ḑ` was much smaller by mean absolute delta.
- Layer 26 had stronger perturbation sensitivity than layer 14 by mean absolute delta.
- Tracked hits concentrated most at the final prompt token, while larger average deltas were often stronger at nearby positions such as `minus_10` and `minus_5`.
- `2722` did not remain strictly ASCII-only in the full matrix.
- `7108` remained mostly ASCII-concentrated, with some `e_to_ē` hits.
- `31784` did not behave as d-stroke-specific; it appeared across multiple perturbation types and boundary positions.

## Generated Text Note

Short greedy generations were saved for context in each scan. For the full matrix, `e_to_ē` caused larger token inflation and stronger SAE perturbation metrics than `d_to_ḑ`, while `d_to_ḑ` produced the clearest single "opening up" snippet in the `removed_sentence_hum` family.

See:

`results/full_controlled_perturbation_matrix/generated_text_by_prompt.tsv`

