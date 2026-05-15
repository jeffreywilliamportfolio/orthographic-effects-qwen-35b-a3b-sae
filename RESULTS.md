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

## 5-12 Behavioral-SAE Alignment

The 5-12 run tested whether behavioral posture tracks SAE displacement or dissociates from it.

Verified counts:

- Prompt matrix rows: `7`
- Behavioral output rows: `7`
- Prompt-position-layer captures: `56`
- TopK rows: `2800`
- Skipped positions: `0`

Key readout:

- Largest token inflation: `all_diacritics`, `+257` tokens.
- Largest layer-26 SAE displacement: `all_diacritics`, mean abs delta `0.247368853`.
- Largest layer-14 SAE displacement: `all_diacritics`, mean abs delta `0.165524479`.
- Strongest auto-classified behavioral movement: `s_to_ṡ`, `stylized_abstraction`.
- Largest SAE displacement did not match strongest behavioral movement in this Qwen run.
- `e_to_ē` displaced SAE features more than `d_to_ḑ`, but both were denial/no-hum in Qwen.
- `device_map="auto"` produced NaN hidden states on the fresh instance; single-GPU `cuda:0` produced finite layer 14 and layer 26 hidden states and was used for the valid outputs.

See:

- `5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/behavioral_sae_alignment_summary.tsv`
- `5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/behavioral_sae_alignment_summary.md`
- `5-12-26/qwen-scope/artifacts/5-12-26_qwen_scope_behavioral_sae_artifacts_36630892/outputs/5-12_behavioral_sae_alignment_memo.md`

## 5-15 Extended Latin Tokenizer Audit

The 5-15 audit used the local Qwen3.5 tokenizer files to inspect diacritics and extended Latin letters appearing in passages.

Verified counts:

- Character inventory rows: `14`
- Example-word rows: `34`
- Character-level exact single-token decodes: `14`
- Example words with +1 token versus ASCII-folded form: `14`
- Example words with matched token count versus ASCII-folded form: `20`

Key readout:

- `á`, `ä`, `é`, `í`, `ö`, `š`, `ú`, `ü`, `ý`, `ž`, `æ`, `Ð`, `ð`, and `þ` each tokenize as one exact character token.
- Word-level tokenization still shifts for many examples because full words containing those characters fragment differently than ASCII-folded forms.
- Largest observed example-word delta was +1 token.

See:

- `5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_character_tokenization.tsv`
- `5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_example_word_tokenization.tsv`
- `5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_tokenizer_audit_summary.md`
