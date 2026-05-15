# Extended Latin Tokenizer Audit

Tokenizer: local Qwen/Qwen3.5-35B-A3B tokenizer files.

## Character-Level Results

- Inventory characters: 14.
- Passage character occurrences represented: 99.
- `single_token_exact_decode`: 14 characters.

## Example-Word Results

- Example words audited: 34.
- Token-inflated relative to ASCII fold: 14.
- Token-matched relative to ASCII fold: 20.
- Token-compacted relative to ASCII fold: 0.

Largest positive deltas:
- `tongá` -> `tonga`: 3 vs 2 (delta 1).
- `asumá` -> `asuma`: 3 vs 2 (delta 1).
- `oémi` -> `oemi`: 3 vs 2 (delta 1).
- `Aksé` -> `Akse`: 3 vs 2 (delta 1).
- `setíval` -> `setival`: 3 vs 2 (delta 1).
- `aíd` -> `aid`: 2 vs 1 (delta 1).
- `örej` -> `orej`: 2 vs 1 (delta 1).
- `úri` -> `uri`: 2 vs 1 (delta 1).
- `ütu` -> `utu`: 2 vs 1 (delta 1).
- `irý` -> `iry`: 2 vs 1 (delta 1).

## Outputs

- `5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_character_tokenization.tsv`
- `5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_example_word_tokenization.tsv`
- `5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_tokenizer_audit_metadata.json`
- `5-15-26/qwen-scope/provenance/extended_latin_tokenizer_audit_20260515.txt`
