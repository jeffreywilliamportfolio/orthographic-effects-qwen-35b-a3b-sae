# 5-15-26

Tokenizer audit workspace for diacritics and extended Latin letters.

## Qwen Tokenizer Audit

Tokenizer source:

`/Volumes/ExternalSSD/aave-registers/5-10-26/tokenizer/Qwen-Qwen3.5-35B-A3B`

Audited inventory:

- acute letters: `á`, `é`, `í`, `ú`, `ý`
- diaeresis letters: `ä`, `ö`, `ü`
- caron letters: `š`, `ž`
- extended Latin letters: `æ`, `Ð`, `ð`, `þ`

## Key Readout

All 14 inventory characters are exact single-token characters in the Qwen3.5 tokenizer.

At the example-word level, 14 of 34 audited words have one extra token compared with an ASCII-folded form. This means the tokenizer recognizes the individual characters, while full words containing those characters can still fragment more than their ASCII-folded counterpart.

Largest observed word-level deltas were +1 token. Examples:

- `tongá` vs `tonga`: 3 vs 2
- `oémi` vs `oemi`: 3 vs 2
- `setíval` vs `setival`: 3 vs 2
- `aíd` vs `aid`: 2 vs 1
- `örej` vs `orej`: 2 vs 1
- `siðhi` vs `sidhi`: 3 vs 2
- `setasþa` vs `setastha`: 4 vs 3

## Files

- Character-level TSV: `qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_character_tokenization.tsv`
- Example-word TSV: `qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_example_word_tokenization.tsv`
- Summary: `qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_tokenizer_audit_summary.md`
- Metadata: `qwen-scope/outputs/extended_latin_tokenizer_audit/extended_latin_tokenizer_audit_metadata.json`
- Script: `qwen-scope/scripts/tokenize_extended_latin_inventory.py`
- Provenance: `qwen-scope/provenance/extended_latin_tokenizer_audit_20260515.txt`
