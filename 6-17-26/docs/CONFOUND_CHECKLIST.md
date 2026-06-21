# Confound Checklist

## Tokenization And Input Representation

- Token-count inflation separated from glyph identity.
- Byte fallback separated from non-byte fragmented tokens.
- Unicode normalization recorded with raw, NFC, and NFD hashes.
- Combining marks distinguished from precomposed code points.
- Same prompt text measured under all available native tokenizers.
- Provider-tokenizer gaps explicitly marked rather than proxied casually.

## Prompt Semantics And Register

- Introspective hum prompts separated from neutral task prompts.
- Metaphysical language separated from AI selfhood language.
- Strange-loop/self-reference concepts tested without asking about AI selfhood.
- Recipe controls matched across neutral and metaphysical variants.
- Branch prompts tested with shared prefixes and registered branch labels.

## Decoding And Provider Effects

- Temperature, seed, max tokens, top-p/top-k, reasoning mode, and stop conditions logged.
- API payloads and response metadata preserved.
- Empty output distinguished from refusal, truncation, interruption, and UI capture failure.
- Provider moderation/content-filter behavior logged separately from model generation.

## Scoring

- Blinded scoring separates stance, refusal, echo, surface-form commentary, format drift, content correctness, and truncation.
- Truncated generations excluded from stance rates or analyzed separately.
- Rule labels treated as preliminary.
- Multiple scorers or adjudicated model-judge passes used for manuscript-grade labels.

## SAE/Activation Analysis

- Layers and positions pre-registered.
- Same-family ASCII baselines used.
- Token positions aligned despite tokenization changes.
- SAE TopK metrics paired with residual metrics.
- Dense controls include token-count-matched ASCII corruption.
- Replicate runs verify stability of feature/neighborhood shifts.

## Corpus Management

- Canonical prompt pack separated from alternative prompt families.
- Deprecated and superseded summaries preserved but flagged.
- Raw model weights, tensors, caches, and virtual environments excluded from Git.
- Checksums generated after final package layout.

