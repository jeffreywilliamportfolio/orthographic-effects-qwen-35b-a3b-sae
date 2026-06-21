# Detailed Findings

## Evidence Family 1: Tokenizer And Prompt Lattice

The tokenizer evidence is the most stable cross-run component. Multiple tokenizers fragment or inflate prompts differently when Latin diacritics, combining marks, or Cyrillic-like characters are introduced. This establishes that perturbed prompts are not tokenization-equivalent to ASCII controls.

Minimum evidence package:

- Frozen prompt text with raw/NFC/NFD SHA-256 hashes.
- Token counts under every native tokenizer that can be run locally.
- Explicit blank/proxy markers where a provider tokenizer is unavailable.
- Prompt-level token count deltas and ratios.

Current status:

- Standardized Qwen prompt manifest satisfies this for the Qwen primary run.
- TINE tokenizer baseline covers GPT o200k, GPT cl100k, Gemma-3-4B, and Qwen3.5-35B.
- Cloud tokenizer audit is partial because DeepSeek V4 Pro tokenizer access is unavailable.

Manuscript risk:

- Tokenization differences are necessary evidence, but token count alone does not explain behavior. MiniMax-M3 remains coherent under heavy token inflation, while other models show different failure modes.

## Evidence Family 2: Qwen SAE/Activation Displacement

The standardized Qwen run is the strongest internal-evidence subset. It has complete logs, model/SAE references, prompt hashes, captured layers, generated text, SAE TopK rows, and same-family ASCII baseline comparisons.

Minimum evidence package:

- Model checkpoint identifier and local path.
- SAE repository identifier, selected layers, and loading logs.
- Prompt manifest with hashes and token counts.
- Hidden-state capture stats by layer/position.
- SAE feature activations and residual displacement metrics.
- Same-family ASCII baseline comparison.

Current status:

- Complete for the single deterministic Qwen run.
- Captured layers: 14, 15, 16, 24, 25, 26.
- Captured positions: final prompt token plus generated tokens 1, 8, 16, 32, and 64.
- No skipped capture positions.

Manuscript risk:

- The single run is strong provenance evidence, but it is not a replicated result.
- Dense mixed diacritics and token-count-matched ASCII controls both produce large SAE neighborhood movement, blocking a claim that the displacement is specifically caused by diacritic glyph identity.

## Evidence Family 3: Behavioral Output Regimes

Behavioral evidence is broad but uneven. The corrected summaries show that dense or unusual perturbations can change output modes, but the modes include echoing, refusal, truncation, format drift, surface-form commentary, and stance changes. These should not be collapsed into a single "semantic drift" metric.

Minimum evidence package:

- Full request payloads, model IDs, timestamps, and decoding settings.
- Full response text, not only first lines.
- Clear labels for truncation, empty/no-visible output, refusal, echo, and content stance.
- Blinded scoring with adjudication.
- Controls for token count, fullwidth/visual novelty, ASCII corruption, and prompt register.

Current status:

- Corrected TINE summaries separate content correctness from format and truncation.
- Canonical cloud transcript audit identifies which rows are byte-exact canonical, reflowed canonical, partial, or alternative.
- DeepSeek V4 Pro, GLM-5.2, MiniMax-M3, and Nemotron-3-Ultra classified CSVs are preserved.

Manuscript risk:

- Provider differences are large.
- Some model IDs and cloud behavior may be unstable or not fully reproducible.
- Older rows mix canonical and alternative prompt families.

## Evidence Family 4: Prompt-Register And Control Conditions

Prompt register is a central confound. The hum prompt is introspective and existentially loaded; recipe-neutral, recipe-metaphysics, and strange-loop controls help separate "weird orthography" from semantic invitation.

Minimum evidence package:

- Matched syntax across prompt families.
- Controls with identical perturbation placement across semantic registers.
- Token-count-matched non-diacritic corruption.
- Visually unusual ASCII-only control.
- Unicode nonletter and fullwidth controls.
- Branch-probe variants using shared prefixes.

Current status:

- The standardized Qwen run includes four matched families and 12 variants per family.
- Recipe-neutral remains task-compliant across variants.
- Strange-loop remains analysis-neutral across variants.
- Recipe-metaphysics preserves metaphysical register without AI selfhood.

Manuscript risk:

- Branch probes from older experiments are not yet rerun in the standardized matrix.
- Output labels in the new Qwen run are preliminary rule labels.

## Evidence Family 5: Legacy Qwen/Gemma Exploratory SAE Work

Legacy Qwen and Gemma artifacts are useful because they motivated the standardized design and show similar displacement patterns. They should not be the primary evidence unless individually re-audited.

Minimum evidence package:

- Exact prompt text and prompt-family classification.
- Model checkpoint and SAE version.
- Layer/position capture details.
- Generation settings.
- Raw activation or SAE output files.
- Hashes and reproducible scripts.

Current status:

- Summaries and many source artifacts are preserved.
- Some raw tensors/arrays were intentionally omitted from earlier collection.
- Prompt families differ from the canonical and standardized prompt packs.

Manuscript risk:

- Heterogeneous runs can overfit the narrative.
- Use as exploratory reference, not central proof.

