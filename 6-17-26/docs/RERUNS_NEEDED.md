# Reruns Needed Before Manuscript-Grade Claims

## Minimum Manuscript-Grade Evidence Package

1. Freeze one prompt manifest with four semantic families and all perturbation/control variants.
2. Run deterministic Qwen SAE captures at pre-registered layers and positions.
3. Run local Gemma-3-4B replication with the same prompt manifest.
4. Run a black-box comparison panel with full API payload/response logging.
5. Run blinded scoring over all generated outputs with a frozen rubric.
6. Generate figures and tables only from standardized run outputs.

## Recommended Qwen Rerun

- Model: `Qwen/Qwen3.5-35B-A3B-Base`.
- SAE: `Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`.
- Layers: 14, 15, 16, 24, 25, 26 unless a pre-analysis narrows them.
- Positions: final prompt token, generated tokens 1, 8, 16, 32, 64.
- Decoding: greedy deterministic, fixed max token budget, no hidden prompt changes.
- Repetitions: at least 3 deterministic reruns or one deterministic rerun plus independent script reproduction on a fresh machine.
- Add branch-probe block: no-prefix, neutral forced prefix, checking prefix, denial prefix, direct-answer prefix.

## Recommended Gemma Rerun

- Model: Gemma-3-4B PT and IT, exact checkpoint strings recorded.
- Same prompt manifest as Qwen.
- Same tokenization audit fields.
- SAE or activation capture layers pre-registered before running.
- Greedy deterministic baseline plus optional small temperature sweep only after primary deterministic outputs are complete.

## Recommended Black-Box API Panel

- Include GPT-5.4, GPT-5.2, GPT-5, GPT-4o, and o3 if the account exposes those model IDs at run time.
- Preserve full request/response JSON, timestamps, model IDs, usage metadata, refusal metadata, and errors.
- Use a single prompt manifest and a single scoring rubric.
- Do not mix canonical cloud prompt-pack runs with TINE or standardized prompt-family runs in the same primary table.

## Go/No-Go Criterion

Go only if:

- Every primary run has exact prompt hashes, tokenizer counts or explicit unavailable markers, decoding settings, raw outputs, logs, model identifiers, and checksums.
- Qwen and Gemma both show tokenization/activation displacement under perturbations while controls constrain the interpretation.
- Behavioral effects are reported as output-regime changes, not as a universal semantic stance shift.
- Black-box results are framed as external validity checks, not mechanistic proof.

No-go if:

- The main result depends on older mixed prompt families.
- GPT/API outputs lack raw request/response provenance.
- Rule labels remain the only behavioral scoring.
- The claim requires diacritic-specific causality where token-count-matched controls produce comparable shifts.

