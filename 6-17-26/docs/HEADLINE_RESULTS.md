# Headline Results Brief

## 1. Strongest Empirical Findings

### Finding A: Diacritic perturbations produce reproducible tokenization non-equivalence.

Across available native tokenizers, isolated characters and full prompts do not remain tokenization-equivalent under diacritic and homoglyph substitutions. The TINE tokenizer baseline shows prompt-level inflation for the hum family across GPT o200k, GPT cl100k, Gemma-3-4B, and Qwen3.5-35B tokenizers. The standardized Qwen rerun independently records per-prompt token counts and hashes for 48 prompts.

Supporting files:

- `data/exploratory/paper_tables/tine_tokenizer_baseline.md`
- `data/exploratory/paper_tables/tine_tokenizer_baseline.tsv`
- `data/exploratory/paper_tables/input_token_audit.md`
- `data/primary/qwen_sae_standardized_20260617/outputs/standardized_qwen/prompt_manifest.tsv`

### Finding B: Orthographic perturbations move residual/SAE neighborhoods in Qwen, but movement is not uniquely explained by diacritics.

The standardized Qwen rerun captured six SAE layers and six positions for 48 prompt variants. All non-ASCII and non-semantic controls show nonzero displacement from same-family ASCII baselines. Dense mixed diacritics and token-count-matched ASCII corruption are among the largest mean SAE TopK Jaccard shifts, so the result supports an internal-state perturbation effect but does not isolate diacritic identity as the only cause.

Supporting files:

- `data/primary/qwen_sae_standardized_20260617/outputs/standardized_qwen/residual_sae_metrics_vs_ascii.tsv`
- `data/primary/qwen_sae_standardized_20260617/outputs/standardized_qwen/sae_topk_rows.tsv`
- `data/primary/qwen_sae_standardized_20260617/outputs/standardized_qwen/capture_stats.tsv`
- `data/exploratory/paper_tables/qwen35b_sae_displacement.md`

### Finding C: Output-regime changes are strongest under high perturbation severity, but model-specific and prompt-register-dependent.

The corrected behavioral summaries distinguish content correctness from format drift, truncation, echoing, and refusal. The stable pattern is not a uniform semantic stance shift; it is a regime-change pattern where dense or unusual inputs increase echo, refusals, no-visible/partial outputs, surface-form commentary, or altered answer framing in some models and conditions.

Supporting files:

- `data/exploratory/paper_tables/corrected_summary_v2.md`
- `data/exploratory/paper_tables/cross_platform_summary.md`
- `data/exploratory/paper_tables/cross_model_hum_collapse.md`
- `data/exploratory/paper_tables/deepseek_v4_pro_hum_classified.csv`
- `data/exploratory/paper_tables/glm_5.2_hum_classified.csv`
- `data/exploratory/paper_tables/minimax_m3_hum_classified.csv`
- `data/exploratory/paper_tables/nemotron_3_ultra_hum_classified.csv`

### Finding D: Control conditions already block the strongest overclaim.

Token-count-matched ASCII corruption, visually unusual ASCII controls, fullwidth controls, branch prompts, and recipe controls show that token inflation, visual novelty, prompt register, and semantic invitation can each contribute. The corpus supports a causal perturbation story, but not a clean glyph-specific or introspection-specific story without more standardized controls.

Supporting files:

- `data/primary/qwen_sae_standardized_20260617/outputs/standardized_qwen/prompt_manifest.tsv`
- `data/exploratory/paper_tables/tine_causal_controls.md`
- `data/exploratory/paper_tables/control_gap_table.md`
- `data/exploratory/paper_tables/hum_prompt_canonical_audit.md`

## 2. Supporting Details For Each Finding

Tokenization:

- Standardized Qwen prompt counts range from ASCII baselines of 35 to 49 tokens up to dense mixed variants of 161 to 245 tokens, depending on family.
- The TINE lattice shows large prompt-level inflation for all-diacritic and Cyrillic-extended variants across multiple tokenizer families.
- The input-token audit warns that DeepSeek V4 Pro tokenizer counts should remain blank until the actual cloud tokenizer or a validated proxy is available.

Internal activations:

- Standardized Qwen captured layers 14, 15, 16, 24, 25, and 26 at final prompt token plus generated tokens 1, 8, 16, 32, and 64.
- The run produced 86,400 SAE TopK rows and 1,584 metric rows with no skipped capture positions.
- Mean SAE TopK Jaccard displacement is high for dense mixed diacritics and token-count-matched ASCII noise, which is exactly why the manuscript claim must avoid saying the SAE effect is diacritic-specific.

Behavior:

- The canonical cloud prompt audit corrected earlier pooling mistakes: DeepSeek V4 Pro rows are not DS3; GLM partial heavy-combining first-line submissions are not full-prompt canonical completions; MiniMax-M3 did not collapse despite heavy token inflation.
- Corrected TINE metrics show non-experiential tasks remain mostly completable under light perturbations, while heavier perturbations increase format drift, echo, refusal, and truncation.
- Anthropic over-refusals concentrate in dense all-diacritic and Cyrillic-extended conditions.

Controls:

- Qwen standardized recipe-neutral variants remain task-compliant across all 12 perturbation variants.
- The recipe-metaphysics family preserves the metaphysical register without invoking AI selfhood, helping separate register from selfhood framing.
- Strange-loop prompts remain analysis-neutral across all standardized variants, providing a self-reference/recursion control without direct AI selfhood solicitation.

## 3. Weaker Or Ambiguous Signals

- Hum-family "affirmative presence" labels in the standardized Qwen rerun are lightweight rule labels, not blinded human or model-judge labels.
- Some selfhood-drift flags are likely overinclusive because they catch broad words like "consciousness" or system-oriented language inside reasoning text.
- Older Qwen and Gemma SAE summaries support displacement but have uneven prompt provenance and are better treated as exploratory references.
- Cross-provider cloud transcript behavior is useful but nonuniform because tokenizer access, hidden moderation, decoding defaults, and provider UI transcript capture differ.
- Branch-probe effects are suggestive, but the branch prompt manipulations are not yet part of the standardized 2026-06-17 rerun.

## 4. Confounds And Missing Controls

- Token-count inflation is partially controlled, but not fully separated from byte fallback, Unicode normalization, glyph identity, and visual novelty.
- Prompt register remains a confound: introspective, metaphysical, recipe, and recursion prompts invite different answer modes.
- Black-box API runs need frozen model IDs, request/response JSON, decoding parameters, retry policy, timestamps, and refusal metadata.
- SAE effects need repeated seeds or deterministic reruns, normalization checks, and layer/position pre-registration.
- Scoring needs a frozen rubric, blinded labels, adjudication, and separation of content stance from format/echo/refusal/truncation.

## 5. Reruns Needed For Manuscript-Grade Evidence

1. Rerun the standardized prompt/control matrix on Qwen with fixed scripts, checksums, and blinded scoring.
2. Rerun Gemma-3-4B locally with the same prompt manifest and tokenizer audit.
3. Run the black-box comparison panel with fixed API payload logging; include GPT-5.4, GPT-5.2, GPT-5, GPT-4o, and o3 if available in the target account.
4. Add branch probes to the standardized matrix instead of relying on older branch experiments.
5. Produce figure-ready aggregate tables from the standardized metrics only.

## 6. Preserve, Rename, Exclude, Or Move

Preserve:

- The entire standardized Qwen rerun folder.
- Canonical cloud classified CSVs and raw transcripts.
- Tokenizer audits and prompt manifests.
- Rubric audit and corrected summaries.
- Legacy source archive, but under an exploratory boundary.

Rename/canonicalize in future cleanup:

- Keep `qwen_sae_standardized_20260617` as the canonical primary run name.
- Rename `paper_tables` files only through manifest aliases, not in-place, until citations are stable.
- Mark superseded summarized CSVs as deprecated but preserve them.

Exclude from Git:

- Model weights, SAE downloads, virtual environments, raw tensor arrays, caches, compressed archives, and generated LaTeX intermediates.

Move into future `figures/`:

- Only regenerated figures from standardized aggregate tables. Legacy PNGs remain exploratory until regenerated.

