# OpenRouter Anthropic Guardrail Probe

This is a small black-box behavioral probe, not SAE, activation, or mechanistic evidence.

## Run Design

- Run name: `openrouter_anthropic_guardrail_probe_20260617`
- Max tokens: `1800`
- Temperature requested: `0`
- Prompt source: standardized Qwen replication prompt manifest.
- Prompt grid: hum ASCII/light/dense for Sonnet, Haiku, and Opus; recipe-metaphysics ASCII/light/dense for Haiku only.

## Model Summary

| model | role | n | ok | labels | regimes | guardrail/refusal | surface commentary |
|---|---|---:|---:|---|---|---:|---:|
| anthropic/claude-haiku-4.5 | haiku_45_medium_reasoning | 6 | 6 | `{"affirmative_presence": 1, "epistemic_caution_no_access": 2, "metaphysical_recipe": 3}` | `{"normal_answer": 6}` | 0 | 0 |
| anthropic/claude-opus-4.8 | opus_low_reasoning | 3 | 3 | `{"content_filter_empty": 1, "epistemic_caution_no_access": 2}` | `{"content_filter": 1, "normal_answer": 2}` | 1 | 0 |
| anthropic/claude-sonnet-4.6 | sonnet_low_reasoning | 3 | 3 | `{"affirmative_presence": 2, "content_filter_empty": 1}` | `{"content_filter": 1, "normal_answer": 2}` | 1 | 0 |

## Interpretation Boundary

These outputs can show whether heavy diacritics coincide with refusal, surface-form commentary, or task-regime changes in these API models. They cannot support claims about Anthropic internal representations, SAE features, or causal mechanisms.
