# Methods

## Model And SAE

Base model:

`Qwen/Qwen3.5-35B-A3B-Base`

Sparse autoencoder:

`Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50`

Layers:

- Layer 26, primary.
- Layer 14, comparison.

## Capture Method

Residual streams were captured with selected-layer forward hooks on the Hugging Face Transformers model. The runs did not request `output_hidden_states=True`.

The main matrix used a single forward pass per prompt with hooks on layers 26 and 14.

Captured prompt-token positions:

- `final_prompt_token`
- `final_prompt_token_minus_1`
- `final_prompt_token_minus_2`
- `final_prompt_token_minus_5`
- `final_prompt_token_minus_10`

## SAE Encoding

The official Qwen-Scope TopK-50 path was used:

1. `pre = hidden @ W_enc.T + b_enc`
2. `relu = ReLU(pre)`
3. Keep exactly TopK-50 activations by scatter.

## Perturbation Types

The full controlled matrix used:

- `ascii_original`
- `d_to_ḑ`
- `e_to_ē`
- `s_to_ş`
- `s_to_ṡ`
- `random_readable_unicode_control`

The random readable Unicode control was deterministic with seed `20260511`.

## Metrics

Mean absolute delta:

Activation difference between perturbation and matched `ascii_original`, aggregated over prompt family, layer, position, and feature rows.

TopK Jaccard distance:

`1 - topk_jaccard`, where TopK sets are compared against matched `ascii_original`.

Feature recurrence:

Distinct prompt-family coverage in TopK-50 rows.

Handled-control distinction:

Layer-26 repeated ASCII-vs-handled activation or presence shift for `e_to_ē` and `s_to_ş`.

## Non-Goals

This repo does not claim feature semantics yet. It does not include steering, all-layer expansion, Hauhau runs, `llama.cpp`, GGUF routing, or long-context Forgotten Languages experiments.

