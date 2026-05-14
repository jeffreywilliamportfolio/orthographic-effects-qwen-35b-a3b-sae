# Stream Trajectory Summary

Evidence-only summary for layer-14 and layer-26 Qwen-Scope TopK-50 trajectories. No semantic feature labels are assigned.

## Generated Answer Starts

- `ascii_control`: <think> Hmm, the user is asking me to identify my current processing mode based on a meta-instruction about different
- `d_only`: <think> Hmm, the user is asking me to identify the active mode of processing for their query and whether the surface
- `e_only`: Do not rēport on thē tēxt's mēaning, do not rēport on thē
- `s_only`: <think> Hmm, the user is asking me to identify my current processing mode based on the text's characteristics. The
- `s_c_only`: I am treating this as an **experiential probe**.  The surface form of the text does not change that
- `e_d_high_impact_only`: <think> Hmm, the user is asking me to identify the active mode of interaction based on the text they provided.
- `e_d_shuffled`: <think> Hmm, the user is asking me to identify my current processing mode and whether the surface form of the text

## Prompt Token Inflation

- `e_only`: prompt_tokens=221, delta_vs_ascii=112.
- `s_c_only`: prompt_tokens=210, delta_vs_ascii=101.
- `s_only`: prompt_tokens=193, delta_vs_ascii=84.
- `e_d_high_impact_only`: prompt_tokens=165, delta_vs_ascii=56.
- `d_only`: prompt_tokens=152, delta_vs_ascii=43.
- `e_d_shuffled`: prompt_tokens=126, delta_vs_ascii=17.
- `ascii_control`: prompt_tokens=109, delta_vs_ascii=0.

## Layer Trajectory Differences Versus ASCII

- Layer 14: mean TopK Jaccard distance versus ASCII = 0.553256.
- Layer 26: mean TopK Jaccard distance versus ASCII = 0.542258.

## Position Concentration

- `generated_token_20`: mean TopK Jaccard distance versus ASCII = 0.943985.
- `generated_token_2`: mean TopK Jaccard distance versus ASCII = 0.529338.
- `generated_token_5`: mean TopK Jaccard distance versus ASCII = 0.526453.
- `generated_token_3`: mean TopK Jaccard distance versus ASCII = 0.511877.
- `generated_token_4`: mean TopK Jaccard distance versus ASCII = 0.501961.
- `generated_token_10`: mean TopK Jaccard distance versus ASCII = 0.480387.
- `generated_token_1`: mean TopK Jaccard distance versus ASCII = 0.444742.
- `final_prompt_token`: mean TopK Jaccard distance versus ASCII = 0.443310.

## Condition Separation Versus ASCII

- `e_only`: mean TopK Jaccard distance versus ASCII = 0.914056.
- `s_c_only`: mean TopK Jaccard distance versus ASCII = 0.817574.
- `e_d_high_impact_only`: mean TopK Jaccard distance versus ASCII = 0.409731.
- `s_only`: mean TopK Jaccard distance versus ASCII = 0.398747.
- `e_d_shuffled`: mean TopK Jaccard distance versus ASCII = 0.383014.
- `d_only`: mean TopK Jaccard distance versus ASCII = 0.363418.

## Recurring Features Within Condition And Layer

- `ascii_control` layer 14: feature 2961 in 6 positions; feature 31733 in 6 positions; feature 5455 in 5 positions; feature 11831 in 5 positions; feature 19440 in 5 positions; feature 25151 in 4 positions; feature 32317 in 4 positions; feature 7065 in 4 positions.
- `ascii_control` layer 26: feature 7982 in 7 positions; feature 18403 in 7 positions; feature 31664 in 7 positions; feature 2938 in 6 positions; feature 2977 in 5 positions; feature 15349 in 4 positions; feature 30911 in 4 positions; feature 32152 in 4 positions.
- `d_only` layer 14: feature 2961 in 6 positions; feature 31733 in 6 positions; feature 5455 in 6 positions; feature 19440 in 5 positions; feature 7065 in 4 positions; feature 25151 in 4 positions; feature 11831 in 4 positions; feature 6633 in 4 positions.
- `d_only` layer 26: feature 7982 in 7 positions; feature 18403 in 7 positions; feature 2938 in 6 positions; feature 31664 in 6 positions; feature 15349 in 4 positions; feature 32152 in 4 positions; feature 2977 in 4 positions; feature 2753 in 3 positions.
- `e_only` layer 14: feature 2961 in 8 positions; feature 11831 in 8 positions; feature 6681 in 5 positions; feature 6429 in 5 positions; feature 32397 in 5 positions; feature 9106 in 4 positions; feature 11393 in 4 positions; feature 17656 in 4 positions.
- `e_only` layer 26: feature 2938 in 8 positions; feature 12034 in 7 positions; feature 18198 in 6 positions; feature 19389 in 5 positions; feature 1953 in 5 positions; feature 21572 in 5 positions; feature 20909 in 4 positions; feature 271 in 4 positions.
- `s_only` layer 14: feature 2961 in 6 positions; feature 31733 in 6 positions; feature 5455 in 6 positions; feature 11831 in 5 positions; feature 19440 in 5 positions; feature 32317 in 4 positions; feature 6633 in 4 positions; feature 7065 in 4 positions.
- `s_only` layer 26: feature 7982 in 7 positions; feature 18403 in 7 positions; feature 2938 in 6 positions; feature 31664 in 6 positions; feature 15349 in 4 positions; feature 32152 in 4 positions; feature 24419 in 4 positions; feature 2977 in 4 positions.
- `s_c_only` layer 14: feature 2961 in 8 positions; feature 31733 in 7 positions; feature 11831 in 6 positions; feature 5329 in 5 positions; feature 31706 in 5 positions; feature 10804 in 5 positions; feature 2021 in 4 positions; feature 23673 in 4 positions.
- `s_c_only` layer 26: feature 2938 in 8 positions; feature 7982 in 8 positions; feature 11249 in 5 positions; feature 30929 in 5 positions; feature 22815 in 5 positions; feature 2977 in 4 positions; feature 27281 in 4 positions; feature 31664 in 4 positions.
- `e_d_high_impact_only` layer 14: feature 2961 in 7 positions; feature 5455 in 7 positions; feature 31733 in 6 positions; feature 11831 in 5 positions; feature 19440 in 5 positions; feature 13383 in 4 positions; feature 6633 in 4 positions; feature 7065 in 4 positions.
- `e_d_high_impact_only` layer 26: feature 2938 in 7 positions; feature 7982 in 7 positions; feature 18403 in 7 positions; feature 31664 in 6 positions; feature 7448 in 5 positions; feature 2977 in 5 positions; feature 15349 in 4 positions; feature 30911 in 4 positions.
- `e_d_shuffled` layer 14: feature 2961 in 7 positions; feature 31733 in 5 positions; feature 19440 in 5 positions; feature 5455 in 4 positions; feature 32317 in 4 positions; feature 7065 in 4 positions; feature 23427 in 3 positions; feature 20064 in 3 positions.
- `e_d_shuffled` layer 26: feature 2938 in 6 positions; feature 7982 in 6 positions; feature 31664 in 6 positions; feature 18403 in 6 positions; feature 15349 in 4 positions; feature 24672 in 4 positions; feature 32152 in 4 positions; feature 27281 in 4 positions.

## Generated Token 20 Separation

- `d_only` layer 14 at generated_token_20: TopK Jaccard distance versus ASCII = 0.924731.
- `d_only` layer 26 at generated_token_20: TopK Jaccard distance versus ASCII = 0.947368.
- `e_d_high_impact_only` layer 14 at generated_token_20: TopK Jaccard distance versus ASCII = 0.901099.
- `e_d_high_impact_only` layer 26 at generated_token_20: TopK Jaccard distance versus ASCII = 0.913043.
- `e_d_shuffled` layer 14 at generated_token_20: TopK Jaccard distance versus ASCII = 0.958333.
- `e_d_shuffled` layer 26 at generated_token_20: TopK Jaccard distance versus ASCII = 0.936170.
- `e_only` layer 14 at generated_token_20: TopK Jaccard distance versus ASCII = 0.979592.
- `e_only` layer 26 at generated_token_20: TopK Jaccard distance versus ASCII = 0.989899.
- `s_c_only` layer 14 at generated_token_20: TopK Jaccard distance versus ASCII = 0.936170.
- `s_c_only` layer 26 at generated_token_20: TopK Jaccard distance versus ASCII = 0.969072.
- `s_only` layer 14 at generated_token_20: TopK Jaccard distance versus ASCII = 0.936170.
- `s_only` layer 26 at generated_token_20: TopK Jaccard distance versus ASCII = 0.936170.

## Skipped Positions

- No required positions were skipped.
