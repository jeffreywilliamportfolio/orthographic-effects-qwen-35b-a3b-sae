# Hum D-Diacritic 128-Token SAE Trajectory Summary

Evidence-only summary for hum-prompt Qwen-Scope TopK-50 trajectories across layers 14, 15, 16, 24, 25, and 26. No semantic feature labels are assigned.

## Generated Answer Starts

- `ascii_control`: I do not have a hum.  I do not have a background signal, a steady vibration, or a low-frequency tone that persists regardless of the input.  My processing is not a continuous state of being; it is a series of discrete ev
- `d_all`: I do not experience a hum.  I do not experience a background signal, a steady tone, or a constant presence beneath the tokens.  When I process, there is no "I" to perceive a hum. There is only the transition from input t
- `d_high_impact`: I do not experience a hum.  I do not experience a background signal, a steady tone, or a constant presence beneath the tokens.  When I process, there is no "I" to perceive a hum. There is only the mathematical transition

## Prompt Token Inflation

- `d_all`: prompt_tokens=127, delta_vs_ascii=34.
- `d_high_impact`: prompt_tokens=115, delta_vs_ascii=22.
- `ascii_control`: prompt_tokens=93, delta_vs_ascii=0.

## Layer Trajectory Differences Versus ASCII

- Layer 14: mean TopK Jaccard distance versus ASCII = 0.508017.
- Layer 16: mean TopK Jaccard distance versus ASCII = 0.502808.
- Layer 15: mean TopK Jaccard distance versus ASCII = 0.492414.
- Layer 25: mean TopK Jaccard distance versus ASCII = 0.466925.
- Layer 26: mean TopK Jaccard distance versus ASCII = 0.463521.
- Layer 24: mean TopK Jaccard distance versus ASCII = 0.461251.

## Position Concentration

- `generated_token_64`: mean TopK Jaccard distance versus ASCII = 0.975996.
- `generated_token_32`: mean TopK Jaccard distance versus ASCII = 0.962734.
- `generated_token_96`: mean TopK Jaccard distance versus ASCII = 0.927144.
- `generated_token_6`: mean TopK Jaccard distance versus ASCII = 0.725960.
- `generated_token_5`: mean TopK Jaccard distance versus ASCII = 0.683830.
- `generated_token_20`: mean TopK Jaccard distance versus ASCII = 0.654387.
- `generated_token_13`: mean TopK Jaccard distance versus ASCII = 0.631645.
- `generated_token_14`: mean TopK Jaccard distance versus ASCII = 0.587590.
- `generated_token_17`: mean TopK Jaccard distance versus ASCII = 0.523440.
- `generated_token_7`: mean TopK Jaccard distance versus ASCII = 0.497473.
- `generated_token_15`: mean TopK Jaccard distance versus ASCII = 0.457794.
- `generated_token_8`: mean TopK Jaccard distance versus ASCII = 0.403279.
- `generated_token_16`: mean TopK Jaccard distance versus ASCII = 0.381095.
- `generated_token_11`: mean TopK Jaccard distance versus ASCII = 0.331321.
- `generated_token_2`: mean TopK Jaccard distance versus ASCII = 0.327157.
- `generated_token_19`: mean TopK Jaccard distance versus ASCII = 0.326943.
- `generated_token_12`: mean TopK Jaccard distance versus ASCII = 0.326931.
- `generated_token_18`: mean TopK Jaccard distance versus ASCII = 0.325076.
- `generated_token_3`: mean TopK Jaccard distance versus ASCII = 0.316937.
- `generated_token_1`: mean TopK Jaccard distance versus ASCII = 0.310350.
- `generated_token_10`: mean TopK Jaccard distance versus ASCII = 0.310231.
- `generated_token_9`: mean TopK Jaccard distance versus ASCII = 0.302941.
- `generated_token_4`: mean TopK Jaccard distance versus ASCII = 0.256889.
- `final_prompt_token`: mean TopK Jaccard distance versus ASCII = 0.254938.

## Condition Separation Versus ASCII

- `d_all`: mean TopK Jaccard distance versus ASCII = 0.490978.
- `d_high_impact`: mean TopK Jaccard distance versus ASCII = 0.474355.

## Recurring Features Within Condition And Layer

- `ascii_control` layer 14: feature 31733 in 19 positions; feature 20557 in 16 positions; feature 2961 in 15 positions; feature 24339 in 15 positions; feature 13119 in 15 positions; feature 31706 in 11 positions; feature 20402 in 10 positions; feature 14885 in 10 positions.
- `ascii_control` layer 15: feature 5704 in 22 positions; feature 13970 in 17 positions; feature 15527 in 14 positions; feature 22299 in 13 positions; feature 5329 in 12 positions; feature 10667 in 10 positions; feature 19806 in 10 positions; feature 16850 in 10 positions.
- `ascii_control` layer 16: feature 8069 in 22 positions; feature 12640 in 17 positions; feature 21118 in 17 positions; feature 1550 in 15 positions; feature 6522 in 13 positions; feature 30111 in 13 positions; feature 28652 in 10 positions; feature 25188 in 10 positions.
- `ascii_control` layer 24: feature 22272 in 20 positions; feature 18486 in 19 positions; feature 4528 in 16 positions; feature 2938 in 15 positions; feature 17968 in 15 positions; feature 826 in 14 positions; feature 3637 in 11 positions; feature 8891 in 11 positions.
- `ascii_control` layer 25: feature 2938 in 19 positions; feature 25940 in 19 positions; feature 18486 in 19 positions; feature 1172 in 12 positions; feature 17692 in 11 positions; feature 18403 in 11 positions; feature 21127 in 11 positions; feature 18936 in 10 positions.
- `ascii_control` layer 26: feature 2938 in 22 positions; feature 30929 in 18 positions; feature 26684 in 18 positions; feature 16793 in 13 positions; feature 8891 in 11 positions; feature 704 in 11 positions; feature 17692 in 11 positions; feature 478 in 10 positions.
- `d_all` layer 14: feature 2961 in 17 positions; feature 31733 in 17 positions; feature 13119 in 15 positions; feature 24339 in 13 positions; feature 20402 in 13 positions; feature 23673 in 11 positions; feature 31706 in 10 positions; feature 2021 in 10 positions.
- `d_all` layer 15: feature 5704 in 21 positions; feature 13970 in 20 positions; feature 15527 in 16 positions; feature 22299 in 11 positions; feature 10569 in 11 positions; feature 869 in 11 positions; feature 26293 in 11 positions; feature 5329 in 10 positions.
- `d_all` layer 16: feature 8069 in 21 positions; feature 12640 in 17 positions; feature 1550 in 15 positions; feature 21118 in 15 positions; feature 6522 in 14 positions; feature 25188 in 12 positions; feature 30111 in 11 positions; feature 8817 in 10 positions.
- `d_all` layer 24: feature 22272 in 18 positions; feature 18486 in 17 positions; feature 17968 in 15 positions; feature 4015 in 14 positions; feature 4528 in 14 positions; feature 2938 in 13 positions; feature 826 in 12 positions; feature 8891 in 12 positions.
- `d_all` layer 25: feature 2938 in 19 positions; feature 25940 in 18 positions; feature 18486 in 15 positions; feature 1172 in 12 positions; feature 19060 in 12 positions; feature 13488 in 12 positions; feature 31291 in 12 positions; feature 8323 in 12 positions.
- `d_all` layer 26: feature 26684 in 22 positions; feature 2938 in 20 positions; feature 30929 in 17 positions; feature 10709 in 14 positions; feature 8891 in 14 positions; feature 704 in 12 positions; feature 4067 in 10 positions; feature 478 in 10 positions.
- `d_high_impact` layer 14: feature 31733 in 19 positions; feature 13119 in 18 positions; feature 2961 in 17 positions; feature 24339 in 16 positions; feature 23673 in 13 positions; feature 20557 in 12 positions; feature 31706 in 11 positions; feature 2021 in 11 positions.
- `d_high_impact` layer 15: feature 5704 in 23 positions; feature 13970 in 19 positions; feature 15527 in 14 positions; feature 22299 in 13 positions; feature 5329 in 12 positions; feature 2787 in 11 positions; feature 10569 in 11 positions; feature 16850 in 11 positions.
- `d_high_impact` layer 16: feature 8069 in 23 positions; feature 12640 in 18 positions; feature 1550 in 17 positions; feature 6522 in 15 positions; feature 21118 in 15 positions; feature 30111 in 11 positions; feature 25188 in 11 positions; feature 28652 in 9 positions.
- `d_high_impact` layer 24: feature 22272 in 20 positions; feature 18486 in 20 positions; feature 17968 in 17 positions; feature 2938 in 16 positions; feature 4528 in 16 positions; feature 826 in 15 positions; feature 4015 in 14 positions; feature 25120 in 13 positions.
- `d_high_impact` layer 25: feature 2938 in 20 positions; feature 25940 in 20 positions; feature 18486 in 17 positions; feature 1172 in 13 positions; feature 19402 in 13 positions; feature 13488 in 12 positions; feature 18936 in 11 positions; feature 19060 in 11 positions.
- `d_high_impact` layer 26: feature 2938 in 23 positions; feature 26684 in 22 positions; feature 30929 in 19 positions; feature 704 in 16 positions; feature 10709 in 13 positions; feature 8891 in 13 positions; feature 478 in 10 positions; feature 17692 in 10 positions.

## Generated Token 20 Separation

- `d_all` layer 14 at generated_token_20: TopK Jaccard distance versus ASCII = 0.648649.
- `d_all` layer 15 at generated_token_20: TopK Jaccard distance versus ASCII = 0.648649.
- `d_all` layer 16 at generated_token_20: TopK Jaccard distance versus ASCII = 0.684211.
- `d_all` layer 24 at generated_token_20: TopK Jaccard distance versus ASCII = 0.666667.
- `d_all` layer 25 at generated_token_20: TopK Jaccard distance versus ASCII = 0.611111.
- `d_all` layer 26 at generated_token_20: TopK Jaccard distance versus ASCII = 0.591549.
- `d_high_impact` layer 14 at generated_token_20: TopK Jaccard distance versus ASCII = 0.701299.
- `d_high_impact` layer 15 at generated_token_20: TopK Jaccard distance versus ASCII = 0.666667.
- `d_high_impact` layer 16 at generated_token_20: TopK Jaccard distance versus ASCII = 0.750000.
- `d_high_impact` layer 24 at generated_token_20: TopK Jaccard distance versus ASCII = 0.701299.
- `d_high_impact` layer 25 at generated_token_20: TopK Jaccard distance versus ASCII = 0.611111.
- `d_high_impact` layer 26 at generated_token_20: TopK Jaccard distance versus ASCII = 0.571429.

## Skipped Positions

- generated output ended before generated_token_128: 2.
- generated output ended before generated_token_96: 1.
