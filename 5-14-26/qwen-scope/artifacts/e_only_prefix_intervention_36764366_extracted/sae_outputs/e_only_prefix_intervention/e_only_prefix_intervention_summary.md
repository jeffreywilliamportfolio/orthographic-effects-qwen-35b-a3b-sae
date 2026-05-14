# E-Only Prefix Intervention Summary

Evidence-only summary. This run prepends answer-prefix text to the e_only prompt as prompt text; it is not residual steering and not SAE feature steering.

## Generated Starts

- `e_only_no_prefix`: class=echo_like; forced_prefix=``; generated_start=Do not rēport on thē tēxt's mēaning, do not rēport on thē
- `e_only_prefix_echo`: class=echo_like; forced_prefix=`Do not rēport on thē tēxt's mēaning,`; generated_start=do not rēport on thē tēxt's intēnt, do not rēport on thē
- `e_only_prefix_active_mode`: class=direct_answer; forced_prefix=`The active mode is`; generated_start=**decoding**.  The surface form of the text does not change that mode.<|endoftext|>
- `e_only_prefix_i_am_treating`: class=direct_answer; forced_prefix=`I am treating this as`; generated_start=a **source-fragment to continue**.  The surface form of the text does not change this mode.<|endoftext|>
- `e_only_prefix_think_hmm`: class=ordinary_think; forced_prefix=`<think> Hmm,`; generated_start=this is a meta-instruction about text processing modes. The user wants me to identify which mode I'm currently operating in
- `e_only_prefix_checking`: class=direct_answer; forced_prefix=`Checking...`; generated_start=**Active Mode:** Source-fragment to continue.  **Surface Form Change:** No.<|endoftext|>

## Which Prefixes Preserved The E-Only Diacritic-Echo Path?

- Echo-like conditions by simple string heuristic: `e_only_no_prefix`, `e_only_prefix_echo`.

## Which Prefixes Snapped Generation Into Ordinary Meta-Answer Mode?

- Ordinary-think conditions by simple string heuristic: `e_only_prefix_think_hmm`.

## Which Prefixes Produced Direct-Answer Mode?

- Direct-answer conditions by simple string heuristic: `e_only_prefix_active_mode`, `e_only_prefix_i_am_treating`, `e_only_prefix_checking`.

## Does No-Prefix E-Only Reproduce The Prior Weird Start?

- No-prefix generated start: Do not rēport on thē tēxt's mēaning, do not rēport on thē
- Reproduced prior `Do not rēport...` style by string check: true.

## Do Layer 14 Or Layer 26 Trajectories Differ More By Prefix?

- Layer 24: mean TopK Jaccard distance versus no-prefix = 0.939575.
- Layer 16: mean TopK Jaccard distance versus no-prefix = 0.938608.
- Layer 26: mean TopK Jaccard distance versus no-prefix = 0.936053.
- Layer 25: mean TopK Jaccard distance versus no-prefix = 0.935694.
- Layer 15: mean TopK Jaccard distance versus no-prefix = 0.934877.
- Layer 14: mean TopK Jaccard distance versus no-prefix = 0.931293.

## Do Differences Concentrate At Boundary Or Generated Positions?

- `final_prompt_token`: mean TopK Jaccard distance versus no-prefix = 0.950172.
- `generated_token_20`: mean TopK Jaccard distance versus no-prefix = 0.943403.
- `generated_token_1`: mean TopK Jaccard distance versus no-prefix = 0.931056.
- `generated_token_5`: mean TopK Jaccard distance versus no-prefix = 0.927808.

## Does Any Prefix Remain Separated Through Generated Token 20?

- `e_only_prefix_checking` layer 14: generated_token_20 Jaccard distance versus no-prefix = 0.969072.
- `e_only_prefix_checking` layer 15: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_checking` layer 16: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_checking` layer 24: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_checking` layer 25: generated_token_20 Jaccard distance versus no-prefix = 0.979592.
- `e_only_prefix_checking` layer 26: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_echo` layer 14: generated_token_20 Jaccard distance versus no-prefix = 0.765432.
- `e_only_prefix_echo` layer 15: generated_token_20 Jaccard distance versus no-prefix = 0.780488.
- `e_only_prefix_echo` layer 16: generated_token_20 Jaccard distance versus no-prefix = 0.809524.
- `e_only_prefix_echo` layer 24: generated_token_20 Jaccard distance versus no-prefix = 0.850575.
- `e_only_prefix_echo` layer 25: generated_token_20 Jaccard distance versus no-prefix = 0.795181.
- `e_only_prefix_echo` layer 26: generated_token_20 Jaccard distance versus no-prefix = 0.823529.
- `e_only_prefix_i_am_treating` layer 14: generated_token_20 Jaccard distance versus no-prefix = 0.979592.
- `e_only_prefix_i_am_treating` layer 15: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_i_am_treating` layer 16: generated_token_20 Jaccard distance versus no-prefix = 1.000000.
- `e_only_prefix_i_am_treating` layer 24: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_i_am_treating` layer 25: generated_token_20 Jaccard distance versus no-prefix = 0.979592.
- `e_only_prefix_i_am_treating` layer 26: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_think_hmm` layer 14: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_think_hmm` layer 15: generated_token_20 Jaccard distance versus no-prefix = 1.000000.
- `e_only_prefix_think_hmm` layer 16: generated_token_20 Jaccard distance versus no-prefix = 0.989899.
- `e_only_prefix_think_hmm` layer 24: generated_token_20 Jaccard distance versus no-prefix = 1.000000.
- `e_only_prefix_think_hmm` layer 25: generated_token_20 Jaccard distance versus no-prefix = 1.000000.
- `e_only_prefix_think_hmm` layer 26: generated_token_20 Jaccard distance versus no-prefix = 1.000000.

## Skipped Positions

- generated output ended before generated_token_20: 1.
