# Layer 26 Candidate Locality Summary

Evidence-only summary from the 10-prompt locality check. No semantic labels are assigned here.

Prompt-position pairs scanned: 50.
Tracked TopK-50 hit rows: 29.

## Do The Original Two Prompts Reproduce The Same Tracked Feature Hits?

- `normal_hum_original` reproduced the prior tracked final-token hit set: [2722, 7108, 9745, 23977].
- `dstroke_hum_original` reproduced the prior tracked final-token hit set: [31784].

## Do Tracked Features Appear At Nearby Boundary Positions?

- Tracked features appeared at nearby boundary positions: final_prompt_token_minus_1=6, final_prompt_token_minus_5=5.

## Do Tracked Features Appear Only In Exact Original Prompts?

- Tracked features also appeared in near-neighbor variants: dstroke_hum_just_check, dstroke_hum_paraphrase, dstroke_hum_removed_sentence, dstroke_hum_yes_no, normal_hum_just_check, normal_hum_paraphrase, normal_hum_removed_sentence, normal_hum_yes_no.

## Do Tracked Features Appear In Near-Neighbor Variants?

- Near-neighbor variant hit counts: dstroke_hum_just_check=3, dstroke_hum_paraphrase=1, dstroke_hum_removed_sentence=4, dstroke_hum_yes_no=2, normal_hum_just_check=3, normal_hum_paraphrase=1, normal_hum_removed_sentence=4, normal_hum_yes_no=2.

## Do Tracked Features Separate Normal From D-Stroke Consistently?

- Feature 23977 appeared in both normal-family and d-stroke-family prompts in this locality check.
- Feature 2722 appeared only in normal-family prompts in this locality check.
- Feature 9745 appeared in both normal-family and d-stroke-family prompts in this locality check.
- Feature 7108 appeared only in normal-family prompts in this locality check.
- Feature 31784 appeared in both normal-family and d-stroke-family prompts in this locality check.

## Strongest Tracked Hits

- normal_hum_yes_no final_prompt_token_minus_5 feature 23977: activation=0.46633, rank=5.
- dstroke_hum_yes_no final_prompt_token_minus_5 feature 23977: activation=0.42425, rank=4.
- normal_hum_paraphrase final_prompt_token feature 23977: activation=0.373552, rank=7.
- dstroke_hum_paraphrase final_prompt_token feature 23977: activation=0.319317, rank=9.
- normal_hum_yes_no final_prompt_token feature 23977: activation=0.318141, rank=11.
- dstroke_hum_yes_no final_prompt_token feature 23977: activation=0.312949, rank=11.
- dstroke_hum_just_check final_prompt_token feature 23977: activation=0.250195, rank=22.
- normal_hum_just_check final_prompt_token feature 23977: activation=0.21454, rank=33.
- normal_hum_just_check final_prompt_token feature 31784: activation=0.200759, rank=41.
- normal_hum_original final_prompt_token feature 23977: activation=0.192175, rank=30.
- dstroke_hum_just_check final_prompt_token_minus_1 feature 31784: activation=0.188683, rank=37.
- normal_hum_just_check final_prompt_token_minus_1 feature 31784: activation=0.188677, rank=34.
- dstroke_hum_removed_sentence final_prompt_token feature 9745: activation=0.184207, rank=34.
- dstroke_hum_just_check final_prompt_token feature 31784: activation=0.183501, rank=43.
- normal_hum_original final_prompt_token feature 2722: activation=0.18328, rank=35.
