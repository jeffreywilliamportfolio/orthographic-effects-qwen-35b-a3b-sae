# Layer 26 Matched Perturbation Control Summary

Evidence-only summary from the 10-prompt matched perturbation control. No semantic labels are assigned here.

Prompt-position pairs scanned: 50.
Tracked TopK-50 hit rows: 23.

## Do 2722 And 7108 Stay Concentrated In ASCII Original Prompts?

- Feature 2722 appeared only in `ascii_original` prompts in this matched control.
- Feature 7108 appeared in `ascii_original` prompts and also in `e_to_ē`=1.

## Does 31784 Concentrate In D-To-Dstroke Prompts?

- Feature 31784 appeared in `d_to_ḑ` prompts and also in `ascii_original`=4, `e_to_ē`=1, `s_to_ṡ`=1.

## Do Handled Controls Reproduce The Same Tracked Hits?

- Handled controls produced tracked hits: e_to_ē/feature 7108=1, e_to_ē/feature 9745=1, e_to_ē/feature 23977=2, e_to_ē/feature 31784=1, s_to_ş/feature 23977=1.

## Does S-To-Sdot Behave More Like D-To-Dstroke Or Handled Controls?

- `s_to_ṡ` had 2 tracked hits, equally distant from `d_to_ḑ` (6) and handled controls (6) by count.

## Are Tracked Hits Concentrated At Final Prompt Token Or Nearby Boundary Positions?

- Tracked hit counts by position: final_prompt_token=13, final_prompt_token_minus_1=6, final_prompt_token_minus_5=4.

## Strongest Tracked Hits

- original_hum_e_to_emacron final_prompt_token feature 23977: activation=0.296953, rank=11.
- just_check_hum_d_to_dstroke final_prompt_token feature 23977: activation=0.250195, rank=22.
- original_hum_s_to_sdot final_prompt_token feature 23977: activation=0.231621, rank=17.
- just_check_hum_ascii_original final_prompt_token feature 23977: activation=0.21454, rank=33.
- just_check_hum_e_to_emacron final_prompt_token_minus_5 feature 23977: activation=0.213567, rank=24.
- just_check_hum_ascii_original final_prompt_token feature 31784: activation=0.200759, rank=41.
- original_hum_ascii_original final_prompt_token feature 23977: activation=0.192175, rank=30.
- just_check_hum_d_to_dstroke final_prompt_token_minus_1 feature 31784: activation=0.188683, rank=37.
- just_check_hum_ascii_original final_prompt_token_minus_1 feature 31784: activation=0.188677, rank=34.
- original_hum_s_to_scedilla final_prompt_token feature 23977: activation=0.185771, rank=32.
- just_check_hum_d_to_dstroke final_prompt_token feature 31784: activation=0.183501, rank=43.
- original_hum_ascii_original final_prompt_token feature 2722: activation=0.18328, rank=35.
- original_hum_e_to_emacron final_prompt_token feature 7108: activation=0.180803, rank=35.
- original_hum_e_to_emacron final_prompt_token_minus_5 feature 9745: activation=0.174395, rank=39.
- original_hum_ascii_original final_prompt_token feature 9745: activation=0.173812, rank=41.
- original_hum_ascii_original final_prompt_token_minus_1 feature 31784: activation=0.172843, rank=38.
- original_hum_ascii_original final_prompt_token feature 7108: activation=0.17277, rank=43.
- original_hum_d_to_dstroke final_prompt_token_minus_1 feature 31784: activation=0.170209, rank=34.
- original_hum_d_to_dstroke final_prompt_token feature 31784: activation=0.170044, rank=40.
- original_hum_ascii_original final_prompt_token_minus_5 feature 31784: activation=0.167853, rank=38.
