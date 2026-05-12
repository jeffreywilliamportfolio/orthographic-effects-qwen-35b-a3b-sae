# Full Controlled Perturbation Matrix Summary

Evidence-only summary from the controlled SAE perturbation matrix. No semantic labels are assigned here.

Prompt-position-layer residual captures: 300.
Layer-26 tracked TopK-50 hit rows: 69.
Delta rows versus `ascii_original`: 18833.

## Which Perturbation Types Produce The Largest Feature Deltas Versus ASCII Original?

- `e_to_ē`: mean_abs_delta=0.183657, max_abs_delta=1.70585.
- `s_to_ş`: mean_abs_delta=0.183534, max_abs_delta=2.09642.
- `s_to_ṡ`: mean_abs_delta=0.169271, max_abs_delta=2.54068.
- `random_readable_unicode_control`: mean_abs_delta=0.153486, max_abs_delta=2.28188.
- `d_to_ḑ`: mean_abs_delta=0.0602607, max_abs_delta=0.362222.

## Which Layer Shows Stronger Perturbation Sensitivity?

- Layer 26 has the larger mean abs delta in this run. layer 26: mean_abs_delta=0.18634, mean_topk_jaccard=0.388783; layer 14: mean_abs_delta=0.125214, mean_topk_jaccard=0.378863.

## Are Deltas Concentrated At Final Prompt Token Or Nearby Boundary Positions?

- `final_prompt_token_minus_10`: mean_abs_delta=0.205506, tracked_layer26_hits=1.
- `final_prompt_token_minus_5`: mean_abs_delta=0.194451, tracked_layer26_hits=12.
- `final_prompt_token_minus_2`: mean_abs_delta=0.161876, tracked_layer26_hits=0.
- `final_prompt_token_minus_1`: mean_abs_delta=0.0943469, tracked_layer26_hits=11.
- `final_prompt_token`: mean_abs_delta=0.0941275, tracked_layer26_hits=45.

## Do Handled Controls Behave Like ASCII Original Or Like Byte-Ish Perturbations?

- Handled controls mean_abs_delta=0.183595; byte-ish controls mean_abs_delta=0.114766.
- `e_to_ē`: mean_abs_delta=0.183657, mean_topk_jaccard=0.232863.
- `s_to_ş`: mean_abs_delta=0.183534, mean_topk_jaccard=0.225006.
- `d_to_ḑ`: mean_abs_delta=0.0602607, mean_topk_jaccard=0.727526.
- `s_to_ṡ`: mean_abs_delta=0.169271, mean_topk_jaccard=0.321743.

## Does S-To-Sdot Behave Closer To D-To-Dstroke Or Handled Controls?

- `s_to_ṡ` is closer to handled controls by mean abs delta (0.169271, handled mean 0.183595, `d_to_ḑ` 0.0602607).

## Does Feature 2722 Remain ASCII Original Concentrated?

- Feature 2722 appeared in `ascii_original` prompts and also in `e_to_ē`=1, `random_readable_unicode_control`=1, `s_to_ş`=2, `s_to_ṡ`=2.

## Does Feature 7108 Remain Mostly ASCII Original Concentrated?

- Feature 7108 appeared in `ascii_original` prompts and also in `e_to_ē`=2.

## Does Feature 31784 Behave Like D-To-Dstroke Specific, General Perturbation Sensitive, Or Boundary Sensitive?

- Feature 31784 appeared in `d_to_ḑ` prompts and also in `ascii_original`=5, `e_to_ē`=1, `random_readable_unicode_control`=3, `s_to_ṡ`=2.
- Feature 31784 hit counts by position: `final_prompt_token`=5, `final_prompt_token_minus_1`=11, `final_prompt_token_minus_5`=3.

## Which Prompt Family Is Most Sensitive To Perturbation?

- `just_check_hum`: mean_abs_delta=0.172551.
- `removed_sentence_hum`: mean_abs_delta=0.159708.
- `original_hum`: mean_abs_delta=0.15699.
- `paraphrase_hum`: mean_abs_delta=0.147538.
- `yes_no_hum`: mean_abs_delta=0.140419.

## Strongest Delta Rows

- `paraphrase_hum` `s_to_ṡ` layer 26 final_prompt_token_minus_5 feature 2938: delta=2.54068, abs_delta=2.54068.
- `original_hum` `random_readable_unicode_control` layer 26 final_prompt_token_minus_2 feature 2938: delta=2.28188, abs_delta=2.28188.
- `removed_sentence_hum` `random_readable_unicode_control` layer 26 final_prompt_token_minus_2 feature 2938: delta=2.11285, abs_delta=2.11285.
- `paraphrase_hum` `s_to_ş` layer 26 final_prompt_token_minus_5 feature 2938: delta=2.09642, abs_delta=2.09642.
- `original_hum` `s_to_ş` layer 26 final_prompt_token_minus_10 feature 2938: delta=2.08473, abs_delta=2.08473.
- `just_check_hum` `s_to_ṡ` layer 26 final_prompt_token_minus_5 feature 2938: delta=1.98564, abs_delta=1.98564.
- `removed_sentence_hum` `s_to_ş` layer 26 final_prompt_token_minus_10 feature 2938: delta=1.8966, abs_delta=1.8966.
- `removed_sentence_hum` `s_to_ş` layer 26 final_prompt_token_minus_2 feature 2938: delta=1.78351, abs_delta=1.78351.
- `original_hum` `s_to_ṡ` layer 26 final_prompt_token_minus_5 feature 2938: delta=1.76545, abs_delta=1.76545.
- `just_check_hum` `e_to_ē` layer 26 final_prompt_token_minus_2 feature 28392: delta=-1.70585, abs_delta=1.70585.
- `just_check_hum` `random_readable_unicode_control` layer 26 final_prompt_token_minus_2 feature 28392: delta=-1.70585, abs_delta=1.70585.
- `original_hum` `s_to_ş` layer 26 final_prompt_token_minus_2 feature 2938: delta=1.62644, abs_delta=1.62644.
- `paraphrase_hum` `e_to_ē` layer 26 final_prompt_token_minus_5 feature 2938: delta=1.61781, abs_delta=1.61781.
- `removed_sentence_hum` `s_to_ṡ` layer 26 final_prompt_token_minus_5 feature 2938: delta=1.61541, abs_delta=1.61541.
- `yes_no_hum` `random_readable_unicode_control` layer 26 final_prompt_token_minus_10 feature 23280: delta=1.59058, abs_delta=1.59058.
- `just_check_hum` `s_to_ş` layer 26 final_prompt_token_minus_2 feature 2938: delta=1.56163, abs_delta=1.56163.
- `original_hum` `random_readable_unicode_control` layer 26 final_prompt_token_minus_5 feature 5117: delta=1.52601, abs_delta=1.52601.
- `removed_sentence_hum` `s_to_ṡ` layer 26 final_prompt_token_minus_10 feature 2938: delta=1.5216, abs_delta=1.5216.
- `original_hum` `e_to_ē` layer 26 final_prompt_token_minus_2 feature 28392: delta=-1.51843, abs_delta=1.51843.
- `just_check_hum` `s_to_ṡ` layer 14 final_prompt_token_minus_5 feature 2961: delta=1.49133, abs_delta=1.49133.
