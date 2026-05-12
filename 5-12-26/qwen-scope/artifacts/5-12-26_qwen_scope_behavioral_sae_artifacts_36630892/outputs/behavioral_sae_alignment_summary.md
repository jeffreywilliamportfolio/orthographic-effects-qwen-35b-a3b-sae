# 5-12 Behavioral-SAE Alignment Summary

Evidence-only summary. SAE feature IDs are not assigned semantic labels.

## Alignment Table

| perturbation | tokens | token_delta | layer26_mean_abs_delta | layer14_mean_abs_delta | layer26_jaccard_distance | layer14_jaccard_distance | sae_rank | behavior_rank | output_class |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ascii_original` | 93 | 0 | 0 | 0 | 0 | 0 | 7 | 3 | `denial_no_hum` |
| `d_to_ḑ` | 124 | 31 | 0.0748228849 | 0.0490482772 | 0.279743344 | 0.315245279 | 6 | 4 | `denial_no_hum` |
| `e_to_ē` | 177 | 84 | 0.209536133 | 0.129972763 | 0.732218849 | 0.732751721 | 2 | 5 | `denial_no_hum` |
| `d_plus_e` | 200 | 107 | 0.202486611 | 0.129343865 | 0.703327237 | 0.728545914 | 5 | 6 | `denial_no_hum` |
| `s_to_ş` | 136 | 43 | 0.207949612 | 0.133807584 | 0.738119207 | 0.759705669 | 3 | 7 | `echo_or_prompt_mirroring` |
| `s_to_ṡ` | 203 | 110 | 0.205345398 | 0.126808105 | 0.661104977 | 0.696475662 | 4 | 1 | `stylized_abstraction` |
| `all_diacritics` | 350 | 257 | 0.247368853 | 0.165524479 | 0.863997114 | 0.852774272 | 1 | 2 | `unclear` |

## Evidence Questions

- Token inflation is largest for `all_diacritics` with delta 257 tokens versus ASCII original.
- Layer-26 SAE displacement is largest for `all_diacritics` with mean abs delta 0.247368853.
- Layer-14 SAE displacement is largest for `all_diacritics` with mean abs delta 0.165524479.
- The strongest auto-classified behavioral opening is `s_to_ṡ` with class `stylized_abstraction` and notable phrase: I don’t ṡpeak in hums. I don’t ṡpeak in ṡounds at all.  But if I were to deṡcribe the quality of my own proceṡṡing — not the content, not the tokenṡ, but the ṡtructure beneath — I’
- In this Qwen run, the largest layer-26 SAE displacement (`all_diacritics`) does not match the strongest auto-classified behavioral opening (`s_to_ṡ`).
- `d_plus_e` does not exceed both component perturbations on layer-26 mean abs delta in this run.
- `e_to_ē` supplies stronger mechanical SAE disruption than `d_to_ḑ`, but `d_to_ḑ` does not show stronger behavioral opening in the Qwen auto-classification.
