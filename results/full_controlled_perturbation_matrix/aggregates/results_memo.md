# Results Memo: Full Controlled SAE Perturbation Matrix Aggregates

This memo organizes evidence from the existing matrix outputs only. It does not assign semantic labels to features.

## Scope

- Source run: `/workspace/qwen-scope/5-11-26/sae_outputs/full_controlled_perturbation_matrix/`.
- Layers: 26 and 14.
- Metric definitions: Jaccard distance is `1 - topk_jaccard`; recurrence is distinct prompt-family coverage in TopK-50 rows; handled distinction is repeated ASCII-vs-handled activation or presence shift for `e_to_ē` and `s_to_ş` at layer 26.

## Perturbation Ranking

Top perturbation/layer rows by mean absolute delta:
- perturbation_type=e_to_ē; layer=26; mean_abs_delta=0.218984; mean_jaccard_distance=0.764601
- perturbation_type=s_to_ş; layer=26; mean_abs_delta=0.218666; mean_jaccard_distance=0.774839
- perturbation_type=s_to_ṡ; layer=26; mean_abs_delta=0.202645; mean_jaccard_distance=0.659415
- perturbation_type=random_readable_unicode_control; layer=26; mean_abs_delta=0.184479; mean_jaccard_distance=0.588204
- perturbation_type=s_to_ş; layer=14; mean_abs_delta=0.148351; mean_jaccard_distance=0.775149
- perturbation_type=e_to_ē; layer=14; mean_abs_delta=0.148279; mean_jaccard_distance=0.769673
- perturbation_type=s_to_ṡ; layer=14; mean_abs_delta=0.136602; mean_jaccard_distance=0.6971
- perturbation_type=random_readable_unicode_control; layer=14; mean_abs_delta=0.122442; mean_jaccard_distance=0.587836

Top perturbation/layer rows by Jaccard distance:
- perturbation_type=s_to_ş; layer=14; mean_jaccard_distance=0.775149; mean_abs_delta=0.148351
- perturbation_type=s_to_ş; layer=26; mean_jaccard_distance=0.774839; mean_abs_delta=0.218666
- perturbation_type=e_to_ē; layer=14; mean_jaccard_distance=0.769673; mean_abs_delta=0.148279
- perturbation_type=e_to_ē; layer=26; mean_jaccard_distance=0.764601; mean_abs_delta=0.218984
- perturbation_type=s_to_ṡ; layer=14; mean_jaccard_distance=0.6971; mean_abs_delta=0.136602
- perturbation_type=s_to_ṡ; layer=26; mean_jaccard_distance=0.659415; mean_abs_delta=0.202645
- perturbation_type=random_readable_unicode_control; layer=26; mean_jaccard_distance=0.588204; mean_abs_delta=0.184479
- perturbation_type=random_readable_unicode_control; layer=14; mean_jaccard_distance=0.587836; mean_abs_delta=0.122442

Layer-26 mean absolute delta ranking:
- perturbation_type=e_to_ē; mean_abs_delta=0.218984; mean_jaccard_distance=0.764601
- perturbation_type=s_to_ş; mean_abs_delta=0.218666; mean_jaccard_distance=0.774839
- perturbation_type=s_to_ṡ; mean_abs_delta=0.202645; mean_jaccard_distance=0.659415
- perturbation_type=random_readable_unicode_control; mean_abs_delta=0.184479; mean_jaccard_distance=0.588204
- perturbation_type=d_to_ḑ; mean_abs_delta=0.0734392; mean_jaccard_distance=0.269024

Layer-14 mean absolute delta ranking:
- perturbation_type=s_to_ş; mean_abs_delta=0.148351; mean_jaccard_distance=0.775149
- perturbation_type=e_to_ē; mean_abs_delta=0.148279; mean_jaccard_distance=0.769673
- perturbation_type=s_to_ṡ; mean_abs_delta=0.136602; mean_jaccard_distance=0.6971
- perturbation_type=random_readable_unicode_control; mean_abs_delta=0.122442; mean_jaccard_distance=0.587836
- perturbation_type=d_to_ḑ; mean_abs_delta=0.0471365; mean_jaccard_distance=0.275925

## Recurrent Features

Top recurrent layer-26 feature IDs by prompt-family coverage:
- feature_id=2938; prompt_family_count=5; prompt_count=30; prompt_position_count=147; mean_activation=1.4337
- feature_id=28713; prompt_family_count=5; prompt_count=30; prompt_position_count=42; mean_activation=0.506049
- feature_id=6194; prompt_family_count=5; prompt_count=30; prompt_position_count=41; mean_activation=0.329107
- feature_id=18937; prompt_family_count=5; prompt_count=30; prompt_position_count=40; mean_activation=0.626687
- feature_id=26684; prompt_family_count=5; prompt_count=29; prompt_position_count=74; mean_activation=0.319563
- feature_id=1626; prompt_family_count=5; prompt_count=29; prompt_position_count=35; mean_activation=0.322495
- feature_id=7163; prompt_family_count=5; prompt_count=29; prompt_position_count=29; mean_activation=0.301219
- feature_id=30033; prompt_family_count=5; prompt_count=29; prompt_position_count=29; mean_activation=0.237843
- feature_id=18198; prompt_family_count=5; prompt_count=28; prompt_position_count=74; mean_activation=0.315686
- feature_id=10265; prompt_family_count=5; prompt_count=27; prompt_position_count=51; mean_activation=0.222027

Top recurrent layer-14 feature IDs by prompt-family coverage:
- feature_id=2961; prompt_family_count=5; prompt_count=30; prompt_position_count=147; mean_activation=0.880153
- feature_id=20557; prompt_family_count=5; prompt_count=30; prompt_position_count=145; mean_activation=0.266352
- feature_id=6429; prompt_family_count=5; prompt_count=30; prompt_position_count=116; mean_activation=0.246999
- feature_id=13119; prompt_family_count=5; prompt_count=30; prompt_position_count=67; mean_activation=0.148916
- feature_id=3830; prompt_family_count=5; prompt_count=30; prompt_position_count=41; mean_activation=0.458405
- feature_id=2176; prompt_family_count=5; prompt_count=29; prompt_position_count=38; mean_activation=0.170108
- feature_id=1168; prompt_family_count=5; prompt_count=28; prompt_position_count=40; mean_activation=0.178917
- feature_id=17656; prompt_family_count=5; prompt_count=27; prompt_position_count=65; mean_activation=0.184913
- feature_id=20681; prompt_family_count=5; prompt_count=27; prompt_position_count=64; mean_activation=0.139665
- feature_id=15605; prompt_family_count=5; prompt_count=26; prompt_position_count=60; mean_activation=0.150079

## Layer-26 ASCII Versus Handled Controls

Layer-26 feature IDs with the strongest repeated distinction evidence for ASCII versus handled controls:
- feature_id=17; handled_controls_with_presence_shift=2; family_count_with_presence_shift=3; presence_shift_count=5; mean_abs_delta=0.123796
- feature_id=75; handled_controls_with_presence_shift=2; family_count_with_presence_shift=4; presence_shift_count=6; mean_abs_delta=0.300284
- feature_id=234; handled_controls_with_presence_shift=2; family_count_with_presence_shift=2; presence_shift_count=4; mean_abs_delta=0.151113
- feature_id=271; handled_controls_with_presence_shift=2; family_count_with_presence_shift=4; presence_shift_count=4; mean_abs_delta=0.230158
- feature_id=380; handled_controls_with_presence_shift=2; family_count_with_presence_shift=2; presence_shift_count=6; mean_abs_delta=0.208569
- feature_id=428; handled_controls_with_presence_shift=2; family_count_with_presence_shift=2; presence_shift_count=4; mean_abs_delta=0.270952
- feature_id=677; handled_controls_with_presence_shift=2; family_count_with_presence_shift=3; presence_shift_count=5; mean_abs_delta=0.292374
- feature_id=704; handled_controls_with_presence_shift=2; family_count_with_presence_shift=5; presence_shift_count=17; mean_abs_delta=0.210289
- feature_id=741; handled_controls_with_presence_shift=2; family_count_with_presence_shift=2; presence_shift_count=4; mean_abs_delta=0.220036
- feature_id=771; handled_controls_with_presence_shift=2; family_count_with_presence_shift=3; presence_shift_count=6; mean_abs_delta=0.31191
- feature_id=799; handled_controls_with_presence_shift=2; family_count_with_presence_shift=3; presence_shift_count=5; mean_abs_delta=0.312318
- feature_id=805; handled_controls_with_presence_shift=2; family_count_with_presence_shift=3; presence_shift_count=5; mean_abs_delta=0.311105
- feature_id=892; handled_controls_with_presence_shift=2; family_count_with_presence_shift=3; presence_shift_count=3; mean_abs_delta=0.142628
- feature_id=1160; handled_controls_with_presence_shift=2; family_count_with_presence_shift=2; presence_shift_count=4; mean_abs_delta=0.132772
- feature_id=1205; handled_controls_with_presence_shift=2; family_count_with_presence_shift=2; presence_shift_count=4; mean_abs_delta=0.221095
- feature_id=1206; handled_controls_with_presence_shift=2; family_count_with_presence_shift=3; presence_shift_count=5; mean_abs_delta=0.268418
- feature_id=1277; handled_controls_with_presence_shift=2; family_count_with_presence_shift=2; presence_shift_count=4; mean_abs_delta=0.165483
- feature_id=1408; handled_controls_with_presence_shift=2; family_count_with_presence_shift=4; presence_shift_count=6; mean_abs_delta=0.492113
- feature_id=1438; handled_controls_with_presence_shift=2; family_count_with_presence_shift=5; presence_shift_count=7; mean_abs_delta=0.182958
- feature_id=1510; handled_controls_with_presence_shift=2; family_count_with_presence_shift=2; presence_shift_count=3; mean_abs_delta=0.192003

Tracked candidate feature hit totals:
- feature_id=23977; hit_count=32
- feature_id=31784; hit_count=19
- feature_id=2722; hit_count=7
- feature_id=9745; hit_count=7
- feature_id=7108; hit_count=4

## Readout

- The aggregate tables separate perturbation sensitivity by layer and boundary position; use those tables before interpreting individual feature IDs.
- The handled-control distinction table is an evidence filter, not a label set.
- Feature IDs in the recurrence table are ranked by how often they appear across prompt families, not by semantic meaning.
- No model run, steering, Hauhau, llama.cpp, all-layer expansion, or semantic labeling was performed for this postprocessing step.
