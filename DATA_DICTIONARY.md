# Data Dictionary

## `prompts/full_controlled_perturbation_matrix.tsv`

- `prompt_id`: unique prompt row identifier.
- `base_prompt_family`: matched prompt family.
- `perturbation_type`: orthographic perturbation variant.
- `prompt_text`: prompt text sent to the model.
- `notes`: construction notes.

## `results/full_controlled_perturbation_matrix/topk_features_by_prompt_layer_position.tsv`

- `prompt_id`: prompt identifier.
- `base_prompt_family`: matched prompt family.
- `perturbation_type`: perturbation variant.
- `layer`: Transformer decoder layer index.
- `position_label`: captured boundary position.
- `token_position`: token index in the prompt.
- `token_string`: decoded token at that position.
- `feature_id`: SAE feature ID.
- `activation`: TopK-50 sparse activation value.
- `rank`: rank within TopK-50.
- `prompt_token_count`: prompt token count.

## `results/full_controlled_perturbation_matrix/perturbation_delta_vs_ascii.tsv`

- `base_prompt_family`: matched prompt family.
- `perturbation_type`: perturbation compared against ASCII.
- `layer`: layer index.
- `position_label`: boundary position.
- `feature_id`: SAE feature ID.
- `ascii_activation`: matched ASCII activation.
- `perturbation_activation`: perturbation activation.
- `delta`: perturbation minus ASCII activation.
- `abs_delta`: absolute delta.
- `ascii_rank`: TopK rank in ASCII condition if present.
- `perturbation_rank`: TopK rank in perturbation condition if present.
- `ascii_present`: whether feature appeared in ASCII TopK-50.
- `perturbation_present`: whether feature appeared in perturbation TopK-50.

## `results/full_controlled_perturbation_matrix/topk_jaccard_vs_ascii.tsv`

- `topk_jaccard`: TopK-50 set overlap against matched ASCII prompt.
- `intersection_count`: number of shared TopK-50 features.

## `results/full_controlled_perturbation_matrix/aggregates/`

- `perturbation_rank_by_layer_position.tsv`: perturbation rankings split by layer and position.
- `perturbation_rank_overall_by_layer.tsv`: perturbation rankings by layer.
- `feature_recurrence_by_family.tsv`: feature recurrence across prompt families.
- `layer26_ascii_vs_handled_distinguishing_features.tsv`: layer-26 ASCII-vs-handled evidence filter.
- `tracked_layer26_hit_counts.tsv`: tracked candidate feature hit counts.

