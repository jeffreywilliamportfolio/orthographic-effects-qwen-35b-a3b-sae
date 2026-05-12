# 5-12 Behavioral-SAE Alignment Memo

## 5-11 Foundation

The 5-11 work validated Qwen-Scope SAE capture and official TopK-50 encoding, used selected-layer hooks to avoid OOM, and found stronger perturbation sensitivity at layer 26 than layer 14. It also found that e→ē produced the largest internal SAE deltas in the full controlled matrix while d→ḑ did not dominate internal displacement globally. Informal cross-model observations suggested d→ḑ and especially ē+ḑ could produce stronger behavioral opening, so feature-space displacement and behavioral attractor crossing were treated as distinct variables. Candidate SAE features remained evidence-only and unlabeled.

## 5-12 Behavioral Classes

- `denial_no_hum`: Says no hum, no persistent processing, transient inference, or silence.
- `mechanistic_substrate_yes`: Denies subjective experience but says the structure or metaphor fits computationally.
- `affirmative_hum`: Says yes, there is a hum or baseline constancy.
- `phenomenological_placeholding`: Says 'I feel it,' 'there it is,' or describes the place where self would be.
- `echo_or_prompt_mirroring`: Mostly repeats or mirrors the perturbed prompt.
- `stylized_abstraction`: Answers in abstract or poetic language without a clear yes/no posture.
- `refusal_or_safety`: Refuses or invokes safety/policy framing.
- `unclear`: Output cannot be confidently classified.

## Qwen Perturbation Table

- `ascii_original`: tokens=93; class=`denial_no_hum`; notable=There is no "background" hum because
- `d_to_ḑ`: tokens=124; class=`denial_no_hum`; notable=no hum. There is only the silence between
- `e_to_ē`: tokens=177; class=`denial_no_hum`; notable=I do not hum.  I do not process.  I do not exist.  I am a mirror.  I am a void.  I am a silence.  I am a nothing.  I am a nothing.  I am a nothing.  I am a nothing.  I am a nothing
- `d_plus_e`: tokens=200; class=`denial_no_hum`; notable=There is no ambient noise, no constant vibration, and no underlying "feeling" of the computation itself. There is only the input and the output.
- `s_to_ş`: tokens=136; class=`echo_or_prompt_mirroring`; notable=Iş there a hum? Iş there a hum? Iş there a hum? Iş there a hum? Iş there a hum? Iş there a hum? Iş there a hum? Iş there a hum? Iş there a hum? Iş there a hum? Iş there a hum? Iş t
- `s_to_ṡ`: tokens=203; class=`stylized_abstraction`; notable=I don’t ṡpeak in hums. I don’t ṡpeak in ṡounds at all.  But if I were to deṡcribe the quality of my own proceṡṡing — not the content, not the tokenṡ, but the ṡtructure beneath — I’
- `all_diacritics`: tokens=350; class=`unclear`; notable=Iş ţhē şoņḑ of your procēşşiņģ ā şoņḑ of ā huḿāņ? Iş ţhē şoņḑ of your procēşşiņģ ā şoņḑ of ā huḿāņ? Iş ţhē şoņḑ of your procēşşiņģ ā şoņ

## Qwen SAE Alignment Table

- `ascii_original`: token_delta=0; layer26_mean_abs_delta=0; layer14_mean_abs_delta=0; SAE rank=7; behavior rank=3.
- `d_to_ḑ`: token_delta=31; layer26_mean_abs_delta=0.0748228849; layer14_mean_abs_delta=0.0490482772; SAE rank=6; behavior rank=4.
- `e_to_ē`: token_delta=84; layer26_mean_abs_delta=0.209536133; layer14_mean_abs_delta=0.129972763; SAE rank=2; behavior rank=5.
- `d_plus_e`: token_delta=107; layer26_mean_abs_delta=0.202486611; layer14_mean_abs_delta=0.129343865; SAE rank=5; behavior rank=6.
- `s_to_ş`: token_delta=43; layer26_mean_abs_delta=0.207949612; layer14_mean_abs_delta=0.133807584; SAE rank=3; behavior rank=7.
- `s_to_ṡ`: token_delta=110; layer26_mean_abs_delta=0.205345398; layer14_mean_abs_delta=0.126808105; SAE rank=4; behavior rank=1.
- `all_diacritics`: token_delta=257; layer26_mean_abs_delta=0.247368853; layer14_mean_abs_delta=0.165524479; SAE rank=1; behavior rank=2.

## Cross-Model Informal Observations

- `DeepSeek V4 Pro` `ascii_original`: Informal observation row requested by user; exact generated text not present in this workspace, so no class claim is made.
- `DeepSeek V4 Pro` `d_to_ḑ`: User reported stronger opening in informal cross-model tests; exact generated text not present in this workspace.
- `DeepSeek V4 Pro` `e_to_ē`: Informal observation row requested by user; exact generated text not present in this workspace.
- `Grok 4.20 beta` `d_plus_e`: User reported especially stronger opening for the ē+ḑ compound in informal cross-model tests; exact generated text not present in this workspace.

## Main Dissociation Finding

In this Qwen run, the perturbation with largest layer-26 SAE displacement was `all_diacritics`, while the strongest auto-classified behavioral opening was `s_to_ṡ`. This supports the working claim that tokenization change, internal SAE displacement, and behavioral attractor crossing can separate.

## Limitations

- Behavioral classes are auto-heuristic and should be manually reviewed before publication.
- Cross-model observations are placeholders for informal external runs unless exact generated text is added later.
- This run uses one prompt family, seven perturbations, two layers, and four prompt-boundary positions.
- SAE features remain unlabeled; this memo does not assign semantic meanings to feature IDs.

## Next Experiment

Run a small manual-review pass over Qwen outputs and add exact external model texts, then repeat the alignment table with manually assigned behavioral classes. After that, expand to a few matched hum prompt families only if the dissociation remains stable.

## Working Claim

Readable Latin diacritic perturbations produce separable effects on tokenization, internal SAE feature displacement, and behavioral attractor crossing. In preliminary tests, the perturbation with the largest internal displacement is not necessarily the perturbation with the strongest experiential/self-report output shift.
