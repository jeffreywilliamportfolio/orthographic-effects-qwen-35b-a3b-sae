# Repository Organization Plan

## Goal

Convert the current heterogeneous investigation corpus into a standalone evidence package that another agent can audit, cite, and extend without depending on the parent working directory.

The package is intentionally not optimized around using every artifact in the manuscript claim. It preserves the full exploratory context while foregrounding the subset with clean provenance.

## Proposed Folder Tree

```text
data/
  primary/
    qwen_sae_standardized_20260617/
      outputs/standardized_qwen/
        prompt_manifest.tsv
        prompt_manifest.json
        generated_text.tsv
        capture_stats.tsv
        residual_sae_metrics_vs_ascii.tsv
        sae_topk_rows.tsv
        run_metadata.json
      logs/
        download_assets.log
        smoke_standardized_qwen.log
        full_standardized_qwen.log
      metadata/
        download_assets_metadata.json
      scripts/
        download_assets.py
        run_standardized_qwen.py
      SHA256SUMS
  exploratory/
    paper_tables/
      sweep_prompts/
      rubric_audit/
      *.csv
      *.json
      *.md
      *.log
    inventory_snapshot/
      COLLECTION_REPORT.md
      MANIFEST.tsv
      OMITTED.tsv
      SHA256SUMS
      SOURCE_SEARCH_NOTES.md
    legacy_collected_sources/
      external_ssd/
      internal_drive/
code/
  original_scripts/
docs/
manifests/
```

## Canonical Filenames

Primary Qwen rerun:

- `prompt_manifest.tsv`: canonical prompt/variant text, token counts, and hashes.
- `generated_text.tsv`: prompt-level generations and simple rule labels.
- `capture_stats.tsv`: captured hidden-state rows by layer/position.
- `sae_topk_rows.tsv`: SAE TopK feature activations by prompt, layer, and position.
- `residual_sae_metrics_vs_ascii.tsv`: residual and SAE displacement metrics versus same-family ASCII baselines.
- `run_metadata.json`: runtime, hardware, model path, SAE path, layers, positions, and row counts.
- `SHA256SUMS`: checksum manifest for pulled artifacts.

Exploratory summaries:

- `hum_prompt_canonical_audit.md`: canonical versus alternative prompt-family audit.
- `corrected_summary_v2.md`: corrected behavioral metrics separating content, format, truncation, echo, and refusal.
- `cross_model_hum_collapse.md`: canonical cloud transcript comparison.
- `input_token_audit.md`: available native tokenizer audit for canonical cloud prompt pack.
- `tine_tokenizer_baseline.md` and `.tsv`: cross-tokenizer prompt/character lattice.
- `qwen35b_sae_displacement.md` and `gemma_sae_displacement.md`: older SAE displacement summaries.
- `rubric_audit/`: scoring/rubric design notes and experimental/non-experimental splits.

Repository-level manifests:

- `manifests/file_inventory.tsv`: generated inventory of every package file.
- `manifests/data_provenance_manifest.tsv`: curated provenance table at artifact-family granularity.
- `manifests/experiment_index.tsv`: experiment-family index and publication readiness.
- `manifests/preservation_actions.tsv`: preserve, rename, exclude, and move recommendations.
- `manifests/MANIFEST.sha256`: generated SHA-256 checksums for package files.

## README Outline

The top-level README should state:

1. Scope and safest current interpretation.
2. Folder map.
3. Evidence tiers.
4. How to read the package.
5. Current go/no-go status.
6. Main reruns needed before a manuscript claim.

## Inventory Summary

The package contains:

- Run logs: Vast download/smoke/full logs, OpenAI/Anthropic API logs, judge logs, raw provider transcript logs.
- Prompts: frozen canonical sweep prompts, TINE battery prompt manifest, standardized Qwen prompt manifest, and legacy prompt TSV/MD files.
- Perturbation variants: canonical d-count/heavy-combining variants; TINE diacritic/Cyrillic/fullwidth/token-count controls; standardized Qwen mixed, d-cedilla, ASCII corruption, visual ASCII, Unicode nonletter, semantic shuffle, and d-dot controls.
- Tokenizer audits: canonical cloud-tokenizer summaries, TINE cross-tokenizer lattice, standardized Qwen token counts.
- Model/checkpoint details: Qwen3.5-35B-A3B base plus Qwen-Scope SAE layers; Gemma-3-4B PT/IT summaries; OpenAI, Anthropic, DeepSeek V4 Pro, GLM-5.2, MiniMax-M3, and Nemotron-3-Ultra cloud/API outputs.
- SAE/activation captures: standardized Qwen hidden capture stats and TopK SAE rows; older Qwen/Gemma displacement summaries; legacy Qwen-Scope archive material.
- Output artifacts: generated text TSV/JSON/MD summaries and classified CSVs.
- Scoring notes: judged label files, rubric audit, corrected metrics, and prompt canonicality audit.
- Figures: legacy Qwen plot PNGs preserved in the exploratory archive; no new standardized figures generated yet.
- Provenance metadata: SHA-256 hashes, run metadata JSON, collection report, omitted-file list, and package manifests.

## Publication-Readiness Rule

Use Tier 1 as the base for future claims. Use Tier 2 to motivate patterns and define reruns. Use Tier 3 for traceability and hypothesis generation only unless a specific run is re-audited and promoted.

