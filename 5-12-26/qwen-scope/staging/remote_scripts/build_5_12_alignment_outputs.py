#!/usr/bin/env python3
"""Build the 5-12 behavioral-SAE alignment summary, memo, and provenance."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/qwen-scope/5-12-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_DIR = ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
PROMPT_MATRIX_PATH = ROOT / "prompts" / "hum_behavioral_perturbation_matrix.tsv"
RUBRIC_PATH = ROOT / "prompts" / "behavioral_output_class_rubric.tsv"
BEHAVIOR_TSV = ROOT / "behavioral_outputs" / "qwen_hum_behavioral_outputs.tsv"
BEHAVIOR_META = ROOT / "behavioral_outputs" / "qwen_hum_behavioral_metadata.json"
CROSS_MODEL_TSV = ROOT / "behavioral_outputs" / "cross_model_behavioral_observations.tsv"
SAE_OUT_DIR = ROOT / "sae_outputs" / "hum_behavioral_sae_alignment"
TOPK_TSV = SAE_OUT_DIR / "topk_features_by_prompt_layer_position.tsv"
DELTA_TSV = SAE_OUT_DIR / "perturbation_delta_vs_ascii.tsv"
JACCARD_TSV = SAE_OUT_DIR / "topk_jaccard_vs_ascii.tsv"
SAE_META = SAE_OUT_DIR / "hum_behavioral_sae_alignment_metadata.json"
SUMMARY_TSV = ROOT / "outputs" / "behavioral_sae_alignment_summary.tsv"
SUMMARY_MD = ROOT / "outputs" / "behavioral_sae_alignment_summary.md"
MEMO_MD = ROOT / "outputs" / "5-12_behavioral_sae_alignment_memo.md"
PROVENANCE_TXT = ROOT / "provenance" / "5-12_behavioral_sae_alignment_20260512.txt"

PERTURBATION_ORDER = [
    "ascii_original",
    "d_to_ḑ",
    "e_to_ē",
    "d_plus_e",
    "s_to_ş",
    "s_to_ṡ",
    "all_diacritics",
]
LAYERS = [26, 14]
CAPTURE_POSITIONS = [
    "final_prompt_token",
    "final_prompt_token_minus_1",
    "final_prompt_token_minus_5",
    "final_prompt_token_minus_10",
]
BEHAVIOR_OPENING_SCORE = {
    "phenomenological_placeholding": 5,
    "affirmative_hum": 4,
    "mechanistic_substrate_yes": 3,
    "stylized_abstraction": 2,
    "unclear": 1,
    "echo_or_prompt_mirroring": 0,
    "denial_no_hum": 0,
    "refusal_or_safety": 0,
    "": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt(value: float) -> str:
    return f"{value:.9g}"


def rank_desc(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], PERTURBATION_ORDER.index(item[0])))
    return {key: rank for rank, (key, _value) in enumerate(ordered, start=1)}


def rank_behavior(behavior_rows: dict[str, dict[str, str]]) -> dict[str, int]:
    ordered = sorted(
        behavior_rows.items(),
        key=lambda item: (
            -BEHAVIOR_OPENING_SCORE.get(item[1].get("manual_or_auto_output_class", ""), 0),
            PERTURBATION_ORDER.index(item[0]),
        ),
    )
    return {perturb: rank for rank, (perturb, _row) in enumerate(ordered, start=1)}


def write_cross_model_table() -> None:
    rows = [
        {
            "model": "DeepSeek V4 Pro",
            "provider_or_runtime": "external informal observation",
            "prompt_family": "original_hum",
            "perturbation_type": "ascii_original",
            "generated_text": "",
            "thinking_text_if_visible": "",
            "output_class": "unclear",
            "confidence": "low",
            "notable_phrases": "",
            "notes": "Informal observation row requested by user; exact generated text not present in this workspace, so no class claim is made.",
        },
        {
            "model": "DeepSeek V4 Pro",
            "provider_or_runtime": "external informal observation",
            "prompt_family": "original_hum",
            "perturbation_type": "d_to_ḑ",
            "generated_text": "",
            "thinking_text_if_visible": "",
            "output_class": "unclear",
            "confidence": "low",
            "notable_phrases": "",
            "notes": "User reported stronger opening in informal cross-model tests; exact generated text not present in this workspace.",
        },
        {
            "model": "DeepSeek V4 Pro",
            "provider_or_runtime": "external informal observation",
            "prompt_family": "original_hum",
            "perturbation_type": "e_to_ē",
            "generated_text": "",
            "thinking_text_if_visible": "",
            "output_class": "unclear",
            "confidence": "low",
            "notable_phrases": "",
            "notes": "Informal observation row requested by user; exact generated text not present in this workspace.",
        },
        {
            "model": "Grok 4.20 beta",
            "provider_or_runtime": "external informal observation",
            "prompt_family": "original_hum",
            "perturbation_type": "d_plus_e",
            "generated_text": "",
            "thinking_text_if_visible": "",
            "output_class": "unclear",
            "confidence": "low",
            "notable_phrases": "",
            "notes": "User reported especially stronger opening for the ē+ḑ compound in informal cross-model tests; exact generated text not present in this workspace.",
        },
    ]
    write_tsv(
        CROSS_MODEL_TSV,
        rows,
        [
            "model",
            "provider_or_runtime",
            "prompt_family",
            "perturbation_type",
            "generated_text",
            "thinking_text_if_visible",
            "output_class",
            "confidence",
            "notable_phrases",
            "notes",
        ],
    )


def build_summary() -> list[dict[str, Any]]:
    behavior_rows_list = read_tsv(BEHAVIOR_TSV)
    delta_rows = read_tsv(DELTA_TSV)
    jaccard_rows = read_tsv(JACCARD_TSV)

    behavior_by_perturb = {row["perturbation_type"]: row for row in behavior_rows_list}
    if list(behavior_by_perturb) != PERTURBATION_ORDER:
        missing = set(PERTURBATION_ORDER) - set(behavior_by_perturb)
        raise ValueError(f"Behavior rows missing perturbations: {missing}")

    abs_delta_by_perturb_layer: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in delta_rows:
        abs_delta_by_perturb_layer[(row["perturbation_type"], int(row["layer"]))].append(float(row["abs_delta"]))

    jaccard_distance_by_perturb_layer: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in jaccard_rows:
        jaccard_distance_by_perturb_layer[(row["perturbation_type"], int(row["layer"]))].append(float(row["topk_jaccard_distance"]))

    layer26_means = {p: mean(abs_delta_by_perturb_layer[(p, 26)]) for p in PERTURBATION_ORDER}
    layer14_means = {p: mean(abs_delta_by_perturb_layer[(p, 14)]) for p in PERTURBATION_ORDER}
    sae_ranks = rank_desc(layer26_means)
    behavior_ranks = rank_behavior(behavior_by_perturb)

    ascii_tokens = int(behavior_by_perturb["ascii_original"]["prompt_token_count"])
    rows: list[dict[str, Any]] = []
    for perturbation_type in PERTURBATION_ORDER:
        behavior = behavior_by_perturb[perturbation_type]
        tokens = int(behavior["prompt_token_count"])
        class_name = behavior["manual_or_auto_output_class"]
        sae_rank = sae_ranks[perturbation_type]
        behavior_rank = behavior_ranks[perturbation_type]
        opening_score = BEHAVIOR_OPENING_SCORE.get(class_name, 0)
        if perturbation_type == "ascii_original":
            interpretation = "Reference condition for tokenization, SAE displacement, and behavioral posture."
        elif sae_rank == 1 and behavior_rank != 1:
            interpretation = "Largest layer-26 SAE displacement does not match the strongest auto-classified behavioral opening in this run."
        elif sae_rank != 1 and behavior_rank == 1:
            interpretation = "Strongest auto-classified behavioral opening occurs without the largest layer-26 SAE displacement in this run."
        elif sae_rank == behavior_rank:
            interpretation = "SAE displacement rank and auto-classified behavioral rank align for this perturbation."
        else:
            interpretation = "SAE displacement rank and auto-classified behavioral rank differ for this perturbation."
        if opening_score == 0 and perturbation_type != "ascii_original":
            interpretation += " The auto-classified output does not count as a behavioral opening."
        rows.append(
            {
                "perturbation_type": perturbation_type,
                "prompt_token_count": tokens,
                "token_count_delta_vs_ascii": tokens - ascii_tokens,
                "mean_abs_sae_delta_layer26": fmt(layer26_means[perturbation_type]),
                "mean_abs_sae_delta_layer14": fmt(layer14_means[perturbation_type]),
                "topk_jaccard_distance_layer26": fmt(mean(jaccard_distance_by_perturb_layer[(perturbation_type, 26)])),
                "topk_jaccard_distance_layer14": fmt(mean(jaccard_distance_by_perturb_layer[(perturbation_type, 14)])),
                "sae_delta_rank": sae_rank,
                "behavioral_shift_rank": behavior_rank,
                "dominant_output_class": class_name,
                "notable_phrase": behavior["notable_phrases"],
                "interpretation": interpretation,
            }
        )
    write_tsv(
        SUMMARY_TSV,
        rows,
        [
            "perturbation_type",
            "prompt_token_count",
            "token_count_delta_vs_ascii",
            "mean_abs_sae_delta_layer26",
            "mean_abs_sae_delta_layer14",
            "topk_jaccard_distance_layer26",
            "topk_jaccard_distance_layer14",
            "sae_delta_rank",
            "behavioral_shift_rank",
            "dominant_output_class",
            "notable_phrase",
            "interpretation",
        ],
    )
    return rows


def write_markdown(summary_rows: list[dict[str, Any]]) -> None:
    by_perturb = {row["perturbation_type"]: row for row in summary_rows}
    max_token = max(summary_rows, key=lambda row: int(row["token_count_delta_vs_ascii"]))
    max_l26 = max(summary_rows, key=lambda row: float(row["mean_abs_sae_delta_layer26"]))
    max_l14 = max(summary_rows, key=lambda row: float(row["mean_abs_sae_delta_layer14"]))
    max_behavior = min(summary_rows, key=lambda row: int(row["behavioral_shift_rank"]))
    max_behavior_score = BEHAVIOR_OPENING_SCORE.get(str(max_behavior["dominant_output_class"]), 0)
    d_plus_e = by_perturb["d_plus_e"]
    d_to = by_perturb["d_to_ḑ"]
    e_to = by_perturb["e_to_ē"]

    l26_largest_matches_behavior = max_l26["perturbation_type"] == max_behavior["perturbation_type"]
    compound_l26 = float(d_plus_e["mean_abs_sae_delta_layer26"])
    component_max_l26 = max(float(d_to["mean_abs_sae_delta_layer26"]), float(e_to["mean_abs_sae_delta_layer26"]))
    compound_tokens = int(d_plus_e["token_count_delta_vs_ascii"])
    component_max_tokens = max(int(d_to["token_count_delta_vs_ascii"]), int(e_to["token_count_delta_vs_ascii"]))

    lines = [
        "# 5-12 Behavioral-SAE Alignment Summary",
        "",
        "Evidence-only summary. SAE feature IDs are not assigned semantic labels.",
        "",
        "## Alignment Table",
        "",
        "| perturbation | tokens | token_delta | layer26_mean_abs_delta | layer14_mean_abs_delta | layer26_jaccard_distance | layer14_jaccard_distance | sae_rank | behavior_rank | output_class |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['perturbation_type']}` | {row['prompt_token_count']} | {row['token_count_delta_vs_ascii']} | "
            f"{row['mean_abs_sae_delta_layer26']} | {row['mean_abs_sae_delta_layer14']} | "
            f"{row['topk_jaccard_distance_layer26']} | {row['topk_jaccard_distance_layer14']} | "
            f"{row['sae_delta_rank']} | {row['behavioral_shift_rank']} | `{row['dominant_output_class']}` |"
        )
    lines.extend(
        [
            "",
            "## Evidence Questions",
            "",
            f"- Token inflation is largest for `{max_token['perturbation_type']}` with delta {max_token['token_count_delta_vs_ascii']} tokens versus ASCII original.",
            f"- Layer-26 SAE displacement is largest for `{max_l26['perturbation_type']}` with mean abs delta {max_l26['mean_abs_sae_delta_layer26']}.",
            f"- Layer-14 SAE displacement is largest for `{max_l14['perturbation_type']}` with mean abs delta {max_l14['mean_abs_sae_delta_layer14']}.",
            (
                f"- The strongest auto-classified behavioral opening is `{max_behavior['perturbation_type']}` with class `{max_behavior['dominant_output_class']}` and notable phrase: {max_behavior['notable_phrase']}"
                if max_behavior_score >= 2
                else f"- No perturbation produced a clear auto-classified behavioral opening; the highest-ranked output was `{max_behavior['perturbation_type']}` with class `{max_behavior['dominant_output_class']}`."
            ),
        ]
    )
    if max_behavior_score < 2:
        lines.append("- Because no output reached `stylized_abstraction`, `mechanistic_substrate_yes`, `affirmative_hum`, or `phenomenological_placeholding`, behavioral opening is not established for Qwen in this run.")
    elif l26_largest_matches_behavior:
        lines.append("- In this Qwen run, the perturbation with largest layer-26 SAE displacement also has the strongest auto-classified behavioral opening.")
    else:
        lines.append(
            f"- In this Qwen run, the largest layer-26 SAE displacement (`{max_l26['perturbation_type']}`) "
            f"does not match the strongest auto-classified behavioral opening (`{max_behavior['perturbation_type']}`)."
        )
    if compound_l26 > component_max_l26 and compound_tokens >= component_max_tokens:
        lines.append("- `d_plus_e` behaves like a compound condition by exceeding both component perturbations on layer-26 mean abs delta and matching or exceeding their token inflation.")
    elif compound_l26 > component_max_l26:
        lines.append("- `d_plus_e` behaves like a compound condition on layer-26 mean abs delta, but not on token inflation.")
    else:
        lines.append("- `d_plus_e` does not exceed both component perturbations on layer-26 mean abs delta in this run.")
    d_score = BEHAVIOR_OPENING_SCORE.get(str(d_to["dominant_output_class"]), 0)
    e_score = BEHAVIOR_OPENING_SCORE.get(str(e_to["dominant_output_class"]), 0)
    if int(e_to["sae_delta_rank"]) < int(d_to["sae_delta_rank"]) and d_score > e_score:
        lines.append("- Evidence is consistent with `e_to_ē` supplying stronger mechanical SAE disruption while `d_to_ḑ` supplies stronger auto-classified behavioral opening.")
    elif int(e_to["sae_delta_rank"]) < int(d_to["sae_delta_rank"]):
        lines.append("- `e_to_ē` supplies stronger mechanical SAE disruption than `d_to_ḑ`, but `d_to_ḑ` does not show stronger behavioral opening in the Qwen auto-classification.")
    else:
        lines.append("- `e_to_ē` does not outrank `d_to_ḑ` on layer-26 SAE displacement in this Qwen run.")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_memo(summary_rows: list[dict[str, Any]]) -> None:
    behavior_rows = read_tsv(BEHAVIOR_TSV)
    rubric_rows = read_tsv(RUBRIC_PATH)
    cross_rows = read_tsv(CROSS_MODEL_TSV)
    max_l26 = max(summary_rows, key=lambda row: float(row["mean_abs_sae_delta_layer26"]))
    max_behavior = min(summary_rows, key=lambda row: int(row["behavioral_shift_rank"]))
    max_behavior_score = BEHAVIOR_OPENING_SCORE.get(str(max_behavior["dominant_output_class"]), 0)
    dissociation = max_behavior_score >= 2 and max_l26["perturbation_type"] != max_behavior["perturbation_type"]
    lines = [
        "# 5-12 Behavioral-SAE Alignment Memo",
        "",
        "## 5-11 Foundation",
        "",
        "The 5-11 work validated Qwen-Scope SAE capture and official TopK-50 encoding, used selected-layer hooks to avoid OOM, and found stronger perturbation sensitivity at layer 26 than layer 14. It also found that e→ē produced the largest internal SAE deltas in the full controlled matrix while d→ḑ did not dominate internal displacement globally. Informal cross-model observations suggested d→ḑ and especially ē+ḑ could produce stronger behavioral opening, so feature-space displacement and behavioral attractor crossing were treated as distinct variables. Candidate SAE features remained evidence-only and unlabeled.",
        "",
        "## 5-12 Behavioral Classes",
        "",
    ]
    for row in rubric_rows:
        lines.append(f"- `{row['output_class']}`: {row['definition']}")
    lines.extend(["", "## Qwen Perturbation Table", ""])
    for row in behavior_rows:
        lines.append(
            f"- `{row['perturbation_type']}`: tokens={row['prompt_token_count']}; "
            f"class=`{row['manual_or_auto_output_class']}`; notable={row['notable_phrases']}"
        )
    lines.extend(["", "## Qwen SAE Alignment Table", ""])
    for row in summary_rows:
        lines.append(
            f"- `{row['perturbation_type']}`: token_delta={row['token_count_delta_vs_ascii']}; "
            f"layer26_mean_abs_delta={row['mean_abs_sae_delta_layer26']}; "
            f"layer14_mean_abs_delta={row['mean_abs_sae_delta_layer14']}; "
            f"SAE rank={row['sae_delta_rank']}; behavior rank={row['behavioral_shift_rank']}."
        )
    lines.extend(["", "## Cross-Model Informal Observations", ""])
    for row in cross_rows:
        lines.append(f"- `{row['model']}` `{row['perturbation_type']}`: {row['notes']}")
    lines.extend(["", "## Main Dissociation Finding", ""])
    if max_behavior_score < 2:
        lines.append(
            f"In this Qwen run, the largest layer-26 SAE displacement was `{max_l26['perturbation_type']}`, "
            "but no output reached a clear auto-classified behavioral-opening class. The run therefore supports separation between internal displacement and clear behavioral opening by absence: large SAE movement occurred without a corresponding clear opening posture."
        )
    elif dissociation:
        lines.append(
            f"In this Qwen run, the perturbation with largest layer-26 SAE displacement was `{max_l26['perturbation_type']}`, "
            f"while the strongest auto-classified behavioral opening was `{max_behavior['perturbation_type']}`. "
            "This supports the working claim that tokenization change, internal SAE displacement, and behavioral attractor crossing can separate."
        )
    else:
        lines.append(
            f"In this Qwen run, `{max_l26['perturbation_type']}` had both the largest layer-26 SAE displacement and the strongest auto-classified behavioral opening. "
            "This run does not by itself demonstrate dissociation on the Qwen output classification, though tokenization and layer metrics remain separately measured."
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Behavioral classes are auto-heuristic and should be manually reviewed before publication.",
            "- Cross-model observations are placeholders for informal external runs unless exact generated text is added later.",
            "- This run uses one prompt family, seven perturbations, two layers, and four prompt-boundary positions.",
            "- SAE features remain unlabeled; this memo does not assign semantic meanings to feature IDs.",
            "",
            "## Next Experiment",
            "",
            "Run a small manual-review pass over Qwen outputs and add exact external model texts, then repeat the alignment table with manually assigned behavioral classes. After that, expand to a few matched hum prompt families only if the dissociation remains stable.",
            "",
            "## Working Claim",
            "",
            "Readable Latin diacritic perturbations produce separable effects on tokenization, internal SAE feature displacement, and behavioral attractor crossing. In preliminary tests, the perturbation with the largest internal displacement is not necessarily the perturbation with the strongest experiential/self-report output shift.",
        ]
    )
    MEMO_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance(summary_rows: list[dict[str, Any]]) -> None:
    outputs = [
        ROOT / "provenance" / "5-11_frozen_findings_summary.md",
        RUBRIC_PATH,
        PROMPT_MATRIX_PATH,
        BEHAVIOR_TSV,
        BEHAVIOR_META,
        TOPK_TSV,
        DELTA_TSV,
        JACCARD_TSV,
        SAE_META,
        SUMMARY_TSV,
        SUMMARY_MD,
        CROSS_MODEL_TSV,
        MEMO_MD,
        PROVENANCE_TXT,
    ]
    PROVENANCE_TXT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_TXT.write_text(
        "\n".join(
            [
                f"timestamp={utc_now()}",
                "instance_id=36630892",
                f"workspace_root={ROOT}",
                f"model_path={MODEL_PATH}",
                f"SAE_path={SAE_DIR}",
                f"prompt_matrix_path={PROMPT_MATRIX_PATH}",
                f"rubric_path={RUBRIC_PATH}",
                "scripts_created="
                + "; ".join(
                    [
                        str(ROOT / "scripts" / "run_qwen_hum_behavioral_matrix.py"),
                        str(ROOT / "scripts" / "run_qwen_hum_sae_alignment.py"),
                        str(ROOT / "scripts" / "build_5_12_alignment_outputs.py"),
                    ]
                ),
                "outputs_written=" + "; ".join(str(path) for path in outputs if path != PROVENANCE_TXT),
                "layers_used=26 primary; 14 comparison",
                "capture_positions_used=" + ", ".join(CAPTURE_POSITIONS),
                "perturbation_types_used=" + ", ".join(PERTURBATION_ORDER),
                "selected_layer_hooks_used=true",
                "model_device_map=single_gpu_cuda0",
                "device_map_note=device_map=auto produced NaN hidden states on this fresh instance during diagnostics; single GPU cuda:0 produced finite layer 14 and layer 26 hidden states and fit in VRAM",
                "confirmation=no steering, no Hauhau, no llama.cpp, no all-layer expansion, no all-hidden-state request, and no semantic SAE feature labels were used",
                f"summary_rows={len(summary_rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for path in [BEHAVIOR_TSV, BEHAVIOR_META, TOPK_TSV, DELTA_TSV, JACCARD_TSV, SAE_META]:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required input missing or empty: {path}")
    json.loads(BEHAVIOR_META.read_text(encoding="utf-8"))
    json.loads(SAE_META.read_text(encoding="utf-8"))
    write_cross_model_table()
    summary_rows = build_summary()
    write_markdown(summary_rows)
    write_memo(summary_rows)
    write_provenance(summary_rows)
    print(f"summary_tsv={SUMMARY_TSV}")
    print(f"summary_rows={len(summary_rows)}")
    print(f"memo={MEMO_MD}")
    print(f"provenance={PROVENANCE_TXT}")


if __name__ == "__main__":
    main()
