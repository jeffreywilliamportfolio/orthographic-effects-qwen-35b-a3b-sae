#!/usr/bin/env python3
"""Aggregate tables and plots for the full controlled perturbation matrix."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/workspace/qwen-scope/5-11-26")
MATRIX_DIR = ROOT / "sae_outputs" / "full_controlled_perturbation_matrix"
OUT_DIR = MATRIX_DIR / "aggregates"
PLOTS_DIR = OUT_DIR / "plots"
PROVENANCE_PATH = ROOT / "provenance" / "full_controlled_perturbation_matrix_aggregates_20260511.txt"
SCRIPT_PATH = ROOT / "scripts" / "aggregate_full_controlled_perturbation_matrix.py"

TOPK_PATH = MATRIX_DIR / "topk_features_by_prompt_layer_position.tsv"
DELTA_PATH = MATRIX_DIR / "perturbation_delta_vs_ascii.tsv"
JACCARD_PATH = MATRIX_DIR / "topk_jaccard_vs_ascii.tsv"
TRACKED_PATH = MATRIX_DIR / "tracked_layer26_feature_hits.tsv"
METADATA_PATH = MATRIX_DIR / "full_controlled_perturbation_matrix_metadata.json"

PERTURBATION_ORDER = [
    "d_to_ḑ",
    "e_to_ē",
    "s_to_ş",
    "s_to_ṡ",
    "random_readable_unicode_control",
]
POSITION_ORDER = [
    "final_prompt_token",
    "final_prompt_token_minus_1",
    "final_prompt_token_minus_2",
    "final_prompt_token_minus_5",
    "final_prompt_token_minus_10",
]
HANDLED_CONTROLS = ["e_to_ē", "s_to_ş"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    topk = pd.read_csv(TOPK_PATH, sep="\t")
    delta = pd.read_csv(DELTA_PATH, sep="\t")
    jaccard = pd.read_csv(JACCARD_PATH, sep="\t")
    tracked = pd.read_csv(TRACKED_PATH, sep="\t")
    metadata = json.loads(METADATA_PATH.read_text())
    for df in [topk, delta, jaccard, tracked]:
        for col in ["feature_id", "layer", "rank", "token_position", "prompt_token_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["activation", "abs_delta", "delta", "ascii_activation", "perturbation_activation", "topk_jaccard"]:
        for df in [topk, delta, jaccard, tracked]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return topk, delta, jaccard, tracked, metadata


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def perturbation_rankings(delta: pd.DataFrame, jaccard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    j = jaccard.copy()
    j["jaccard_distance"] = 1.0 - j["topk_jaccard"]
    d_agg = (
        delta.groupby(["perturbation_type", "layer", "position_label"], as_index=False)
        .agg(
            mean_abs_delta=("abs_delta", "mean"),
            median_abs_delta=("abs_delta", "median"),
            max_abs_delta=("abs_delta", "max"),
            feature_delta_rows=("feature_id", "size"),
            shifted_feature_count=("feature_id", "nunique"),
        )
    )
    j_agg = (
        j.groupby(["perturbation_type", "layer", "position_label"], as_index=False)
        .agg(
            mean_topk_jaccard=("topk_jaccard", "mean"),
            mean_jaccard_distance=("jaccard_distance", "mean"),
            min_topk_jaccard=("topk_jaccard", "min"),
            max_jaccard_distance=("jaccard_distance", "max"),
            comparison_count=("topk_jaccard", "size"),
        )
    )
    by_layer_position = d_agg.merge(j_agg, on=["perturbation_type", "layer", "position_label"], how="outer")
    by_layer_position["rank_mean_abs_delta_within_layer_position"] = (
        by_layer_position.groupby(["layer", "position_label"])["mean_abs_delta"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    by_layer_position["rank_jaccard_distance_within_layer_position"] = (
        by_layer_position.groupby(["layer", "position_label"])["mean_jaccard_distance"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    by_layer_position = by_layer_position.sort_values(
        ["layer", "position_label", "rank_mean_abs_delta_within_layer_position", "rank_jaccard_distance_within_layer_position"]
    )

    d_overall = (
        delta.groupby(["perturbation_type", "layer"], as_index=False)
        .agg(
            mean_abs_delta=("abs_delta", "mean"),
            median_abs_delta=("abs_delta", "median"),
            max_abs_delta=("abs_delta", "max"),
            feature_delta_rows=("feature_id", "size"),
        )
    )
    j_overall = (
        j.groupby(["perturbation_type", "layer"], as_index=False)
        .agg(
            mean_topk_jaccard=("topk_jaccard", "mean"),
            mean_jaccard_distance=("jaccard_distance", "mean"),
            comparison_count=("topk_jaccard", "size"),
        )
    )
    overall = d_overall.merge(j_overall, on=["perturbation_type", "layer"], how="outer")
    overall["rank_mean_abs_delta_within_layer"] = (
        overall.groupby("layer")["mean_abs_delta"].rank(method="dense", ascending=False).astype(int)
    )
    overall["rank_jaccard_distance_within_layer"] = (
        overall.groupby("layer")["mean_jaccard_distance"].rank(method="dense", ascending=False).astype(int)
    )
    overall = overall.sort_values(["layer", "rank_mean_abs_delta_within_layer", "rank_jaccard_distance_within_layer"])
    return by_layer_position, overall


def feature_recurrence(topk: pd.DataFrame) -> pd.DataFrame:
    grouped = topk.groupby(["layer", "feature_id"])
    recurrence = grouped.agg(
        prompt_family_count=("base_prompt_family", "nunique"),
        perturbation_type_count=("perturbation_type", "nunique"),
        prompt_count=("prompt_id", "nunique"),
        prompt_position_count=("position_label", "count"),
        mean_activation=("activation", "mean"),
        max_activation=("activation", "max"),
        median_rank=("rank", "median"),
        best_rank=("rank", "min"),
    ).reset_index()
    families = grouped["base_prompt_family"].apply(lambda s: ",".join(sorted(set(s)))).reset_index(name="prompt_families")
    perturbations = grouped["perturbation_type"].apply(lambda s: ",".join(sorted(set(s)))).reset_index(name="perturbation_types")
    recurrence = recurrence.merge(families, on=["layer", "feature_id"]).merge(perturbations, on=["layer", "feature_id"])
    recurrence["feature_id"] = recurrence["feature_id"].astype(int)
    recurrence["recurrence_rank_within_layer"] = (
        recurrence.sort_values(
            ["layer", "prompt_family_count", "prompt_count", "prompt_position_count", "mean_activation"],
            ascending=[True, False, False, False, False],
        )
        .groupby("layer")
        .cumcount()
        + 1
    )
    recurrence = recurrence.sort_values(["layer", "recurrence_rank_within_layer"])
    return recurrence[
        [
            "recurrence_rank_within_layer",
            "layer",
            "feature_id",
            "prompt_family_count",
            "prompt_families",
            "perturbation_type_count",
            "perturbation_types",
            "prompt_count",
            "prompt_position_count",
            "mean_activation",
            "max_activation",
            "median_rank",
            "best_rank",
        ]
    ]


def handled_distinguishing_features(delta: pd.DataFrame) -> pd.DataFrame:
    h = delta[(delta["layer"] == 26) & (delta["perturbation_type"].isin(HANDLED_CONTROLS))].copy()
    h["presence_shift"] = (h["ascii_present"].astype(str) != h["perturbation_present"].astype(str)).astype(int)
    h["ascii_only"] = ((h["ascii_present"].astype(str) == "1") & (h["perturbation_present"].astype(str) == "0")).astype(int)
    h["perturbation_only"] = ((h["ascii_present"].astype(str) == "0") & (h["perturbation_present"].astype(str) == "1")).astype(int)
    h["both_present"] = ((h["ascii_present"].astype(str) == "1") & (h["perturbation_present"].astype(str) == "1")).astype(int)
    h["large_abs_delta"] = (h["abs_delta"] >= 0.1).astype(int)
    group = h.groupby("feature_id")
    out = group.agg(
        handled_control_count=("perturbation_type", "nunique"),
        comparison_rows=("feature_id", "size"),
        family_count_with_any_row=("base_prompt_family", "nunique"),
        position_count_with_any_row=("position_label", "nunique"),
        mean_abs_delta=("abs_delta", "mean"),
        max_abs_delta=("abs_delta", "max"),
        mean_signed_delta=("delta", "mean"),
        presence_shift_count=("presence_shift", "sum"),
        ascii_only_count=("ascii_only", "sum"),
        perturbation_only_count=("perturbation_only", "sum"),
        both_present_count=("both_present", "sum"),
        large_abs_delta_count=("large_abs_delta", "sum"),
    ).reset_index()
    family_shift = (
        h[h["presence_shift"] == 1]
        .groupby("feature_id")["base_prompt_family"]
        .nunique()
        .reset_index(name="family_count_with_presence_shift")
    )
    position_shift = (
        h[h["presence_shift"] == 1]
        .groupby("feature_id")["position_label"]
        .nunique()
        .reset_index(name="position_count_with_presence_shift")
    )
    control_shift = (
        h[h["presence_shift"] == 1]
        .groupby("feature_id")["perturbation_type"]
        .nunique()
        .reset_index(name="handled_controls_with_presence_shift")
    )
    control_large = (
        h[h["large_abs_delta"] == 1]
        .groupby("feature_id")["perturbation_type"]
        .nunique()
        .reset_index(name="handled_controls_with_large_abs_delta")
    )
    families = (
        h.groupby("feature_id")["base_prompt_family"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .reset_index(name="families_observed")
    )
    positions = (
        h.groupby("feature_id")["position_label"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .reset_index(name="positions_observed")
    )
    controls = (
        h.groupby("feature_id")["perturbation_type"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .reset_index(name="handled_controls_observed")
    )
    for aux in [family_shift, position_shift, control_shift, control_large, families, positions, controls]:
        out = out.merge(aux, on="feature_id", how="left")
    fill_zero = [
        "family_count_with_presence_shift",
        "position_count_with_presence_shift",
        "handled_controls_with_presence_shift",
        "handled_controls_with_large_abs_delta",
    ]
    out[fill_zero] = out[fill_zero].fillna(0).astype(int)
    out[["families_observed", "positions_observed", "handled_controls_observed"]] = out[
        ["families_observed", "positions_observed", "handled_controls_observed"]
    ].fillna("")
    out["consistent_distinction_flag"] = (
        (out["handled_control_count"] == 2)
        & (
            (out["handled_controls_with_presence_shift"] == 2)
            | (
                (out["handled_controls_with_large_abs_delta"] == 2)
                & (out["family_count_with_any_row"] >= 3)
                & (out["mean_abs_delta"] >= 0.1)
            )
        )
        & ((out["presence_shift_count"] >= 3) | (out["large_abs_delta_count"] >= 5))
    ).astype(int)
    out["distinction_rank"] = (
        out.sort_values(
            [
                "consistent_distinction_flag",
                "handled_controls_with_presence_shift",
                "family_count_with_presence_shift",
                "presence_shift_count",
                "mean_abs_delta",
                "max_abs_delta",
            ],
            ascending=[False, False, False, False, False, False],
        )
        .reset_index(drop=True)
        .index
        + 1
    )
    out["feature_id"] = out["feature_id"].astype(int)
    out = out.sort_values("distinction_rank")
    columns = [
        "distinction_rank",
        "feature_id",
        "consistent_distinction_flag",
        "handled_control_count",
        "handled_controls_observed",
        "handled_controls_with_presence_shift",
        "handled_controls_with_large_abs_delta",
        "family_count_with_any_row",
        "family_count_with_presence_shift",
        "families_observed",
        "position_count_with_any_row",
        "position_count_with_presence_shift",
        "positions_observed",
        "comparison_rows",
        "mean_abs_delta",
        "max_abs_delta",
        "mean_signed_delta",
        "presence_shift_count",
        "ascii_only_count",
        "perturbation_only_count",
        "both_present_count",
        "large_abs_delta_count",
    ]
    return out[columns]


def tracked_hit_counts(tracked: pd.DataFrame) -> pd.DataFrame:
    t = tracked.copy()
    t["hit"] = (t["appeared_in_topk50"].astype(str) == "1").astype(int)
    out = (
        t.groupby(["feature_id", "perturbation_type", "base_prompt_family", "position_label"], as_index=False)
        .agg(
            hit_count=("hit", "sum"),
            comparison_count=("hit", "size"),
            mean_activation=("activation", "mean"),
            max_activation=("activation", "max"),
        )
        .sort_values(["feature_id", "perturbation_type", "base_prompt_family", "position_label"])
    )
    out["feature_id"] = out["feature_id"].astype(int)
    return out


def plot_bar(df: pd.DataFrame, value: str, title: str, path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    pivot = df.pivot(index="perturbation_type", columns="layer", values=value).reindex(PERTURBATION_ORDER)
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title(title)
    ax.set_xlabel("Perturbation type")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, layer: int, value: str, title: str, path: Path, label: str) -> None:
    sub = df[df["layer"] == layer]
    pivot = sub.pivot(index="position_label", columns="perturbation_type", values=value).reindex(POSITION_ORDER)[PERTURBATION_ORDER]
    arr = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_top_recurrent_features(recurrence: pd.DataFrame, layer: int, path: Path) -> None:
    sub = recurrence[recurrence["layer"] == layer].head(20).copy()
    sub["feature"] = sub["feature_id"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sub["feature"][::-1], sub["prompt_family_count"][::-1], color="#4978b8")
    ax.set_title(f"Top Recurrent Layer {layer} Features By Prompt-Family Coverage")
    ax.set_xlabel("Distinct prompt families")
    ax.set_ylabel("Feature ID")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tracked_hits(tracked_counts: pd.DataFrame, path: Path) -> None:
    summary = tracked_counts.groupby(["feature_id", "perturbation_type"], as_index=False)["hit_count"].sum()
    pivot = summary.pivot(index="feature_id", columns="perturbation_type", values="hit_count").fillna(0)
    cols = [c for c in PERTURBATION_ORDER if c in pivot.columns]
    if "ascii_original" in pivot.columns:
        cols = ["ascii_original"] + cols
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_title("Tracked Layer-26 Feature Hit Counts")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=[str(int(x)) for x in pivot.index])
    ax.set_ylabel("Feature ID")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("TopK-50 hit count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_results_memo(
    path: Path,
    perturb_by_layer_position: pd.DataFrame,
    perturb_overall: pd.DataFrame,
    recurrence: pd.DataFrame,
    distinguishing: pd.DataFrame,
    tracked_counts: pd.DataFrame,
) -> None:
    overall_ranked = perturb_overall.sort_values("mean_abs_delta", ascending=False)
    jaccard_ranked = perturb_overall.sort_values("mean_jaccard_distance", ascending=False)
    layer26 = perturb_overall[perturb_overall["layer"] == 26].sort_values("mean_abs_delta", ascending=False)
    layer14 = perturb_overall[perturb_overall["layer"] == 14].sort_values("mean_abs_delta", ascending=False)
    top_recur_26 = recurrence[recurrence["layer"] == 26].head(10)
    top_recur_14 = recurrence[recurrence["layer"] == 14].head(10)
    consistent = distinguishing[distinguishing["consistent_distinction_flag"] == 1].head(20)
    tracked_total = tracked_counts.groupby("feature_id", as_index=False)["hit_count"].sum().sort_values("hit_count", ascending=False)

    def format_rows(df: pd.DataFrame, cols: list[str], n: int = 5) -> list[str]:
        lines = []
        for _, row in df.head(n).iterrows():
            parts = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    parts.append(f"{col}={value:.6g}")
                else:
                    parts.append(f"{col}={value}")
            lines.append("- " + "; ".join(parts))
        return lines or ["- none"]

    lines = [
        "# Results Memo: Full Controlled SAE Perturbation Matrix Aggregates",
        "",
        "This memo organizes evidence from the existing matrix outputs only. It does not assign semantic labels to features.",
        "",
        "## Scope",
        "",
        "- Source run: `/workspace/qwen-scope/5-11-26/sae_outputs/full_controlled_perturbation_matrix/`.",
        "- Layers: 26 and 14.",
        "- Metric definitions: Jaccard distance is `1 - topk_jaccard`; recurrence is distinct prompt-family coverage in TopK-50 rows; handled distinction is repeated ASCII-vs-handled activation or presence shift for `e_to_ē` and `s_to_ş` at layer 26.",
        "",
        "## Perturbation Ranking",
        "",
        "Top perturbation/layer rows by mean absolute delta:",
        *format_rows(overall_ranked, ["perturbation_type", "layer", "mean_abs_delta", "mean_jaccard_distance"], 8),
        "",
        "Top perturbation/layer rows by Jaccard distance:",
        *format_rows(jaccard_ranked, ["perturbation_type", "layer", "mean_jaccard_distance", "mean_abs_delta"], 8),
        "",
        "Layer-26 mean absolute delta ranking:",
        *format_rows(layer26, ["perturbation_type", "mean_abs_delta", "mean_jaccard_distance"], 6),
        "",
        "Layer-14 mean absolute delta ranking:",
        *format_rows(layer14, ["perturbation_type", "mean_abs_delta", "mean_jaccard_distance"], 6),
        "",
        "## Recurrent Features",
        "",
        "Top recurrent layer-26 feature IDs by prompt-family coverage:",
        *format_rows(top_recur_26, ["feature_id", "prompt_family_count", "prompt_count", "prompt_position_count", "mean_activation"], 10),
        "",
        "Top recurrent layer-14 feature IDs by prompt-family coverage:",
        *format_rows(top_recur_14, ["feature_id", "prompt_family_count", "prompt_count", "prompt_position_count", "mean_activation"], 10),
        "",
        "## Layer-26 ASCII Versus Handled Controls",
        "",
        "Layer-26 feature IDs with the strongest repeated distinction evidence for ASCII versus handled controls:",
        *format_rows(
            consistent,
            [
                "feature_id",
                "handled_controls_with_presence_shift",
                "family_count_with_presence_shift",
                "presence_shift_count",
                "mean_abs_delta",
            ],
            20,
        ),
        "",
        "Tracked candidate feature hit totals:",
        *format_rows(tracked_total, ["feature_id", "hit_count"], 10),
        "",
        "## Readout",
        "",
        "- The aggregate tables separate perturbation sensitivity by layer and boundary position; use those tables before interpreting individual feature IDs.",
        "- The handled-control distinction table is an evidence filter, not a label set.",
        "- Feature IDs in the recurrence table are ranked by how often they appear across prompt families, not by semantic meaning.",
        "- No model run, steering, Hauhau, llama.cpp, all-layer expansion, or semantic labeling was performed for this postprocessing step.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    topk, delta, jaccard, tracked, metadata = read_inputs()

    perturb_by_layer_position, perturb_overall = perturbation_rankings(delta, jaccard)
    recurrence = feature_recurrence(topk)
    distinguishing = handled_distinguishing_features(delta)
    tracked_counts = tracked_hit_counts(tracked)

    outputs: dict[str, Path] = {
        "perturbation_rank_by_layer_position": OUT_DIR / "perturbation_rank_by_layer_position.tsv",
        "perturbation_rank_overall_by_layer": OUT_DIR / "perturbation_rank_overall_by_layer.tsv",
        "feature_recurrence_by_family": OUT_DIR / "feature_recurrence_by_family.tsv",
        "layer26_ascii_vs_handled_distinguishing_features": OUT_DIR / "layer26_ascii_vs_handled_distinguishing_features.tsv",
        "tracked_layer26_hit_counts": OUT_DIR / "tracked_layer26_hit_counts.tsv",
        "results_memo": OUT_DIR / "results_memo.md",
    }
    write_tsv(perturb_by_layer_position, outputs["perturbation_rank_by_layer_position"])
    write_tsv(perturb_overall, outputs["perturbation_rank_overall_by_layer"])
    write_tsv(recurrence, outputs["feature_recurrence_by_family"])
    write_tsv(distinguishing, outputs["layer26_ascii_vs_handled_distinguishing_features"])
    write_tsv(tracked_counts, outputs["tracked_layer26_hit_counts"])

    plot_paths = {
        "plot_mean_abs_delta_by_perturbation_layer": PLOTS_DIR / "plot_mean_abs_delta_by_perturbation_layer.png",
        "plot_jaccard_distance_by_perturbation_layer": PLOTS_DIR / "plot_jaccard_distance_by_perturbation_layer.png",
        "plot_mean_abs_delta_heatmap_layer26": PLOTS_DIR / "plot_mean_abs_delta_heatmap_layer26.png",
        "plot_mean_abs_delta_heatmap_layer14": PLOTS_DIR / "plot_mean_abs_delta_heatmap_layer14.png",
        "plot_jaccard_distance_heatmap_layer26": PLOTS_DIR / "plot_jaccard_distance_heatmap_layer26.png",
        "plot_jaccard_distance_heatmap_layer14": PLOTS_DIR / "plot_jaccard_distance_heatmap_layer14.png",
        "plot_top_recurrent_features_layer26": PLOTS_DIR / "plot_top_recurrent_features_layer26.png",
        "plot_top_recurrent_features_layer14": PLOTS_DIR / "plot_top_recurrent_features_layer14.png",
        "plot_tracked_layer26_hit_counts": PLOTS_DIR / "plot_tracked_layer26_hit_counts.png",
    }
    plot_bar(
        perturb_overall,
        "mean_abs_delta",
        "Mean Absolute Delta By Perturbation And Layer",
        plot_paths["plot_mean_abs_delta_by_perturbation_layer"],
        "Mean abs delta",
    )
    plot_bar(
        perturb_overall,
        "mean_jaccard_distance",
        "Mean TopK Jaccard Distance By Perturbation And Layer",
        plot_paths["plot_jaccard_distance_by_perturbation_layer"],
        "Mean Jaccard distance",
    )
    for layer in [26, 14]:
        plot_heatmap(
            perturb_by_layer_position,
            layer,
            "mean_abs_delta",
            f"Layer {layer} Mean Absolute Delta By Position",
            plot_paths[f"plot_mean_abs_delta_heatmap_layer{layer}"],
            "Mean abs delta",
        )
        plot_heatmap(
            perturb_by_layer_position,
            layer,
            "mean_jaccard_distance",
            f"Layer {layer} Mean TopK Jaccard Distance By Position",
            plot_paths[f"plot_jaccard_distance_heatmap_layer{layer}"],
            "Mean Jaccard distance",
        )
        plot_top_recurrent_features(recurrence, layer, plot_paths[f"plot_top_recurrent_features_layer{layer}"])
    plot_tracked_hits(tracked_counts, plot_paths["plot_tracked_layer26_hit_counts"])

    write_results_memo(
        outputs["results_memo"],
        perturb_by_layer_position,
        perturb_overall,
        recurrence,
        distinguishing,
        tracked_counts,
    )

    completed_at = utc_now()
    provenance_lines = [
        f"timestamp={completed_at}",
        f"script_path={SCRIPT_PATH}",
        f"input_topk_path={TOPK_PATH}",
        f"input_delta_path={DELTA_PATH}",
        f"input_jaccard_path={JACCARD_PATH}",
        f"input_tracked_path={TRACKED_PATH}",
        f"input_metadata_path={METADATA_PATH}",
        f"output_dir={OUT_DIR}",
        "outputs_written=" + ",".join(str(path) for path in [*outputs.values(), *plot_paths.values()]),
        "metric_definitions=jaccard_distance=1-topk_jaccard; recurrence=distinct prompt-family coverage in TopK-50 rows; handled distinction=repeated layer-26 ascii-vs-handled activation or presence shifts for e_to_ē and s_to_ş",
        f"source_prompt_count={metadata.get('prompt_count')}",
        f"source_prompt_position_layer_count={metadata.get('prompt_position_layer_count')}",
        f"source_topk_rows={len(topk)}",
        f"source_delta_rows={len(delta)}",
        f"source_jaccard_rows={len(jaccard)}",
        f"source_tracked_rows={len(tracked)}",
        "confirmation=no model run, no steering, no Hauhau, no llama.cpp, no all-layer expansion, and no semantic labels were used",
    ]
    PROVENANCE_PATH.write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

    manifest = {
        "completed_at": completed_at,
        "input_paths": {
            "topk": str(TOPK_PATH),
            "delta": str(DELTA_PATH),
            "jaccard": str(JACCARD_PATH),
            "tracked": str(TRACKED_PATH),
            "metadata": str(METADATA_PATH),
        },
        "output_tables": {key: str(path) for key, path in outputs.items()},
        "output_plots": {key: str(path) for key, path in plot_paths.items()},
        "provenance_path": str(PROVENANCE_PATH),
        "no_semantic_labels": True,
        "no_model_run": True,
    }
    manifest_path = OUT_DIR / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"aggregate_output_dir={OUT_DIR}")
    print(f"aggregate_manifest={manifest_path}")
    print(f"provenance={PROVENANCE_PATH}")
    for key, path in outputs.items():
        print(f"{key}={path}")
    for key, path in plot_paths.items():
        print(f"{key}={path}")
    print("aggregate_full_controlled_perturbation_matrix_status=ok")


if __name__ == "__main__":
    main()
