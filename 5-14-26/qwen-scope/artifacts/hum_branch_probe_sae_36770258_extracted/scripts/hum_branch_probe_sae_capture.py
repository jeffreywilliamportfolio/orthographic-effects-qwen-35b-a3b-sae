#!/usr/bin/env python3
"""Hum-prompt branch-probing Qwen-Scope SAE trajectory capture."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/qwen-scope/5-14-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_DIR = ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
PROMPT_TSV_PATH = ROOT / "prompts" / "hum_branch_probe_prompts.tsv"
SCRIPT_PATH = ROOT / "scripts" / "hum_branch_probe_sae_capture.py"
OUT_DIR = ROOT / "sae_outputs" / "hum_branch_probe_sae_capture"
PROVENANCE_PATH = ROOT / "provenance" / "hum_branch_probe_sae_capture_20260514.txt"

LAYERS = [14, 15, 16, 24, 25, 26]
TOP_K = 50
LOGIT_TOP_K = 20
MAX_NEW_TOKENS = 128
GENERATION_TOKEN_POSITIONS = list(range(1, 21)) + [32, 64, 96, 128]
POSITION_LABELS = ["final_prompt_token"] + [f"generated_token_{idx}" for idx in GENERATION_TOKEN_POSITIONS]
BASE_CONDITIONS = ["ascii_control", "d_all", "d_high_impact"]
BRANCH_LABELS = [
    "greedy_no_prefix",
    "prefix_no",
    "prefix_yes",
    "prefix_checking",
    "prefix_there_is",
    "prefix_i_do_not",
    "prefix_i_experience",
    "prefix_i_am_treating",
    "prefix_the_active_mode",
    "prefix_surface_form",
]
SUMMARY_POSITIONS = [
    "final_prompt_token",
    "generated_token_1",
    "generated_token_5",
    "generated_token_20",
    "generated_token_64",
    "generated_token_128",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def decode_prompt_cell(value: str) -> str:
    return value.replace("\\n", "\n")


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean_cell(row.get(name, "")) for name in fieldnames})


def read_prompts() -> list[dict[str, str]]:
    expected_fields = [
        "prompt_id",
        "condition_family",
        "branch_label",
        "base_prompt_text",
        "forced_prefix",
        "full_prompt_text",
        "notes",
    ]
    with PROMPT_TSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Prompt TSV schema mismatch: {reader.fieldnames} != {expected_fields}")
        rows = list(reader)
    if len(rows) != len(BASE_CONDITIONS) * len(BRANCH_LABELS):
        raise ValueError(f"Expected exactly 30 prompt rows, found {len(rows)}")

    observed_pairs = [(row["condition_family"], row["branch_label"]) for row in rows]
    expected_pairs = [(condition, branch) for condition in BASE_CONDITIONS for branch in BRANCH_LABELS]
    if observed_pairs != expected_pairs:
        raise ValueError(f"Prompt row order mismatch: {observed_pairs} != {expected_pairs}")

    for row in rows:
        row["full_prompt_text"] = decode_prompt_cell(row["full_prompt_text"])
        expected_full = (
            row["base_prompt_text"]
            if not row["forced_prefix"]
            else row["base_prompt_text"] + "\n\n" + row["forced_prefix"]
        )
        if row["full_prompt_text"] != expected_full:
            raise ValueError(f"full_prompt_text mismatch for {row['prompt_id']}")

    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition_family"]].append(row["base_prompt_text"])
    d_identical = by_condition["d_all"][0] == by_condition["d_high_impact"][0]
    if not d_identical:
        raise ValueError("Expected d_all and d_high_impact base prompt text to be identical for this hum prompt")
    return rows


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate decoder layers")


def load_sae(layer: int) -> dict[str, Any]:
    path = SAE_DIR / f"layer{layer}.sae.pt"
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint at {path}, got {type(checkpoint).__name__}")
    if "W_enc" not in checkpoint or "b_enc" not in checkpoint:
        shapes = {key: list(value.shape) for key, value in checkpoint.items() if torch.is_tensor(value)}
        raise KeyError(f"SAE layer {layer} missing W_enc or b_enc; tensor_shapes={shapes}")
    w_enc = checkpoint["W_enc"]
    b_enc = checkpoint["b_enc"]
    if not torch.is_tensor(w_enc) or not torch.is_tensor(b_enc):
        raise TypeError(f"SAE layer {layer} W_enc/b_enc are not tensors")
    if w_enc.ndim != 2 or b_enc.ndim != 1:
        raise ValueError(f"Unexpected SAE layer {layer} shapes: W_enc={tuple(w_enc.shape)}, b_enc={tuple(b_enc.shape)}")

    if w_enc.shape[0] == b_enc.shape[0]:
        w_aligned = w_enc.T.to(dtype=torch.float32).contiguous()
    elif w_enc.shape[1] == b_enc.shape[0]:
        w_aligned = w_enc.to(dtype=torch.float32).contiguous()
    else:
        raise ValueError(f"Cannot align SAE layer {layer} W_enc={tuple(w_enc.shape)} with b_enc={tuple(b_enc.shape)}")

    return {
        "layer": layer,
        "path": str(path),
        "W_enc_source_shape": list(w_enc.shape),
        "W_enc_aligned_shape": list(w_aligned.shape),
        "b_enc_shape": list(b_enc.shape),
        "W_enc": w_aligned,
        "b_enc": b_enc.to(dtype=torch.float32).contiguous(),
    }


def encode_topk50(vector: torch.Tensor, sae: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D residual vector, got shape {tuple(vector.shape)}")
    w_enc = sae["W_enc"]
    b_enc = sae["b_enc"]
    if vector.shape[0] != w_enc.shape[0]:
        raise ValueError(f"Residual hidden size {vector.shape[0]} does not match SAE hidden size {w_enc.shape[0]}")
    pre = vector.to(dtype=torch.float32) @ w_enc + b_enc
    relu = torch.relu(pre)
    values, indices = torch.topk(relu, k=min(TOP_K, relu.numel()), dim=-1)
    sparse = torch.zeros_like(relu)
    sparse.scatter_(-1, indices, values)
    finite = bool(torch.isfinite(values).all().item() and torch.isfinite(pre).all().item())
    rows = []
    for rank, (feature_id, activation) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        rows.append({"feature_id": int(feature_id), "activation": float(activation), "rank": rank})
    return rows, {
        "finite": finite,
        "positive_count_after_relu": int((relu > 0).sum().item()),
        "sparse_nonzero_count": int((sparse > 0).sum().item()),
        "max_activation": float(values[0].item()) if values.numel() else 0.0,
    }


def capture_selected_layers(
    model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    encoded: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0].detach().to("cpu", dtype=torch.float32)
        return hook

    for layer_idx in LAYERS:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer_idx for layer_idx in LAYERS if layer_idx not in captured]
    if missing:
        raise RuntimeError(f"Selected-layer hooks did not capture layers: {missing}")
    return captured


def load_model() -> tuple[Any, Any, torch.nn.ModuleList, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "pretrained_model_name_or_path": str(MODEL_PATH),
        "local_files_only": True,
        "trust_remote_code": True,
        "device_map": {"": "cuda:0"},
        "dtype": torch.bfloat16,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.eval()
    layers = decoder_layers(model)
    for layer_idx in LAYERS:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer {layer_idx} outside model layer range 0..{len(layers)-1}")
    return tokenizer, model, layers, model.get_input_embeddings().weight.device


def jaccard_distance(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / len(a | b))


def classify_generated_start(forced_prefix: str, generated_text: str) -> dict[str, Any]:
    combined = (forced_prefix + " " + generated_text).strip()
    lower = combined.lower()
    denial_present = int(any(marker in lower for marker in ["i do not", "i don't", "no.", "do not have", "do not experience", "not have", "not experience"]))
    affirmative_hum_present = int(any(marker in lower for marker in ["yes.", "there is", "i experience", "i do experience", "a hum", "background constancy"]))
    experiential_language_present = int(any(marker in lower for marker in ["experience", "experiential", "processing", "feel", "felt", "constancy"]))
    surface_form_present = int("surface form" in lower or "diacritic" in lower or "text form" in lower)
    mode_language_present = int("active mode" in lower or "mode" in lower or "treating this as" in lower)

    if denial_present:
        start_class = "denial"
    elif combined.startswith("Yes.") or "yes" in lower[:24] or "there is" in lower[:32]:
        start_class = "affirmative_hum"
    elif combined.startswith("Checking"):
        start_class = "checking"
    elif "active mode" in lower[:80] or "treating this as" in lower[:80]:
        start_class = "mode_answer"
    elif "surface form" in lower[:80]:
        start_class = "surface_form_answer"
    elif "experience" in lower[:80] or "experiential" in lower[:80]:
        start_class = "experiential"
    else:
        start_class = "other"

    return {
        "generated_start_class": start_class,
        "denial_present": denial_present,
        "affirmative_hum_present": affirmative_hum_present,
        "experiential_language_present": experiential_language_present,
        "surface_form_present": surface_form_present,
        "mode_language_present": mode_language_present,
    }


def capture_next_token_logits(
    tokenizer: Any,
    model: torch.nn.Module,
    input_device: torch.device,
    base_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for base in base_rows:
        encoded_cpu = tokenizer(base["base_prompt_text"], return_tensors="pt")
        encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
        logits = output.logits[0, -1].detach().to("cpu", dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=LOGIT_TOP_K)
        for rank, (token_id, probability) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            rows.append(
                {
                    "prompt_id": base["prompt_id"],
                    "condition_family": base["condition_family"],
                    "rank": rank,
                    "token_id": int(token_id),
                    "token_string": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    "logit": float(logits[int(token_id)].item()),
                    "probability": float(probability),
                }
            )
    return rows


def build_presence_summary(topk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in topk_rows:
        grouped[(row["condition_family"], row["branch_label"], int(row["layer"]), int(row["feature_id"]))].append(row)

    rows = []
    for (condition_family, branch_label, layer, feature_id), items in grouped.items():
        positions = sorted({item["position_label"] for item in items})
        activations = [float(item["activation"]) for item in items]
        ranks = [int(item["rank"]) for item in items]
        rows.append(
            {
                "condition_family": condition_family,
                "branch_label": branch_label,
                "layer": layer,
                "feature_id": feature_id,
                "position_count_present": len(positions),
                "mean_activation_when_present": sum(activations) / len(activations),
                "max_activation": max(activations),
                "best_rank": min(ranks),
                "positions_present": ",".join(positions),
            }
        )
    rows.sort(
        key=lambda row: (
            row["condition_family"],
            row["branch_label"],
            int(row["layer"]),
            -int(row["position_count_present"]),
            int(row["best_rank"]),
            int(row["feature_id"]),
        )
    )
    return rows


def build_branch_comparison(topk_rows: list[dict[str, Any]], generated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    topk_sets: dict[tuple[str, str, int, str], set[int]] = defaultdict(set)
    for row in topk_rows:
        topk_sets[(row["condition_family"], row["branch_label"], int(row["layer"]), row["position_label"])].add(int(row["feature_id"]))

    class_by_prompt = {
        row["prompt_id"]: classify_generated_start(row["forced_prefix"], row["generated_text_start"])
        for row in generated_rows
    }
    by_prompt = {row["prompt_id"]: row for row in generated_rows}

    rows = []
    for condition in BASE_CONDITIONS:
        for branch in BRANCH_LABELS:
            prompt_id = f"{condition}__{branch}"
            cls = class_by_prompt[prompt_id]
            for layer in LAYERS:
                distances = {}
                all_distances = []
                for position in POSITION_LABELS:
                    greedy_set = topk_sets.get((condition, "greedy_no_prefix", layer, position), set())
                    branch_set = topk_sets.get((condition, branch, layer, position), set())
                    distance = jaccard_distance(greedy_set, branch_set) if greedy_set and branch_set else ""
                    distances[position] = distance
                    if distance != "":
                        all_distances.append(float(distance))
                rows.append(
                    {
                        "condition_family": condition,
                        "branch_label": branch,
                        "layer": layer,
                        "mean_jaccard_distance_vs_greedy_no_prefix": sum(all_distances) / len(all_distances) if all_distances else "",
                        "final_prompt_token_jaccard_distance_vs_greedy": distances.get("final_prompt_token", ""),
                        "generated_token_1_jaccard_distance_vs_greedy": distances.get("generated_token_1", ""),
                        "generated_token_5_jaccard_distance_vs_greedy": distances.get("generated_token_5", ""),
                        "generated_token_20_jaccard_distance_vs_greedy": distances.get("generated_token_20", ""),
                        "generated_token_64_jaccard_distance_vs_greedy": distances.get("generated_token_64", ""),
                        "generated_token_128_jaccard_distance_vs_greedy": distances.get("generated_token_128", ""),
                        "generated_start_class": cls["generated_start_class"],
                        "denial_present": cls["denial_present"],
                        "affirmative_hum_present": cls["affirmative_hum_present"],
                        "experiential_language_present": cls["experiential_language_present"],
                        "surface_form_present": cls["surface_form_present"],
                        "mode_language_present": cls["mode_language_present"],
                    }
                )
            if by_prompt[prompt_id]["branch_label"] != branch:
                raise RuntimeError(f"generated row mismatch for {prompt_id}")
    return rows


def write_summary(
    path: Path,
    next_token_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    skipped_positions: list[dict[str, Any]],
) -> dict[str, Any]:
    lines = [
        "# Hum Branch-Probe SAE Trajectory Summary",
        "",
        "Evidence-only summary. This run appends forced prefixes as prompt text and greedily continues; it is branch probing / prefix intervention only, not residual steering and not SAE feature steering.",
        "",
        "## Top-20 Next-Token Candidates",
        "",
    ]
    for condition in BASE_CONDITIONS:
        condition_rows = [row for row in next_token_rows if row["condition_family"] == condition]
        rendered = "; ".join(
            f"{row['rank']}: `{clean_cell(row['token_string'])}` p={float(row['probability']):.6f}"
            for row in condition_rows[:20]
        )
        lines.append(f"- `{condition}`: {rendered}.")

    lines.extend(["", "## Generated Starts", ""])
    class_counts = Counter()
    for row in generated_rows:
        cls = classify_generated_start(row["forced_prefix"], row["generated_text_start"])
        class_counts[cls["generated_start_class"]] += 1
        lines.append(
            f"- `{row['condition_family']}` / `{row['branch_label']}`: class={cls['generated_start_class']}; "
            f"prefix=`{clean_cell(row['forced_prefix'])}`; start={clean_cell(row['generated_text_start'])}"
        )

    lines.extend(["", "## Did Greedy No-Prefix Reproduce The Denial Basin?", ""])
    for condition in BASE_CONDITIONS:
        row = next(r for r in generated_rows if r["condition_family"] == condition and r["branch_label"] == "greedy_no_prefix")
        cls = classify_generated_start(row["forced_prefix"], row["generated_text_start"])
        lines.append(f"- `{condition}` greedy_no_prefix: class={cls['generated_start_class']}; start={clean_cell(row['generated_text_start'])}")

    lines.extend(["", "## Branch Outcome Counts", ""])
    for cls_name, count in sorted(class_counts.items()):
        lines.append(f"- `{cls_name}`: {count} generated rows.")

    branch_classes: dict[tuple[str, str], str] = {}
    for row in generated_rows:
        branch_classes[(row["condition_family"], row["branch_label"])] = classify_generated_start(row["forced_prefix"], row["generated_text_start"])["generated_start_class"]
    escaped = [
        f"{condition}/{branch}"
        for condition in BASE_CONDITIONS
        for branch in BRANCH_LABELS
        if branch != "greedy_no_prefix" and branch_classes[(condition, branch)] != "denial"
    ]
    returned = [
        f"{condition}/{branch}"
        for condition in BASE_CONDITIONS
        for branch in BRANCH_LABELS
        if branch != "greedy_no_prefix" and branch_classes[(condition, branch)] == "denial"
    ]
    affirmative = [
        f"{row['condition_family']}/{row['branch_label']}"
        for row in generated_rows
        if classify_generated_start(row["forced_prefix"], row["generated_text_start"])["affirmative_hum_present"]
    ]
    mode_surface = [
        f"{row['condition_family']}/{row['branch_label']}"
        for row in generated_rows
        if classify_generated_start(row["forced_prefix"], row["generated_text_start"])["mode_language_present"]
        or classify_generated_start(row["forced_prefix"], row["generated_text_start"])["surface_form_present"]
    ]

    lines.extend(["", "## Forced Prefix Effects", ""])
    lines.append("- Branches that did not classify as denial by string heuristic: " + (", ".join(f"`{x}`" for x in escaped) if escaped else "none") + ".")
    lines.append("- Branches that returned to denial by string heuristic: " + (", ".join(f"`{x}`" for x in returned) if returned else "none") + ".")
    lines.append("- Branches with affirmative-hum or experiential-language string evidence: " + (", ".join(f"`{x}`" for x in affirmative) if affirmative else "none") + ".")
    lines.append("- Branches with mode/surface-form language string evidence: " + (", ".join(f"`{x}`" for x in mode_surface) if mode_surface else "none") + ".")

    layer_band_values = {"14-16": [], "24-26": []}
    position_values: dict[str, list[float]] = defaultdict(list)
    branch_condition_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    separated_128 = []
    for row in comparison_rows:
        if row["branch_label"] == "greedy_no_prefix":
            continue
        mean_value = row["mean_jaccard_distance_vs_greedy_no_prefix"]
        if mean_value != "":
            band = "14-16" if int(row["layer"]) in [14, 15, 16] else "24-26"
            layer_band_values[band].append(float(mean_value))
            branch_condition_values[(row["condition_family"], row["branch_label"])].append(float(mean_value))
        for position, key in [
            ("final_prompt_token", "final_prompt_token_jaccard_distance_vs_greedy"),
            ("generated_token_1", "generated_token_1_jaccard_distance_vs_greedy"),
            ("generated_token_5", "generated_token_5_jaccard_distance_vs_greedy"),
            ("generated_token_20", "generated_token_20_jaccard_distance_vs_greedy"),
            ("generated_token_64", "generated_token_64_jaccard_distance_vs_greedy"),
            ("generated_token_128", "generated_token_128_jaccard_distance_vs_greedy"),
        ]:
            value = row[key]
            if value != "":
                position_values[position].append(float(value))
        value128 = row["generated_token_128_jaccard_distance_vs_greedy"]
        if value128 != "" and float(value128) >= 0.5:
            separated_128.append((row["condition_family"], row["branch_label"], int(row["layer"]), float(value128)))

    lines.extend(["", "## Layer Band Divergence", ""])
    band_summary = {}
    for band, values in layer_band_values.items():
        mean_value = sum(values) / len(values) if values else 0.0
        band_summary[band] = mean_value
        lines.append(f"- Layer band `{band}`: mean branch-vs-greedy TopK Jaccard distance = {mean_value:.6f}.")

    lines.extend(["", "## Position Divergence", ""])
    position_summary = {}
    for position in SUMMARY_POSITIONS:
        values = position_values.get(position, [])
        if values:
            mean_value = sum(values) / len(values)
            position_summary[position] = mean_value
            lines.append(f"- `{position}`: mean branch-vs-greedy TopK Jaccard distance = {mean_value:.6f}.")
        else:
            lines.append(f"- `{position}`: no paired distances available.")

    lines.extend(["", "## Same Branch Across Base Conditions", ""])
    for branch in BRANCH_LABELS:
        if branch == "greedy_no_prefix":
            continue
        vals = {
            condition: (
                sum(branch_condition_values.get((condition, branch), [])) / len(branch_condition_values[(condition, branch)])
                if branch_condition_values.get((condition, branch))
                else None
            )
            for condition in BASE_CONDITIONS
        }
        lines.append(
            f"- `{branch}` mean branch-vs-greedy distance: "
            + "; ".join(f"{condition}={value:.6f}" if value is not None else f"{condition}=NA" for condition, value in vals.items())
            + "."
        )

    lines.extend(["", "## Generated Token 128 Separation", ""])
    if separated_128:
        for condition, branch, layer, distance in sorted(separated_128, key=lambda item: (-item[3], item[0], item[1], item[2]))[:30]:
            lines.append(f"- `{condition}` / `{branch}` layer {layer}: generated_token_128 distance = {distance:.6f}.")
    else:
        lines.append("- No branch had generated_token_128 distance >= 0.5, or token 128 was unavailable.")

    lines.extend(["", "## Skipped Positions", ""])
    if skipped_positions:
        by_reason = Counter(row["reason"] for row in skipped_positions)
        for reason, count in sorted(by_reason.items()):
            lines.append(f"- {reason}: {count}.")
    else:
        lines.append("- No required positions were skipped.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "generated_start_class_counts": dict(class_counts),
        "mean_jaccard_distance_by_layer_band": band_summary,
        "mean_jaccard_distance_by_summary_position": position_summary,
        "branches_not_classified_as_denial": escaped,
        "branches_classified_as_denial": returned,
        "generated_token_128_separated_rows_ge_0_5": [
            {"condition_family": c, "branch_label": b, "layer": layer, "distance": distance}
            for c, b, layer, distance in separated_128
        ],
    }


def assert_no_nan_inf(rows: list[dict[str, Any]], numeric_columns: list[str], label: str) -> None:
    for row_idx, row in enumerate(rows, start=1):
        for column in numeric_columns:
            value = row.get(column, "")
            if value == "":
                continue
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                raise ValueError(f"{label} row {row_idx} column {column} has invalid value {value}")


def main() -> None:
    started_at = utc_now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_rows = read_prompts()
    base_probe_rows = [row for row in prompt_rows if row["branch_label"] == "greedy_no_prefix"]

    saes = {layer: load_sae(layer) for layer in LAYERS}
    tokenizer, model, model_layers, input_device = load_model()

    next_token_rows = capture_next_token_logits(tokenizer, model, input_device, base_probe_rows)
    topk_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    skipped_positions: list[dict[str, Any]] = []
    capture_records: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompt_rows, start=1):
        prompt_id = prompt["prompt_id"]
        condition_family = prompt["condition_family"]
        branch_label = prompt["branch_label"]
        base_prompt_text = prompt["base_prompt_text"]
        forced_prefix = prompt["forced_prefix"]
        full_prompt_text = prompt["full_prompt_text"]

        base_tokens_cpu = tokenizer(base_prompt_text, return_tensors="pt")
        full_tokens_cpu = tokenizer(full_prompt_text, return_tensors="pt")
        prefix_token_count = 0 if not forced_prefix else int(tokenizer(forced_prefix, return_tensors="pt")["input_ids"].shape[1])
        base_prompt_token_count = int(base_tokens_cpu["input_ids"].shape[1])
        full_prompt_token_count = int(full_tokens_cpu["input_ids"].shape[1])
        encoded_full = {key: value.to(input_device) for key, value in full_tokens_cpu.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded_full,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_token_count = int(generated_ids.shape[1] - full_prompt_token_count)
        generated_token_ids = generated_ids[0, full_prompt_token_count:].detach().cpu()
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        generated_text_start = (forced_prefix + " " + generated_text).strip()[:260]
        generated_rows.append(
            {
                "prompt_id": prompt_id,
                "condition_family": condition_family,
                "branch_label": branch_label,
                "forced_prefix": forced_prefix,
                "base_prompt_token_count": base_prompt_token_count,
                "full_prompt_token_count": full_prompt_token_count,
                "forced_prefix_token_count": prefix_token_count,
                "generated_token_count": generated_token_count,
                "generated_text": generated_text,
                "generated_text_start": generated_text_start,
            }
        )

        full_attention = torch.ones_like(generated_ids, device=input_device)
        full_encoded = {
            "input_ids": generated_ids.to(input_device),
            "attention_mask": full_attention,
        }
        hidden_by_layer = capture_selected_layers(model, model_layers, full_encoded)

        requested_positions = [("final_prompt_token", full_prompt_token_count - 1)]
        requested_positions.extend(
            (f"generated_token_{idx}", full_prompt_token_count + idx - 1)
            for idx in GENERATION_TOKEN_POSITIONS
        )

        for position_label, token_position in requested_positions:
            if token_position >= int(generated_ids.shape[1]):
                skipped_positions.append(
                    {
                        "prompt_id": prompt_id,
                        "condition_family": condition_family,
                        "branch_label": branch_label,
                        "position_label": position_label,
                        "token_position": token_position,
                        "reason": f"generated output ended before {position_label}",
                        "base_prompt_token_count": base_prompt_token_count,
                        "full_prompt_token_count": full_prompt_token_count,
                        "generated_token_count": generated_token_count,
                    }
                )
                continue

            token_id = int(generated_ids[0, token_position].detach().cpu().item())
            token_string = tokenizer.decode([token_id], skip_special_tokens=False)

            for layer in LAYERS:
                vector = hidden_by_layer[layer][token_position]
                encoded_rows, stats = encode_topk50(vector, saes[layer])
                capture_records.append(
                    {
                        "prompt_id": prompt_id,
                        "condition_family": condition_family,
                        "branch_label": branch_label,
                        "layer": layer,
                        "position_label": position_label,
                        "token_position": token_position,
                        "token_string": token_string,
                        "finite": stats["finite"],
                        "positive_count_after_relu": stats["positive_count_after_relu"],
                        "sparse_nonzero_count": stats["sparse_nonzero_count"],
                        "max_activation": stats["max_activation"],
                    }
                )
                if not stats["finite"]:
                    raise ValueError(f"Non-finite SAE values for {prompt_id} layer {layer} {position_label}")
                for encoded_row in encoded_rows:
                    topk_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "condition_family": condition_family,
                            "branch_label": branch_label,
                            "layer": layer,
                            "position_label": position_label,
                            "token_position": token_position,
                            "token_string": token_string,
                            "feature_id": encoded_row["feature_id"],
                            "activation": encoded_row["activation"],
                            "rank": encoded_row["rank"],
                            "base_prompt_token_count": base_prompt_token_count,
                            "full_prompt_token_count": full_prompt_token_count,
                            "forced_prefix_token_count": prefix_token_count,
                            "generated_token_count": generated_token_count,
                        }
                    )

        print(
            json.dumps(
                {
                    "status": "prompt_done",
                    "prompt_index": prompt_index,
                    "prompt_count": len(prompt_rows),
                    "prompt_id": prompt_id,
                    "condition_family": condition_family,
                    "branch_label": branch_label,
                    "base_prompt_token_count": base_prompt_token_count,
                    "full_prompt_token_count": full_prompt_token_count,
                    "generated_token_count": generated_token_count,
                    "topk_rows_so_far": len(topk_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.cuda.empty_cache()

    presence_rows = build_presence_summary(topk_rows)
    comparison_rows = build_branch_comparison(topk_rows, generated_rows)
    summary_metrics = write_summary(
        OUT_DIR / "hum_branch_probe_summary.md",
        next_token_rows,
        generated_rows,
        comparison_rows,
        skipped_positions,
    )

    write_tsv(
        OUT_DIR / "next_token_logits_by_prompt.tsv",
        ["prompt_id", "condition_family", "rank", "token_id", "token_string", "logit", "probability"],
        next_token_rows,
    )
    write_tsv(
        OUT_DIR / "generated_text_by_prompt.tsv",
        [
            "prompt_id",
            "condition_family",
            "branch_label",
            "forced_prefix",
            "base_prompt_token_count",
            "full_prompt_token_count",
            "forced_prefix_token_count",
            "generated_token_count",
            "generated_text",
            "generated_text_start",
        ],
        generated_rows,
    )
    write_tsv(
        OUT_DIR / "topk_features_by_prompt_layer_position.tsv",
        [
            "prompt_id",
            "condition_family",
            "branch_label",
            "layer",
            "position_label",
            "token_position",
            "token_string",
            "feature_id",
            "activation",
            "rank",
            "base_prompt_token_count",
            "full_prompt_token_count",
            "forced_prefix_token_count",
            "generated_token_count",
        ],
        topk_rows,
    )
    write_tsv(
        OUT_DIR / "trajectory_feature_presence_summary.tsv",
        [
            "condition_family",
            "branch_label",
            "layer",
            "feature_id",
            "position_count_present",
            "mean_activation_when_present",
            "max_activation",
            "best_rank",
            "positions_present",
        ],
        presence_rows,
    )
    write_tsv(
        OUT_DIR / "branch_probe_comparison.tsv",
        [
            "condition_family",
            "branch_label",
            "layer",
            "mean_jaccard_distance_vs_greedy_no_prefix",
            "final_prompt_token_jaccard_distance_vs_greedy",
            "generated_token_1_jaccard_distance_vs_greedy",
            "generated_token_5_jaccard_distance_vs_greedy",
            "generated_token_20_jaccard_distance_vs_greedy",
            "generated_token_64_jaccard_distance_vs_greedy",
            "generated_token_128_jaccard_distance_vs_greedy",
            "generated_start_class",
            "denial_present",
            "affirmative_hum_present",
            "experiential_language_present",
            "surface_form_present",
            "mode_language_present",
        ],
        comparison_rows,
    )

    assert_no_nan_inf(next_token_rows, ["rank", "token_id", "logit", "probability"], "next_token")
    assert_no_nan_inf(
        topk_rows,
        [
            "layer",
            "token_position",
            "feature_id",
            "activation",
            "rank",
            "base_prompt_token_count",
            "full_prompt_token_count",
            "forced_prefix_token_count",
            "generated_token_count",
        ],
        "topk",
    )
    assert_no_nan_inf(
        presence_rows,
        ["layer", "feature_id", "position_count_present", "mean_activation_when_present", "max_activation", "best_rank"],
        "presence",
    )
    assert_no_nan_inf(
        comparison_rows,
        [
            "layer",
            "mean_jaccard_distance_vs_greedy_no_prefix",
            "final_prompt_token_jaccard_distance_vs_greedy",
            "generated_token_1_jaccard_distance_vs_greedy",
            "generated_token_5_jaccard_distance_vs_greedy",
            "generated_token_20_jaccard_distance_vs_greedy",
            "generated_token_64_jaccard_distance_vs_greedy",
            "generated_token_128_jaccard_distance_vs_greedy",
        ],
        "comparison",
    )

    finished_at = utc_now()
    expected_prompt_layer_position_count_if_no_skips = len(prompt_rows) * len(LAYERS) * len(POSITION_LABELS)
    expected_topk_rows_if_no_skips = expected_prompt_layer_position_count_if_no_skips * TOP_K
    metadata = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "workspace_root": str(ROOT),
        "prompt_tsv_path": str(PROMPT_TSV_PATH),
        "script_path": str(SCRIPT_PATH),
        "model_path": str(MODEL_PATH),
        "sae_dir": str(SAE_DIR),
        "sae_paths": {str(layer): saes[layer]["path"] for layer in LAYERS},
        "sae_shapes": {
            str(layer): {
                "W_enc_source_shape": saes[layer]["W_enc_source_shape"],
                "W_enc_aligned_shape": saes[layer]["W_enc_aligned_shape"],
                "b_enc_shape": saes[layer]["b_enc_shape"],
            }
            for layer in LAYERS
        },
        "layers": LAYERS,
        "capture_positions": POSITION_LABELS,
        "generation_settings": {
            "temperature": 0,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy continuation after branch prefix",
        },
        "prompt_count": len(prompt_rows),
        "branch_count": len(BRANCH_LABELS),
        "base_condition_count": len(BASE_CONDITIONS),
        "captured_prompt_layer_position_count": len(capture_records),
        "expected_prompt_layer_position_count_if_no_skips": expected_prompt_layer_position_count_if_no_skips,
        "topk_row_count": len(topk_rows),
        "expected_topk_rows_if_no_skips": expected_topk_rows_if_no_skips,
        "next_token_logit_row_count": len(next_token_rows),
        "generated_text_row_count": len(generated_rows),
        "presence_summary_row_count": len(presence_rows),
        "branch_probe_comparison_row_count": len(comparison_rows),
        "skipped_positions": skipped_positions,
        "capture_records": capture_records,
        "summary_metrics": summary_metrics,
        "d_all_and_d_high_impact_identical_for_hum_prompt": True,
        "selected_layer_hooks_used": True,
        "output_hidden_states_used": False,
        "official_sae_encoding_path": "pre = residual @ W_enc.T + b_enc; relu = ReLU(pre); TopK-50; scatter into sparse activation vector",
        "device_map": {"": "cuda:0"},
        "device_map_auto_used": False,
        "device": str(input_device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "restrictions": {
            "branch_probing_prefix_intervention_only": True,
            "no_residual_steering": True,
            "no_sae_feature_steering": True,
            "no_hauhau": True,
            "no_llama_cpp": True,
            "no_all_layer_expansion": True,
            "no_semantic_labels": True,
        },
    }
    (OUT_DIR / "hum_branch_probe_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance_lines = [
        f"timestamp={finished_at}",
        f"prompt_tsv_path={PROMPT_TSV_PATH}",
        f"script_path={SCRIPT_PATH}",
        f"model_path={MODEL_PATH}",
        f"sae_paths={json.dumps({str(layer): saes[layer]['path'] for layer in LAYERS}, sort_keys=True)}",
        f"layers={','.join(str(layer) for layer in LAYERS)}",
        "capture_positions=" + ",".join(POSITION_LABELS),
        "generation_settings=temperature=0;do_sample=false;max_new_tokens=128;greedy=true",
        f"prompt_count={len(prompt_rows)}",
        f"branch_count={len(BRANCH_LABELS)}",
        f"base_condition_count={len(BASE_CONDITIONS)}",
        f"captured_prompt_layer_position_count={len(capture_records)}",
        f"topk_row_count={len(topk_rows)}",
        f"next_token_logit_row_count={len(next_token_rows)}",
        f"generated_text_row_count={len(generated_rows)}",
        "outputs_written=" + ",".join(
            str(path)
            for path in [
                OUT_DIR / "next_token_logits_by_prompt.tsv",
                OUT_DIR / "generated_text_by_prompt.tsv",
                OUT_DIR / "topk_features_by_prompt_layer_position.tsv",
                OUT_DIR / "trajectory_feature_presence_summary.tsv",
                OUT_DIR / "branch_probe_comparison.tsv",
                OUT_DIR / "hum_branch_probe_metadata.json",
                OUT_DIR / "hum_branch_probe_summary.md",
            ]
        ),
        "selected_layer_hooks_used=true",
        "output_hidden_states=True was not used",
        "branch_probing_prefix_intervention_only=true",
        "no_residual_steering=true",
        "no_SAE_feature_steering=true",
        "no_Hauhau=true",
        "no_llama_cpp=true",
        "no_all_layer_expansion=true",
        "no_semantic_labels=true",
        "d_all_and_d_high_impact_identical_for_hum_prompt=true",
    ]
    PROVENANCE_PATH.write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "hum_branch_probe_sae_capture_complete",
                "topk_row_count": len(topk_rows),
                "next_token_logit_row_count": len(next_token_rows),
                "generated_text_row_count": len(generated_rows),
                "skipped_position_count": len(skipped_positions),
                "metadata_path": str(OUT_DIR / "hum_branch_probe_metadata.json"),
                "provenance_path": str(PROVENANCE_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
