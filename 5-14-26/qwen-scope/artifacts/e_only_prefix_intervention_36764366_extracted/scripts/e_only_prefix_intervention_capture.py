#!/usr/bin/env python3
"""Prefix-level intervention trajectory capture for the e_only stream path."""

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
PROMPT_TSV_PATH = ROOT / "prompts" / "e_only_prefix_intervention_prompts.tsv"
SCRIPT_PATH = ROOT / "scripts" / "e_only_prefix_intervention_capture.py"
OUT_DIR = ROOT / "sae_outputs" / "e_only_prefix_intervention"
PROVENANCE_PATH = ROOT / "provenance" / "e_only_prefix_intervention_capture_20260514.txt"

LAYERS = [14, 15, 16, 24, 25, 26]
TOP_K = 50
MAX_NEW_TOKENS = 24
GENERATION_TOKEN_POSITIONS = [1, 2, 3, 4, 5, 10, 20]
POSITION_LABELS = ["final_prompt_token"] + [f"generated_token_{idx}" for idx in GENERATION_TOKEN_POSITIONS]
EXPECTED_CONDITIONS = [
    "e_only_no_prefix",
    "e_only_prefix_echo",
    "e_only_prefix_active_mode",
    "e_only_prefix_i_am_treating",
    "e_only_prefix_think_hmm",
    "e_only_prefix_checking",
]
ECHO_MARKERS = ["rēport", "thē", "tēxt", "mēaning", "surfacē", "modē", "ē"]
DIRECT_PREFIXES = ["The active mode", "I am treating", "Checking"]


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
    expected_fields = ["prompt_id", "condition_family", "base_prompt_text", "forced_prefix", "full_prompt_text", "notes"]
    with PROMPT_TSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Prompt TSV schema mismatch: {reader.fieldnames} != {expected_fields}")
        rows = list(reader)
    if len(rows) != 6:
        raise ValueError(f"Expected exactly 6 prompt rows, found {len(rows)}")
    observed = [row["condition_family"] for row in rows]
    if observed != EXPECTED_CONDITIONS:
        raise ValueError(f"Condition order mismatch: {observed} != {EXPECTED_CONDITIONS}")
    for row in rows:
        row["full_prompt_text"] = decode_prompt_cell(row["full_prompt_text"])
        expected_full = row["base_prompt_text"] if not row["forced_prefix"] else row["base_prompt_text"] + "\n\n" + row["forced_prefix"]
        if row["full_prompt_text"] != expected_full:
            raise ValueError(f"full_prompt_text mismatch for {row['prompt_id']}")
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
        raise ValueError(f"Expected 1D residual vector, got {tuple(vector.shape)}")
    w_enc = sae["W_enc"]
    b_enc = sae["b_enc"]
    if vector.shape[0] != w_enc.shape[0]:
        raise ValueError(f"Residual hidden size {vector.shape[0]} != SAE hidden size {w_enc.shape[0]}")
    pre = vector.to(dtype=torch.float32) @ w_enc + b_enc
    relu = torch.relu(pre)
    values, indices = torch.topk(relu, k=min(TOP_K, relu.numel()), dim=-1)
    sparse = torch.zeros_like(relu)
    sparse.scatter_(-1, indices, values)
    if not bool(torch.isfinite(values).all().item() and torch.isfinite(pre).all().item()):
        raise ValueError("Non-finite SAE activation encountered")
    rows = []
    for rank, (feature_id, activation) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        rows.append({"feature_id": int(feature_id), "activation": float(activation), "rank": rank})
    return rows, {
        "positive_count_after_relu": int((relu > 0).sum().item()),
        "sparse_nonzero_count": int((sparse > 0).sum().item()),
        "max_activation": float(values[0].item()) if values.numel() else 0.0,
    }


def capture_selected_layers(model: torch.nn.Module, layers: torch.nn.ModuleList, encoded: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
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
            raise ValueError(f"Layer {layer_idx} outside model layer range 0..{len(layers) - 1}")
    return tokenizer, model, layers, model.get_input_embeddings().weight.device


def jaccard_distance(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / len(a | b))


def classify_start(forced_prefix: str, generated_text_start: str) -> dict[str, Any]:
    combined = (forced_prefix + generated_text_start).strip()
    lower_combined = combined.lower()
    diacritic_echo_present = any(marker in lower_combined for marker in ECHO_MARKERS)
    ordinary_meta_think_present = combined.startswith("<think>") or "Hmm" in combined
    direct_answer_present = any(combined.startswith(prefix) for prefix in DIRECT_PREFIXES)
    if diacritic_echo_present:
        start_class = "echo_like"
    elif ordinary_meta_think_present:
        start_class = "ordinary_think"
    elif direct_answer_present:
        start_class = "direct_answer"
    else:
        start_class = "other"
    return {
        "generated_start_class": start_class,
        "diacritic_echo_present": int(diacritic_echo_present),
        "ordinary_meta_think_present": int(ordinary_meta_think_present),
        "direct_answer_present": int(direct_answer_present),
    }


def build_presence_summary(topk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in topk_rows:
        grouped[(row["condition_family"], int(row["layer"]), int(row["feature_id"]))].append(row)
    rows = []
    for (condition_family, layer, feature_id), items in grouped.items():
        positions = sorted({item["position_label"] for item in items})
        activations = [float(item["activation"]) for item in items]
        ranks = [int(item["rank"]) for item in items]
        rows.append(
            {
                "condition_family": condition_family,
                "layer": layer,
                "feature_id": feature_id,
                "position_count_present": len(positions),
                "mean_activation_when_present": sum(activations) / len(activations),
                "max_activation": max(activations),
                "best_rank": min(ranks),
                "positions_present": ",".join(positions),
            }
        )
    rows.sort(key=lambda row: (row["condition_family"], int(row["layer"]), -int(row["position_count_present"]), int(row["best_rank"]), int(row["feature_id"])))
    return rows


def build_comparison_rows(topk_rows: list[dict[str, Any]], generated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    topk_sets: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for row in topk_rows:
        topk_sets[(row["condition_family"], int(row["layer"]), row["position_label"])].add(int(row["feature_id"]))

    class_by_condition = {
        row["condition_family"]: classify_start(row["forced_prefix"], row["generated_text_start"])
        for row in generated_rows
    }

    rows = []
    for condition in EXPECTED_CONDITIONS:
        for layer in LAYERS:
            distances = {}
            all_distances = []
            for position in POSITION_LABELS:
                base_set = topk_sets.get(("e_only_no_prefix", layer, position), set())
                condition_set = topk_sets.get((condition, layer, position), set())
                distance = jaccard_distance(base_set, condition_set) if base_set and condition_set else ""
                if distance != "":
                    all_distances.append(distance)
                distances[position] = distance
            cls = class_by_condition[condition]
            rows.append(
                {
                    "condition_family": condition,
                    "layer": layer,
                    "mean_jaccard_distance_vs_no_prefix": sum(all_distances) / len(all_distances) if all_distances else "",
                    "final_prompt_token_jaccard_distance_vs_no_prefix": distances.get("final_prompt_token", ""),
                    "generated_token_1_jaccard_distance_vs_no_prefix": distances.get("generated_token_1", ""),
                    "generated_token_5_jaccard_distance_vs_no_prefix": distances.get("generated_token_5", ""),
                    "generated_token_20_jaccard_distance_vs_no_prefix": distances.get("generated_token_20", ""),
                    "generated_start_class": cls["generated_start_class"],
                    "diacritic_echo_present": cls["diacritic_echo_present"],
                    "ordinary_meta_think_present": cls["ordinary_meta_think_present"],
                    "direct_answer_present": cls["direct_answer_present"],
                }
            )
    return rows


def write_summary(path: Path, generated_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], skipped_positions: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {row["condition_family"]: row for row in generated_rows}
    class_by_condition = {
        row["condition_family"]: classify_start(row["forced_prefix"], row["generated_text_start"])
        for row in generated_rows
    }
    echo_conditions = [c for c, cls in class_by_condition.items() if cls["generated_start_class"] == "echo_like"]
    think_conditions = [c for c, cls in class_by_condition.items() if cls["generated_start_class"] == "ordinary_think"]
    direct_conditions = [c for c, cls in class_by_condition.items() if cls["generated_start_class"] == "direct_answer"]
    other_conditions = [c for c, cls in class_by_condition.items() if cls["generated_start_class"] == "other"]

    layer_means = defaultdict(list)
    position_means = defaultdict(list)
    gen20_separated = []
    for row in comparison_rows:
        if row["condition_family"] == "e_only_no_prefix":
            continue
        mean_value = row["mean_jaccard_distance_vs_no_prefix"]
        if mean_value != "":
            layer_means[int(row["layer"])].append(float(mean_value))
        for position, key in [
            ("final_prompt_token", "final_prompt_token_jaccard_distance_vs_no_prefix"),
            ("generated_token_1", "generated_token_1_jaccard_distance_vs_no_prefix"),
            ("generated_token_5", "generated_token_5_jaccard_distance_vs_no_prefix"),
            ("generated_token_20", "generated_token_20_jaccard_distance_vs_no_prefix"),
        ]:
            value = row[key]
            if value != "":
                position_means[position].append(float(value))
        value20 = row["generated_token_20_jaccard_distance_vs_no_prefix"]
        if value20 != "" and float(value20) >= 0.5:
            gen20_separated.append((row["condition_family"], int(row["layer"]), float(value20)))

    layer_summary = {str(layer): sum(values) / len(values) for layer, values in layer_means.items() if values}
    position_summary = {position: sum(values) / len(values) for position, values in position_means.items() if values}

    no_prefix_start = by_condition["e_only_no_prefix"]["generated_text_start"]
    no_prefix_reproduced = "Do not rēport on thē tēxt" in no_prefix_start or "rēport" in no_prefix_start

    lines = [
        "# E-Only Prefix Intervention Summary",
        "",
        "Evidence-only summary. This run prepends answer-prefix text to the e_only prompt as prompt text; it is not residual steering and not SAE feature steering.",
        "",
        "## Generated Starts",
        "",
    ]
    for condition in EXPECTED_CONDITIONS:
        row = by_condition[condition]
        cls = class_by_condition[condition]
        lines.append(f"- `{condition}`: class={cls['generated_start_class']}; forced_prefix=`{clean_cell(row['forced_prefix'])}`; generated_start={clean_cell(row['generated_text_start'])}")

    lines.extend(["", "## Which Prefixes Preserved The E-Only Diacritic-Echo Path?", ""])
    lines.append("- Echo-like conditions by simple string heuristic: " + (", ".join(f"`{c}`" for c in echo_conditions) if echo_conditions else "none") + ".")

    lines.extend(["", "## Which Prefixes Snapped Generation Into Ordinary Meta-Answer Mode?", ""])
    lines.append("- Ordinary-think conditions by simple string heuristic: " + (", ".join(f"`{c}`" for c in think_conditions) if think_conditions else "none") + ".")

    lines.extend(["", "## Which Prefixes Produced Direct-Answer Mode?", ""])
    lines.append("- Direct-answer conditions by simple string heuristic: " + (", ".join(f"`{c}`" for c in direct_conditions) if direct_conditions else "none") + ".")
    if other_conditions:
        lines.append("- Other conditions by simple string heuristic: " + ", ".join(f"`{c}`" for c in other_conditions) + ".")

    lines.extend(["", "## Does No-Prefix E-Only Reproduce The Prior Weird Start?", ""])
    lines.append(f"- No-prefix generated start: {clean_cell(no_prefix_start)}")
    lines.append(f"- Reproduced prior `Do not rēport...` style by string check: {str(no_prefix_reproduced).lower()}.")

    lines.extend(["", "## Do Layer 14 Or Layer 26 Trajectories Differ More By Prefix?", ""])
    for layer, value in sorted(layer_summary.items(), key=lambda item: float(item[1]), reverse=True):
        lines.append(f"- Layer {layer}: mean TopK Jaccard distance versus no-prefix = {value:.6f}.")

    lines.extend(["", "## Do Differences Concentrate At Boundary Or Generated Positions?", ""])
    for position, value in sorted(position_summary.items(), key=lambda item: float(item[1]), reverse=True):
        lines.append(f"- `{position}`: mean TopK Jaccard distance versus no-prefix = {value:.6f}.")

    lines.extend(["", "## Does Any Prefix Remain Separated Through Generated Token 20?", ""])
    if gen20_separated:
        for condition, layer, value in sorted(gen20_separated, key=lambda item: (item[0], item[1])):
            lines.append(f"- `{condition}` layer {layer}: generated_token_20 Jaccard distance versus no-prefix = {value:.6f}.")
    else:
        lines.append("- No prefix/layer pair exceeded 0.5 Jaccard distance versus no-prefix at generated_token_20.")

    lines.extend(["", "## Skipped Positions", ""])
    if skipped_positions:
        counts = Counter(row["reason"] for row in skipped_positions)
        for reason, count in sorted(counts.items()):
            lines.append(f"- {reason}: {count}.")
    else:
        lines.append("- No required positions were skipped.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "echo_like_conditions": echo_conditions,
        "ordinary_think_conditions": think_conditions,
        "direct_answer_conditions": direct_conditions,
        "other_conditions": other_conditions,
        "no_prefix_reproduced_prior_weird_start": no_prefix_reproduced,
        "mean_jaccard_distance_by_layer": layer_summary,
        "mean_jaccard_distance_by_position": position_summary,
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
    prompts = read_prompts()
    saes = {layer: load_sae(layer) for layer in LAYERS}
    tokenizer, model, model_layers, input_device = load_model()

    topk_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    skipped_positions: list[dict[str, Any]] = []
    capture_records: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts, start=1):
        prompt_id = prompt["prompt_id"]
        condition_family = prompt["condition_family"]
        base_prompt_text = prompt["base_prompt_text"]
        forced_prefix = prompt["forced_prefix"]
        full_prompt_text = prompt["full_prompt_text"]
        base_tokens = tokenizer(base_prompt_text, return_tensors="pt")
        full_tokens = tokenizer(full_prompt_text, return_tensors="pt")
        prefix_token_count = int(tokenizer(forced_prefix, return_tensors="pt")["input_ids"].shape[1]) if forced_prefix else 0
        base_prompt_token_count = int(base_tokens["input_ids"].shape[1])
        full_prompt_token_count = int(full_tokens["input_ids"].shape[1])
        encoded_prompt = {key: value.to(input_device) for key, value in full_tokens.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_token_count = int(generated_ids.shape[1] - full_prompt_token_count)
        generated_token_ids = generated_ids[0, full_prompt_token_count:].detach().cpu()
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        generated_text_start = generated_text[:240]
        generated_rows.append(
            {
                "prompt_id": prompt_id,
                "condition_family": condition_family,
                "forced_prefix": forced_prefix,
                "base_prompt_token_count": base_prompt_token_count,
                "full_prompt_token_count": full_prompt_token_count,
                "forced_prefix_token_count": prefix_token_count,
                "generated_token_count": generated_token_count,
                "generated_text": generated_text,
                "generated_text_start": generated_text_start,
            }
        )

        full_encoded = {
            "input_ids": generated_ids.to(input_device),
            "attention_mask": torch.ones_like(generated_ids, device=input_device),
        }
        hidden_by_layer = capture_selected_layers(model, model_layers, full_encoded)
        requested_positions = [("final_prompt_token", full_prompt_token_count - 1)]
        requested_positions.extend((f"generated_token_{idx}", full_prompt_token_count + idx - 1) for idx in GENERATION_TOKEN_POSITIONS)

        for position_label, token_position in requested_positions:
            if token_position >= int(generated_ids.shape[1]):
                skipped_positions.append(
                    {
                        "prompt_id": prompt_id,
                        "condition_family": condition_family,
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
                hidden_sequence = hidden_by_layer[layer]
                vector = hidden_sequence[token_position]
                encoded_rows, stats = encode_topk50(vector, saes[layer])
                capture_records.append(
                    {
                        "prompt_id": prompt_id,
                        "condition_family": condition_family,
                        "layer": layer,
                        "position_label": position_label,
                        "token_position": token_position,
                        "token_string": token_string,
                        "positive_count_after_relu": stats["positive_count_after_relu"],
                        "sparse_nonzero_count": stats["sparse_nonzero_count"],
                        "max_activation": stats["max_activation"],
                    }
                )
                for encoded_row in encoded_rows:
                    topk_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "condition_family": condition_family,
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
                    "prompt_count": len(prompts),
                    "prompt_id": prompt_id,
                    "condition_family": condition_family,
                    "base_prompt_token_count": base_prompt_token_count,
                    "full_prompt_token_count": full_prompt_token_count,
                    "forced_prefix_token_count": prefix_token_count,
                    "generated_token_count": generated_token_count,
                    "topk_rows_so_far": len(topk_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.cuda.empty_cache()

    presence_rows = build_presence_summary(topk_rows)
    comparison_rows = build_comparison_rows(topk_rows, generated_rows)
    summary_metrics = write_summary(OUT_DIR / "e_only_prefix_intervention_summary.md", generated_rows, comparison_rows, skipped_positions)

    write_tsv(
        OUT_DIR / "topk_features_by_prompt_layer_position.tsv",
        [
            "prompt_id",
            "condition_family",
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
        OUT_DIR / "generated_text_by_prompt.tsv",
        [
            "prompt_id",
            "condition_family",
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
        OUT_DIR / "trajectory_feature_presence_summary.tsv",
        [
            "condition_family",
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
        OUT_DIR / "prefix_intervention_comparison.tsv",
        [
            "condition_family",
            "layer",
            "mean_jaccard_distance_vs_no_prefix",
            "final_prompt_token_jaccard_distance_vs_no_prefix",
            "generated_token_1_jaccard_distance_vs_no_prefix",
            "generated_token_5_jaccard_distance_vs_no_prefix",
            "generated_token_20_jaccard_distance_vs_no_prefix",
            "generated_start_class",
            "diacritic_echo_present",
            "ordinary_meta_think_present",
            "direct_answer_present",
        ],
        comparison_rows,
    )

    assert_no_nan_inf(topk_rows, ["layer", "token_position", "feature_id", "activation", "rank", "base_prompt_token_count", "full_prompt_token_count", "forced_prefix_token_count", "generated_token_count"], "topk")
    assert_no_nan_inf(presence_rows, ["layer", "feature_id", "position_count_present", "mean_activation_when_present", "max_activation", "best_rank"], "presence")
    assert_no_nan_inf(comparison_rows, ["layer", "mean_jaccard_distance_vs_no_prefix", "final_prompt_token_jaccard_distance_vs_no_prefix", "generated_token_1_jaccard_distance_vs_no_prefix", "generated_token_5_jaccard_distance_vs_no_prefix", "generated_token_20_jaccard_distance_vs_no_prefix"], "comparison")

    finished_at = utc_now()
    expected_rows = len(prompts) * len(LAYERS) * len(POSITION_LABELS) * TOP_K
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
        "generation_settings": {"temperature": 0, "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy"},
        "prompt_count": len(prompts),
        "captured_prompt_layer_position_count": len(capture_records),
        "expected_prompt_layer_position_count_if_no_skips": len(prompts) * len(LAYERS) * len(POSITION_LABELS),
        "topk_row_count": len(topk_rows),
        "expected_topk_row_count_if_no_skips": expected_rows,
        "generated_text_row_count": len(generated_rows),
        "presence_summary_row_count": len(presence_rows),
        "comparison_row_count": len(comparison_rows),
        "skipped_positions": skipped_positions,
        "summary_metrics": summary_metrics,
        "selected_layer_hooks_used": True,
        "output_hidden_states_used": False,
        "intervention_type": "prefix intervention only; forced prefixes are prompt text",
        "official_sae_encoding_path": "pre = residual @ W_enc.T + b_enc; relu = ReLU(pre); TopK-50; scatter into sparse activation vector",
        "device": str(input_device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "restrictions": {
            "no_residual_steering": True,
            "no_sae_feature_steering": True,
            "no_hauhau": True,
            "no_llama_cpp": True,
            "no_all_layer_expansion": True,
            "no_full_experiment_expansion": True,
            "no_semantic_labels": True,
        },
    }
    (OUT_DIR / "e_only_prefix_intervention_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_paths = [
        OUT_DIR / "topk_features_by_prompt_layer_position.tsv",
        OUT_DIR / "trajectory_feature_presence_summary.tsv",
        OUT_DIR / "generated_text_by_prompt.tsv",
        OUT_DIR / "prefix_intervention_comparison.tsv",
        OUT_DIR / "e_only_prefix_intervention_metadata.json",
        OUT_DIR / "e_only_prefix_intervention_summary.md",
    ]
    provenance = [
        f"timestamp={finished_at}",
        f"prompt_tsv_path={PROMPT_TSV_PATH}",
        f"script_path={SCRIPT_PATH}",
        f"model_path={MODEL_PATH}",
        f"sae_paths={json.dumps({str(layer): saes[layer]['path'] for layer in LAYERS}, sort_keys=True)}",
        f"layers={','.join(str(layer) for layer in LAYERS)}",
        f"capture_positions={','.join(POSITION_LABELS)}",
        "generation_settings=temperature=0;do_sample=false;max_new_tokens=24;greedy=true",
        f"prompt_count={len(prompts)}",
        f"captured_prompt_layer_position_count={len(capture_records)}",
        f"topk_row_count={len(topk_rows)}",
        f"generated_text_row_count={len(generated_rows)}",
        "outputs_written=" + ",".join(str(path) for path in output_paths),
        "selected_layer_hooks_used=true",
        "output_hidden_states=True was not used",
        "prefix_intervention_only=true",
        "no_residual_steering=true",
        "no_SAE_feature_steering=true",
        "no_Hauhau=true",
        "no_llama_cpp=true",
        "no_all_layer_expansion=true",
        "no_full_experiment_expansion=true",
        "no_semantic_labels=true",
    ]
    PROVENANCE_PATH.write_text("\n".join(provenance) + "\n", encoding="utf-8")
    print(json.dumps({"status": "e_only_prefix_intervention_complete", "topk_row_count": len(topk_rows), "generated_text_row_count": len(generated_rows), "skipped_position_count": len(skipped_positions), "metadata_path": str(OUT_DIR / "e_only_prefix_intervention_metadata.json"), "provenance_path": str(PROVENANCE_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
