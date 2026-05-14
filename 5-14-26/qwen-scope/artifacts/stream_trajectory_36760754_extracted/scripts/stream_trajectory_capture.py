#!/usr/bin/env python3
"""Token-by-token Qwen-Scope SAE trajectory capture for the 5-14 run."""

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
PROMPT_TSV_PATH = ROOT / "prompts" / "stream_trajectory_prompts.tsv"
SCRIPT_PATH = ROOT / "scripts" / "stream_trajectory_capture.py"
OUT_DIR = ROOT / "sae_outputs" / "stream_trajectory_capture"
PROVENANCE_PATH = ROOT / "provenance" / "stream_trajectory_capture_20260514.txt"

LAYERS = [14, 26]
TOP_K = 50
MAX_NEW_TOKENS = 24
GENERATION_TOKEN_POSITIONS = [1, 2, 3, 4, 5, 10, 20]
EXPECTED_CONDITIONS = [
    "ascii_control",
    "d_only",
    "e_only",
    "s_only",
    "s_c_only",
    "e_d_high_impact_only",
    "e_d_shuffled",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean_cell(row.get(name, "")) for name in fieldnames})


def read_prompts() -> list[dict[str, str]]:
    with PROMPT_TSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = ["prompt_id", "condition_family", "prompt_text", "notes"]
        if reader.fieldnames != expected:
            raise ValueError(f"Prompt TSV schema mismatch: {reader.fieldnames} != {expected}")
        rows = list(reader)
    if len(rows) != 7:
        raise ValueError(f"Expected exactly 7 prompt rows, found {len(rows)}")
    observed = [row["condition_family"] for row in rows]
    if observed != EXPECTED_CONDITIONS:
        raise ValueError(f"Condition order mismatch: {observed} != {EXPECTED_CONDITIONS}")
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt IDs are not unique")
    return rows


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate decoder layers on model.model.layers or model.transformer.h")


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
    input_device = model.get_input_embeddings().weight.device
    return tokenizer, model, layers, input_device


def jaccard_distance(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / len(a | b))


def build_presence_summary(topk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in topk_rows:
        grouped[(row["condition_family"], int(row["layer"]), int(row["feature_id"]))].append(row)

    summary_rows = []
    for (condition_family, layer, feature_id), rows in grouped.items():
        positions = sorted({row["position_label"] for row in rows})
        activations = [float(row["activation"]) for row in rows]
        ranks = [int(row["rank"]) for row in rows]
        summary_rows.append(
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
    summary_rows.sort(
        key=lambda row: (
            row["condition_family"],
            int(row["layer"]),
            -int(row["position_count_present"]),
            int(row["best_rank"]),
            int(row["feature_id"]),
        )
    )
    return summary_rows


def write_summary(
    path: Path,
    generated_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    presence_rows: list[dict[str, Any]],
    skipped_positions: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_counts = {row["condition_family"]: int(row["prompt_token_count"]) for row in generated_rows}
    ascii_count = prompt_counts.get("ascii_control", 0)
    token_inflation = sorted(
        ((condition, count, count - ascii_count) for condition, count in prompt_counts.items()),
        key=lambda item: item[2],
        reverse=True,
    )

    generated_starts = {
        row["condition_family"]: clean_cell(row["generated_text"])[:220]
        for row in generated_rows
    }

    topk_sets: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for row in topk_rows:
        topk_sets[(row["condition_family"], int(row["layer"]), row["position_label"])].add(int(row["feature_id"]))

    jaccard_by_layer: dict[int, list[float]] = defaultdict(list)
    jaccard_by_position: dict[str, list[float]] = defaultdict(list)
    jaccard_by_condition: dict[str, list[float]] = defaultdict(list)
    gen20_rows = []
    for condition in EXPECTED_CONDITIONS:
        if condition == "ascii_control":
            continue
        for layer in LAYERS:
            for position in ["final_prompt_token"] + [f"generated_token_{idx}" for idx in GENERATION_TOKEN_POSITIONS]:
                ascii_set = topk_sets.get(("ascii_control", layer, position), set())
                condition_set = topk_sets.get((condition, layer, position), set())
                if not ascii_set or not condition_set:
                    continue
                dist = jaccard_distance(ascii_set, condition_set)
                jaccard_by_layer[layer].append(dist)
                jaccard_by_position[position].append(dist)
                jaccard_by_condition[condition].append(dist)
                if position == "generated_token_20":
                    gen20_rows.append((condition, layer, dist))

    layer_ranking = sorted(
        ((layer, sum(values) / len(values)) for layer, values in jaccard_by_layer.items() if values),
        key=lambda item: item[1],
        reverse=True,
    )
    position_ranking = sorted(
        ((position, sum(values) / len(values)) for position, values in jaccard_by_position.items() if values),
        key=lambda item: item[1],
        reverse=True,
    )
    condition_ranking = sorted(
        ((condition, sum(values) / len(values)) for condition, values in jaccard_by_condition.items() if values),
        key=lambda item: item[1],
        reverse=True,
    )

    recurring = [row for row in presence_rows if int(row["position_count_present"]) >= 3]
    recurring_by_condition_layer: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in recurring:
        recurring_by_condition_layer[(row["condition_family"], int(row["layer"]))].append(row)

    lines = [
        "# Stream Trajectory Summary",
        "",
        "Evidence-only summary for layer-14 and layer-26 Qwen-Scope TopK-50 trajectories. No semantic feature labels are assigned.",
        "",
        "## Generated Answer Starts",
        "",
    ]
    for condition in EXPECTED_CONDITIONS:
        lines.append(f"- `{condition}`: {generated_starts.get(condition, '')}")

    lines.extend(["", "## Prompt Token Inflation", ""])
    for condition, count, delta in token_inflation:
        lines.append(f"- `{condition}`: prompt_tokens={count}, delta_vs_ascii={delta}.")

    lines.extend(["", "## Layer Trajectory Differences Versus ASCII", ""])
    if layer_ranking:
        for layer, mean_distance in layer_ranking:
            lines.append(f"- Layer {layer}: mean TopK Jaccard distance versus ASCII = {mean_distance:.6f}.")
    else:
        lines.append("- No paired ASCII/non-ASCII Jaccard distances were available.")

    lines.extend(["", "## Position Concentration", ""])
    for position, mean_distance in position_ranking:
        lines.append(f"- `{position}`: mean TopK Jaccard distance versus ASCII = {mean_distance:.6f}.")

    lines.extend(["", "## Condition Separation Versus ASCII", ""])
    for condition, mean_distance in condition_ranking:
        lines.append(f"- `{condition}`: mean TopK Jaccard distance versus ASCII = {mean_distance:.6f}.")

    lines.extend(["", "## Recurring Features Within Condition And Layer", ""])
    for condition in EXPECTED_CONDITIONS:
        for layer in LAYERS:
            rows = recurring_by_condition_layer.get((condition, layer), [])[:8]
            if not rows:
                lines.append(f"- `{condition}` layer {layer}: no feature appeared in 3 or more captured positions.")
                continue
            parts = [
                f"feature {row['feature_id']} in {row['position_count_present']} positions"
                for row in rows
            ]
            lines.append(f"- `{condition}` layer {layer}: " + "; ".join(parts) + ".")

    lines.extend(["", "## Generated Token 20 Separation", ""])
    if gen20_rows:
        for condition, layer, distance in sorted(gen20_rows, key=lambda item: (item[0], item[1])):
            lines.append(f"- `{condition}` layer {layer} at generated_token_20: TopK Jaccard distance versus ASCII = {distance:.6f}.")
    else:
        lines.append("- generated_token_20 was not available for enough paired conditions; skipped positions record the reason.")

    lines.extend(["", "## Skipped Positions", ""])
    if skipped_positions:
        by_reason = Counter(row["reason"] for row in skipped_positions)
        for reason, count in sorted(by_reason.items()):
            lines.append(f"- {reason}: {count}.")
    else:
        lines.append("- No required positions were skipped.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "mean_jaccard_distance_by_layer": {str(layer): value for layer, value in layer_ranking},
        "mean_jaccard_distance_by_position": {position: value for position, value in position_ranking},
        "mean_jaccard_distance_by_condition": {condition: value for condition, value in condition_ranking},
        "token_inflation_vs_ascii": [
            {"condition_family": condition, "prompt_token_count": count, "delta_vs_ascii": delta}
            for condition, count, delta in token_inflation
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
    saes = {layer: load_sae(layer) for layer in LAYERS}
    tokenizer, model, model_layers, input_device = load_model()

    topk_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    skipped_positions: list[dict[str, Any]] = []
    capture_records: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompt_rows, start=1):
        prompt_id = prompt["prompt_id"]
        condition_family = prompt["condition_family"]
        prompt_text = prompt["prompt_text"]
        encoded_prompt_cpu = tokenizer(prompt_text, return_tensors="pt")
        prompt_token_count = int(encoded_prompt_cpu["input_ids"].shape[1])
        encoded_prompt = {key: value.to(input_device) for key, value in encoded_prompt_cpu.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_token_count = int(generated_ids.shape[1] - prompt_token_count)
        generated_token_ids = generated_ids[0, prompt_token_count:].detach().cpu()
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        generated_rows.append(
            {
                "prompt_id": prompt_id,
                "condition_family": condition_family,
                "prompt_token_count": prompt_token_count,
                "generated_token_count": generated_token_count,
                "generated_text": generated_text,
            }
        )

        full_attention = torch.ones_like(generated_ids, device=input_device)
        full_encoded = {
            "input_ids": generated_ids.to(input_device),
            "attention_mask": full_attention,
        }
        hidden_by_layer = capture_selected_layers(model, model_layers, full_encoded)

        requested_positions = [("final_prompt_token", prompt_token_count - 1)]
        requested_positions.extend(
            (f"generated_token_{idx}", prompt_token_count + idx - 1)
            for idx in GENERATION_TOKEN_POSITIONS
        )

        for position_label, token_position in requested_positions:
            if token_position < 0:
                skipped_positions.append(
                    {
                        "prompt_id": prompt_id,
                        "condition_family": condition_family,
                        "position_label": position_label,
                        "token_position": token_position,
                        "reason": "negative token position",
                    }
                )
                continue
            if token_position >= int(generated_ids.shape[1]):
                skipped_positions.append(
                    {
                        "prompt_id": prompt_id,
                        "condition_family": condition_family,
                        "position_label": position_label,
                        "token_position": token_position,
                        "reason": f"generated output ended before {position_label}",
                        "prompt_token_count": prompt_token_count,
                        "generated_token_count": generated_token_count,
                    }
                )
                continue

            token_id = int(generated_ids[0, token_position].detach().cpu().item())
            token_string = tokenizer.decode([token_id], skip_special_tokens=False)

            for layer in LAYERS:
                hidden_sequence = hidden_by_layer[layer]
                if token_position >= hidden_sequence.shape[0]:
                    raise RuntimeError(
                        f"{prompt_id} layer {layer} missing token_position={token_position}; "
                        f"hidden_sequence_shape={tuple(hidden_sequence.shape)}"
                    )
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
                            "layer": layer,
                            "position_label": position_label,
                            "token_position": token_position,
                            "token_string": token_string,
                            "feature_id": encoded_row["feature_id"],
                            "activation": encoded_row["activation"],
                            "rank": encoded_row["rank"],
                            "prompt_token_count": prompt_token_count,
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
                    "prompt_token_count": prompt_token_count,
                    "generated_token_count": generated_token_count,
                    "topk_rows_so_far": len(topk_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.cuda.empty_cache()

    presence_rows = build_presence_summary(topk_rows)
    summary_metrics = write_summary(OUT_DIR / "stream_trajectory_summary.md", generated_rows, topk_rows, presence_rows, skipped_positions)

    topk_fieldnames = [
        "prompt_id",
        "condition_family",
        "layer",
        "position_label",
        "token_position",
        "token_string",
        "feature_id",
        "activation",
        "rank",
        "prompt_token_count",
        "generated_token_count",
    ]
    presence_fieldnames = [
        "condition_family",
        "layer",
        "feature_id",
        "position_count_present",
        "mean_activation_when_present",
        "max_activation",
        "best_rank",
        "positions_present",
    ]
    generated_fieldnames = [
        "prompt_id",
        "condition_family",
        "prompt_token_count",
        "generated_token_count",
        "generated_text",
    ]
    write_tsv(OUT_DIR / "topk_features_by_prompt_layer_position.tsv", topk_fieldnames, topk_rows)
    write_tsv(OUT_DIR / "trajectory_feature_presence_summary.tsv", presence_fieldnames, presence_rows)
    write_tsv(OUT_DIR / "generated_text_by_prompt.tsv", generated_fieldnames, generated_rows)

    assert_no_nan_inf(topk_rows, ["activation", "rank", "prompt_token_count", "generated_token_count"], "topk")
    assert_no_nan_inf(presence_rows, ["position_count_present", "mean_activation_when_present", "max_activation", "best_rank"], "presence")

    finished_at = utc_now()
    expected_topk_rows_if_no_skips = len(prompt_rows) * len(LAYERS) * (1 + len(GENERATION_TOKEN_POSITIONS)) * TOP_K
    metadata = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "workspace_root": str(ROOT),
        "path_correction": "Task header named /workspace/qwen-scope/5-14-26; stale 5-11 paths in task body were normalized to 5-14.",
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
        "capture_positions": ["final_prompt_token"] + [f"generated_token_{idx}" for idx in GENERATION_TOKEN_POSITIONS],
        "generation_settings": {
            "temperature": 0,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy",
        },
        "prompt_count": len(prompt_rows),
        "captured_prompt_layer_position_count": len(capture_records),
        "expected_prompt_layer_position_count_if_no_skips": len(prompt_rows) * len(LAYERS) * (1 + len(GENERATION_TOKEN_POSITIONS)),
        "topk_row_count": len(topk_rows),
        "expected_topk_rows_if_no_skips": expected_topk_rows_if_no_skips,
        "generated_text_row_count": len(generated_rows),
        "presence_summary_row_count": len(presence_rows),
        "skipped_positions": skipped_positions,
        "capture_records": capture_records,
        "summary_metrics": summary_metrics,
        "selected_layer_hooks_used": True,
        "output_hidden_states_used": False,
        "official_sae_encoding_path": "pre = residual @ W_enc.T + b_enc; relu = ReLU(pre); TopK-50; scatter into sparse activation vector",
        "device": str(input_device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "restrictions": {
            "no_steering": True,
            "no_hauhau": True,
            "no_llama_cpp": True,
            "no_all_layer_expansion": True,
            "no_full_experiment_expansion": True,
            "no_semantic_labels": True,
        },
    }
    (OUT_DIR / "stream_trajectory_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance_lines = [
        f"timestamp={finished_at}",
        f"prompt_tsv_path={PROMPT_TSV_PATH}",
        f"script_path={SCRIPT_PATH}",
        f"model_path={MODEL_PATH}",
        f"sae_paths={json.dumps({str(layer): saes[layer]['path'] for layer in LAYERS}, sort_keys=True)}",
        f"layers={','.join(str(layer) for layer in LAYERS)}",
        "capture_positions=" + ",".join(metadata["capture_positions"]),
        "generation_settings=temperature=0;do_sample=false;max_new_tokens=24;greedy=true",
        f"prompt_count={len(prompt_rows)}",
        f"captured_prompt_layer_position_count={len(capture_records)}",
        f"topk_row_count={len(topk_rows)}",
        f"generated_text_row_count={len(generated_rows)}",
        "outputs_written=" + ",".join(
            str(path)
            for path in [
                OUT_DIR / "topk_features_by_prompt_layer_position.tsv",
                OUT_DIR / "trajectory_feature_presence_summary.tsv",
                OUT_DIR / "generated_text_by_prompt.tsv",
                OUT_DIR / "stream_trajectory_metadata.json",
                OUT_DIR / "stream_trajectory_summary.md",
            ]
        ),
        "selected_layer_hooks_used=true",
        "output_hidden_states=True was not used",
        "no_steering=true",
        "no_Hauhau=true",
        "no_llama_cpp=true",
        "no_all_layer_expansion=true",
        "no_full_experiment_expansion=true",
        "no_semantic_labels=true",
        "path_correction=stale 5-11 task body paths normalized to /workspace/qwen-scope/5-14-26",
    ]
    PROVENANCE_PATH.write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "stream_trajectory_capture_complete",
                "topk_row_count": len(topk_rows),
                "generated_text_row_count": len(generated_rows),
                "skipped_position_count": len(skipped_positions),
                "metadata_path": str(OUT_DIR / "stream_trajectory_metadata.json"),
                "provenance_path": str(PROVENANCE_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
