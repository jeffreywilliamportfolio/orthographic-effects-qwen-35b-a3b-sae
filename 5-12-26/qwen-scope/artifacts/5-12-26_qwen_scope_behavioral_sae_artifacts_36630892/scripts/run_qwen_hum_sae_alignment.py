#!/usr/bin/env python3
"""Capture layer-14/layer-26 residuals and encode Qwen-Scope TopK-50 features."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/qwen-scope/5-12-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_DIR = ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
PROMPT_MATRIX_PATH = ROOT / "prompts" / "hum_behavioral_perturbation_matrix.tsv"
OUT_DIR = ROOT / "sae_outputs" / "hum_behavioral_sae_alignment"
TOPK_TSV = OUT_DIR / "topk_features_by_prompt_layer_position.tsv"
DELTA_TSV = OUT_DIR / "perturbation_delta_vs_ascii.tsv"
JACCARD_TSV = OUT_DIR / "topk_jaccard_vs_ascii.tsv"
METADATA_JSON = OUT_DIR / "hum_behavioral_sae_alignment_metadata.json"
OFFLOAD_DIR = ROOT / ".offload" / "qwen_hum_sae_alignment"

SELECTED_LAYERS = [26, 14]
CAPTURE_POSITIONS = [
    ("final_prompt_token", 0),
    ("final_prompt_token_minus_1", 1),
    ("final_prompt_token_minus_5", 5),
    ("final_prompt_token_minus_10", 10),
]
TOP_K = 50
PERTURBATION_ORDER = [
    "ascii_original",
    "d_to_ḑ",
    "e_to_ē",
    "d_plus_e",
    "s_to_ş",
    "s_to_ṡ",
    "all_diacritics",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def load_prompt_rows() -> list[dict[str, str]]:
    with PROMPT_MATRIX_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = ["prompt_id", "prompt_family", "perturbation_type", "prompt_text", "notes"]
        if reader.fieldnames != expected:
            raise ValueError(f"Prompt matrix schema mismatch: {reader.fieldnames} != {expected}")
        rows = list(reader)
    if len(rows) != 7:
        raise ValueError(f"Expected 7 prompt rows, found {len(rows)}")
    observed = [row["perturbation_type"] for row in rows]
    if observed != PERTURBATION_ORDER:
        raise ValueError(f"Unexpected perturbation order: {observed}")
    return rows


def load_sae(layer: int) -> dict[str, Any]:
    path = SAE_DIR / f"layer{layer}.sae.pt"
    try:
        sae = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sae = torch.load(path, map_location="cpu")
    if not isinstance(sae, dict):
        raise TypeError(f"Expected SAE checkpoint dict at {path}, got {type(sae).__name__}")
    for key in ("W_enc", "b_enc"):
        if key not in sae or not torch.is_tensor(sae[key]):
            shapes = {name: list(value.shape) for name, value in sae.items() if torch.is_tensor(value)}
            raise KeyError(f"SAE layer {layer} missing tensor key {key}; tensor_shapes={shapes}")
    w_enc = sae["W_enc"]
    b_enc = sae["b_enc"]
    if w_enc.ndim != 2 or b_enc.ndim != 1:
        raise ValueError(f"Unexpected SAE shapes for layer {layer}: W_enc={tuple(w_enc.shape)}, b_enc={tuple(b_enc.shape)}")
    if w_enc.shape[0] == b_enc.shape[0]:
        w_enc_t = w_enc.T.to(dtype=torch.float32).contiguous()
    elif w_enc.shape[1] == b_enc.shape[0]:
        w_enc_t = w_enc.to(dtype=torch.float32).contiguous()
    else:
        raise ValueError(f"Cannot align layer {layer} W_enc={tuple(w_enc.shape)} with b_enc={tuple(b_enc.shape)}")
    return {
        "path": str(path),
        "W_enc_source_shape": list(w_enc.shape),
        "W_enc_transposed_shape": list(w_enc_t.shape),
        "b_enc_shape": list(b_enc.shape),
        "_W_enc": w_enc_t,
        "_b_enc": b_enc.to(dtype=torch.float32).contiguous(),
    }


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate decoder layers on model.model.layers or model.transformer.h")


def capture_selected_layer_sequences(
    model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    encoded: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
            hidden = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hidden[0].detach().to("cpu", dtype=torch.float32)

        return hook

    for layer_idx in SELECTED_LAYERS:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in SELECTED_LAYERS if layer not in captured]
    if missing:
        raise RuntimeError(f"Selected-layer hooks did not capture layers: {missing}")
    return captured


def encode_topk50(vector: torch.Tensor, sae: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[int], list[float]]:
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D residual vector, got {tuple(vector.shape)}")
    w_enc = sae["_W_enc"]
    b_enc = sae["_b_enc"]
    if vector.shape[0] != w_enc.shape[0]:
        raise ValueError(f"Vector hidden size {vector.shape[0]} does not match SAE hidden size {w_enc.shape[0]}")
    pre = vector.to(dtype=torch.float32) @ w_enc + b_enc
    relu = torch.relu(pre)
    values, indices = torch.topk(relu, k=min(TOP_K, relu.numel()), dim=-1)
    sparse = torch.zeros_like(relu)
    sparse.scatter_(-1, indices, values)
    sparse_values, sparse_indices = torch.topk(sparse, k=min(TOP_K, sparse.numel()), dim=-1)
    features: dict[int, dict[str, Any]] = {}
    for rank, (feature_id, activation) in enumerate(zip(sparse_indices.tolist(), sparse_values.tolist()), start=1):
        features[int(feature_id)] = {"activation": float(activation), "rank": rank}
    return features, [int(x) for x in sparse_indices.tolist()], [float(x) for x in sparse_values.tolist()]


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    prompt_rows = load_prompt_rows()
    saes = {layer: load_sae(layer) for layer in SELECTED_LAYERS}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    layers = decoder_layers(model)
    for layer_idx in SELECTED_LAYERS:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer {layer_idx} outside available layer range 0..{len(layers) - 1}")
    input_device = model.get_input_embeddings().weight.device

    topk_rows: list[dict[str, Any]] = []
    encoded_results: dict[tuple[str, int, str], dict[str, Any]] = {}
    prompt_metadata: dict[str, Any] = {}
    skipped_positions: list[dict[str, Any]] = []
    capture_count = 0

    for prompt_idx, prompt in enumerate(prompt_rows, start=1):
        prompt_id = prompt["prompt_id"]
        perturbation_type = prompt["perturbation_type"]
        encoded_cpu = tokenizer(prompt["prompt_text"], return_tensors="pt")
        prompt_token_count = int(encoded_cpu["input_ids"].shape[1])
        final_index = prompt_token_count - 1
        encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}
        hidden_sequences = capture_selected_layer_sequences(model, layers, encoded)
        prompt_metadata[prompt_id] = {
            "prompt_family": prompt["prompt_family"],
            "perturbation_type": perturbation_type,
            "prompt_token_count": prompt_token_count,
            "positions": {},
        }

        for position_label, offset in CAPTURE_POSITIONS:
            token_position = final_index - offset
            if token_position < 0:
                skipped_positions.append(
                    {
                        "prompt_id": prompt_id,
                        "perturbation_type": perturbation_type,
                        "position_label": position_label,
                        "reason": "prompt too short",
                        "prompt_token_count": prompt_token_count,
                    }
                )
                continue
            token_id = int(encoded_cpu["input_ids"][0, token_position].item())
            token_string = clean_cell(tokenizer.decode([token_id]))
            prompt_metadata[prompt_id]["positions"][position_label] = {
                "token_position": token_position,
                "token_string": token_string,
            }
            for layer_idx in SELECTED_LAYERS:
                hidden_sequence = hidden_sequences[layer_idx]
                if hidden_sequence.shape[0] != prompt_token_count:
                    raise RuntimeError(
                        f"{prompt_id} layer {layer_idx} hidden length {hidden_sequence.shape[0]} "
                        f"!= prompt token count {prompt_token_count}"
                    )
                features, feature_ids, activations = encode_topk50(hidden_sequence[token_position, :], saes[layer_idx])
                for rank, feature_id in enumerate(feature_ids, start=1):
                    topk_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "prompt_family": prompt["prompt_family"],
                            "perturbation_type": perturbation_type,
                            "layer": layer_idx,
                            "position_label": position_label,
                            "token_position": token_position,
                            "token_string": token_string,
                            "feature_id": feature_id,
                            "activation": activations[rank - 1],
                            "rank": rank,
                            "prompt_token_count": prompt_token_count,
                        }
                    )
                encoded_results[(perturbation_type, layer_idx, position_label)] = {
                    "prompt_id": prompt_id,
                    "prompt_token_count": prompt_token_count,
                    "features": features,
                    "topk_set": set(features),
                }
                capture_count += 1

        print(f"captured {prompt_idx}/7 {prompt_id} tokens={prompt_token_count}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    delta_rows: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []
    for perturbation_type in PERTURBATION_ORDER:
        for layer_idx in SELECTED_LAYERS:
            for position_label, _offset in CAPTURE_POSITIONS:
                ascii_result = encoded_results[("ascii_original", layer_idx, position_label)]
                pert_result = encoded_results[(perturbation_type, layer_idx, position_label)]
                ascii_features = ascii_result["features"]
                pert_features = pert_result["features"]
                union_features = sorted(set(ascii_features) | set(pert_features))
                for feature_id in union_features:
                    ascii_data = ascii_features.get(feature_id)
                    pert_data = pert_features.get(feature_id)
                    ascii_activation = float(ascii_data["activation"]) if ascii_data else 0.0
                    pert_activation = float(pert_data["activation"]) if pert_data else 0.0
                    delta = pert_activation - ascii_activation
                    delta_rows.append(
                        {
                            "prompt_family": "original_hum",
                            "perturbation_type": perturbation_type,
                            "layer": layer_idx,
                            "position_label": position_label,
                            "feature_id": feature_id,
                            "ascii_activation": ascii_activation,
                            "perturbation_activation": pert_activation,
                            "delta": delta,
                            "abs_delta": abs(delta),
                            "ascii_rank": ascii_data["rank"] if ascii_data else "",
                            "perturbation_rank": pert_data["rank"] if pert_data else "",
                            "ascii_present": "1" if ascii_data else "0",
                            "perturbation_present": "1" if pert_data else "0",
                            "ascii_prompt_id": ascii_result["prompt_id"],
                            "perturbation_prompt_id": pert_result["prompt_id"],
                            "ascii_prompt_token_count": ascii_result["prompt_token_count"],
                            "perturbation_prompt_token_count": pert_result["prompt_token_count"],
                            "token_count_delta_vs_ascii": int(pert_result["prompt_token_count"]) - int(ascii_result["prompt_token_count"]),
                        }
                    )
                intersection_count = len(ascii_result["topk_set"] & pert_result["topk_set"])
                ascii_count = len(ascii_result["topk_set"])
                pert_count = len(pert_result["topk_set"])
                union_count = len(ascii_result["topk_set"] | pert_result["topk_set"])
                jaccard = intersection_count / union_count if union_count else 0.0
                jaccard_rows.append(
                    {
                        "prompt_family": "original_hum",
                        "perturbation_type": perturbation_type,
                        "layer": layer_idx,
                        "position_label": position_label,
                        "ascii_prompt_id": ascii_result["prompt_id"],
                        "perturbation_prompt_id": pert_result["prompt_id"],
                        "topk_jaccard": jaccard,
                        "topk_jaccard_distance": 1.0 - jaccard,
                        "ascii_topk_count": ascii_count,
                        "perturbation_topk_count": pert_count,
                        "intersection_count": intersection_count,
                        "union_count": union_count,
                        "ascii_prompt_token_count": ascii_result["prompt_token_count"],
                        "perturbation_prompt_token_count": pert_result["prompt_token_count"],
                        "token_count_delta_vs_ascii": int(pert_result["prompt_token_count"]) - int(ascii_result["prompt_token_count"]),
                    }
                )

    write_tsv(
        TOPK_TSV,
        topk_rows,
        [
            "prompt_id",
            "prompt_family",
            "perturbation_type",
            "layer",
            "position_label",
            "token_position",
            "token_string",
            "feature_id",
            "activation",
            "rank",
            "prompt_token_count",
        ],
    )
    write_tsv(
        DELTA_TSV,
        delta_rows,
        [
            "prompt_family",
            "perturbation_type",
            "layer",
            "position_label",
            "feature_id",
            "ascii_activation",
            "perturbation_activation",
            "delta",
            "abs_delta",
            "ascii_rank",
            "perturbation_rank",
            "ascii_present",
            "perturbation_present",
            "ascii_prompt_id",
            "perturbation_prompt_id",
            "ascii_prompt_token_count",
            "perturbation_prompt_token_count",
            "token_count_delta_vs_ascii",
        ],
    )
    write_tsv(
        JACCARD_TSV,
        jaccard_rows,
        [
            "prompt_family",
            "perturbation_type",
            "layer",
            "position_label",
            "ascii_prompt_id",
            "perturbation_prompt_id",
            "topk_jaccard",
            "topk_jaccard_distance",
            "ascii_topk_count",
            "perturbation_topk_count",
            "intersection_count",
            "union_count",
            "ascii_prompt_token_count",
            "perturbation_prompt_token_count",
            "token_count_delta_vs_ascii",
        ],
    )
    metadata = {
        "timestamp_utc": utc_now(),
        "workspace_root": str(ROOT),
        "model_path": str(MODEL_PATH),
        "sae_dir": str(SAE_DIR),
        "sae_paths": {str(layer): saes[layer]["path"] for layer in SELECTED_LAYERS},
        "script_path": str(ROOT / "scripts" / "run_qwen_hum_sae_alignment.py"),
        "prompt_matrix_path": str(PROMPT_MATRIX_PATH),
        "layers": SELECTED_LAYERS,
        "capture_positions": [label for label, _ in CAPTURE_POSITIONS],
        "encoding_path": "official Qwen-Scope TopK-50: pre=hidden@W_enc.T+b_enc; relu; topk; scatter",
        "model_device_map": "single_gpu_cuda0",
        "model_device_map_reason": "device_map=auto produced NaN hidden states on this fresh instance; single GPU fits in 98GB VRAM and produced finite diagnostics",
        "selected_layer_hooks_used": True,
        "all_hidden_states_requested": False,
        "prompt_count": len(prompt_rows),
        "prompt_position_layer_capture_count": capture_count,
        "expected_topk_rows": len(prompt_rows) * len(SELECTED_LAYERS) * len(CAPTURE_POSITIONS) * TOP_K,
        "topk_rows": len(topk_rows),
        "delta_rows": len(delta_rows),
        "jaccard_rows": len(jaccard_rows),
        "skipped_positions": skipped_positions,
        "sae_shapes": {
            str(layer): {
                "W_enc_source_shape": saes[layer]["W_enc_source_shape"],
                "W_enc_transposed_shape": saes[layer]["W_enc_transposed_shape"],
                "b_enc_shape": saes[layer]["b_enc_shape"],
            }
            for layer in SELECTED_LAYERS
        },
        "outputs": {
            "topk_features_by_prompt_layer_position": str(TOPK_TSV),
            "perturbation_delta_vs_ascii": str(DELTA_TSV),
            "topk_jaccard_vs_ascii": str(JACCARD_TSV),
            "metadata": str(METADATA_JSON),
        },
        "prompt_metadata": prompt_metadata,
        "restrictions": [
            "no steering",
            "no Hauhau",
            "no llama.cpp",
            "no all-layer expansion",
            "no all-hidden-state request",
            "no semantic SAE feature labels",
        ],
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"topk_rows={len(topk_rows)}")
    print(f"capture_count={capture_count}")
    print(f"metadata={METADATA_JSON}")


if __name__ == "__main__":
    main()
