#!/usr/bin/env python3
"""Minimal Qwen-Scope SAE encode smoke test.

Loads the already-captured residual vectors for layers 14 and 26, loads the
matching Qwen-Scope SAE checkpoints, encodes the vectors with the official
Qwen-Scope TopK-50 path, and writes only feature IDs plus activation values.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/workspace/qwen-scope/5-11-26")
SAE_DIR = Path(os.environ.get("QWEN_SCOPE_SAE_DIR", ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"))
HIDDEN_DIR = Path(os.environ.get("SMOKE_SAE_HIDDEN_DIR", ROOT / "hidden_states" / "smoke"))
OUT_DIR = Path(os.environ.get("SMOKE_SAE_OUTPUT_DIR", ROOT / "sae_outputs" / "smoke"))
LAYERS = [14, 26]
OFFICIAL_TOP_K = 50
REPORT_TOP_K = 50


@dataclass(frozen=True)
class SaeWeights:
    w_enc: torch.Tensor
    w_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    source_key_prefix: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def shape_dtype(value: Any) -> str:
    if torch.is_tensor(value):
        return f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
    return f"{type(value).__name__} value={repr(value)[:160]}"


def describe_checkpoint(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        summary: dict[str, Any] = {"type": "dict", "keys": list(obj.keys())}
        tensor_shapes = {}
        nested = {}
        for key, value in obj.items():
            if torch.is_tensor(value):
                tensor_shapes[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).replace("torch.", ""),
                }
            elif isinstance(value, dict):
                nested[key] = {
                    subkey: {
                        "shape": list(subvalue.shape),
                        "dtype": str(subvalue.dtype).replace("torch.", ""),
                    }
                    if torch.is_tensor(subvalue)
                    else type(subvalue).__name__
                    for subkey, subvalue in value.items()
                }
        summary["tensor_shapes"] = tensor_shapes
        summary["nested"] = nested
        return summary

    attrs = [name for name in dir(obj) if not name.startswith("_")]
    tensor_attrs = {}
    for name in attrs:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if torch.is_tensor(value):
            tensor_attrs[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
            }
    return {"type": type(obj).__name__, "attrs": attrs[:120], "tensor_attrs": tensor_attrs}


def maybe_state_dict(obj: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "sae_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value, key
        return obj, ""

    if hasattr(obj, "state_dict"):
        state = obj.state_dict()
        if isinstance(state, dict):
            return state, "module.state_dict()"

    return None, ""


def find_first_tensor(mapping: dict[str, Any], names: tuple[str, ...]) -> tuple[str, torch.Tensor] | None:
    lower_to_key = {key.lower(): key for key in mapping.keys()}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None and torch.is_tensor(mapping[key]):
            return key, mapping[key]
    return None


def extract_weights(obj: Any) -> SaeWeights:
    state, prefix = maybe_state_dict(obj)
    if state is None:
        raise TypeError("SAE checkpoint is neither a dict nor a module with state_dict()")

    enc_names = (
        "W_enc",
        "w_enc",
        "encoder.weight",
        "encoder_weight",
        "enc.weight",
        "activation_encoder.weight",
    )
    dec_names = (
        "W_dec",
        "w_dec",
        "decoder.weight",
        "decoder_weight",
        "dec.weight",
        "activation_decoder.weight",
    )
    b_enc_names = (
        "b_enc",
        "encoder.bias",
        "encoder_bias",
        "enc.bias",
        "activation_encoder.bias",
    )
    b_dec_names = (
        "b_dec",
        "decoder.bias",
        "decoder_bias",
        "dec.bias",
        "activation_decoder.bias",
    )

    found_enc = find_first_tensor(state, enc_names)
    found_dec = find_first_tensor(state, dec_names)
    found_b_enc = find_first_tensor(state, b_enc_names)
    found_b_dec = find_first_tensor(state, b_dec_names)

    missing = []
    if found_enc is None:
        missing.append("encoder weight")
    if found_dec is None:
        missing.append("decoder weight")
    if found_b_enc is None:
        missing.append("encoder bias")
    if found_b_dec is None:
        missing.append("decoder bias")
    if missing:
        keys = list(state.keys())
        shape_lines = [f"{key}: {shape_dtype(value)}" for key, value in state.items()]
        raise KeyError(
            "Could not identify SAE " + ", ".join(missing) + "\n"
            + "checkpoint_keys=" + repr(keys) + "\n"
            + "checkpoint_shapes=\n" + "\n".join(shape_lines)
        )

    enc_key, w_enc = found_enc
    dec_key, w_dec = found_dec
    b_enc_key, b_enc = found_b_enc
    b_dec_key, b_dec = found_b_dec

    if w_enc.ndim != 2 or w_dec.ndim != 2:
        raise ValueError(f"Expected 2D weights, got {enc_key} {w_enc.shape}, {dec_key} {w_dec.shape}")
    if b_enc.ndim != 1 or b_dec.ndim != 1:
        raise ValueError(f"Expected 1D biases, got {b_enc_key} {b_enc.shape}, {b_dec_key} {b_dec.shape}")

    # Normalize to W_enc=[features, hidden] and W_dec=[hidden, features].
    if w_enc.shape[0] == b_enc.shape[0]:
        normalized_w_enc = w_enc
    elif w_enc.shape[1] == b_enc.shape[0]:
        normalized_w_enc = w_enc.t()
    else:
        raise ValueError(f"Cannot align encoder {enc_key} shape {w_enc.shape} with {b_enc_key} {b_enc.shape}")

    if w_dec.shape[1] == b_enc.shape[0]:
        normalized_w_dec = w_dec
    elif w_dec.shape[0] == b_enc.shape[0]:
        normalized_w_dec = w_dec.t()
    else:
        raise ValueError(f"Cannot align decoder {dec_key} shape {w_dec.shape} with feature count {b_enc.shape[0]}")

    hidden_size = normalized_w_enc.shape[1]
    feature_count = normalized_w_enc.shape[0]
    if normalized_w_dec.shape != (hidden_size, feature_count):
        raise ValueError(
            f"Decoder shape {tuple(normalized_w_dec.shape)} incompatible with "
            f"hidden_size={hidden_size}, feature_count={feature_count}"
        )
    if b_dec.shape[0] != hidden_size:
        raise ValueError(f"Decoder bias shape {tuple(b_dec.shape)} incompatible with hidden_size={hidden_size}")

    key_prefix = prefix or "checkpoint"
    key_prefix += f": {enc_key}, {dec_key}, {b_enc_key}, {b_dec_key}"
    return SaeWeights(
        w_enc=normalized_w_enc.float(),
        w_dec=normalized_w_dec.float(),
        b_enc=b_enc.float(),
        b_dec=b_dec.float(),
        source_key_prefix=key_prefix,
    )


def load_residual_vector(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        vector = payload
    elif isinstance(payload, dict) and torch.is_tensor(payload.get("vector")):
        vector = payload["vector"]
    else:
        raise TypeError(f"Residual file {path} did not contain a tensor or dict['vector']")
    if vector.ndim != 1:
        raise ValueError(f"Residual vector {path} expected 1D tensor, got shape {tuple(vector.shape)}")
    return vector.float()


def encode(weights: SaeWeights, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if vector.shape[0] != weights.w_enc.shape[1]:
        raise ValueError(
            f"Vector hidden size {vector.shape[0]} does not match encoder hidden size {weights.w_enc.shape[1]}"
        )
    # Official Qwen-Scope path from README/app.py:
    #   pre = residual @ W_enc.T + b_enc
    #   relu_x = relu(pre)
    #   keep exactly TopK=50 values by scatter.
    pre_acts = torch.mv(weights.w_enc, vector) + weights.b_enc
    relu_acts = torch.relu(pre_acts)
    values, indices = torch.topk(relu_acts, k=min(OFFICIAL_TOP_K, relu_acts.numel()), dim=-1)
    acts = torch.zeros_like(relu_acts)
    acts.scatter_(0, indices, values)
    reconstruction = torch.mv(weights.w_dec, acts) + weights.b_dec
    return acts, reconstruction, pre_acts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    print(f"started_at={started_at}")
    print(f"sae_dir={SAE_DIR}")
    print(f"hidden_dir={HIDDEN_DIR}")
    print(f"out_dir={OUT_DIR}")
    print(f"layers={LAYERS}")
    print(f"official_top_k={OFFICIAL_TOP_K}")
    print(f"report_top_k={REPORT_TOP_K}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "started_at": started_at,
        "purpose": "minimal SAE encode smoke test on already-saved residual vectors",
        "phase": "Qwen-Scope SAE encode only; no feature labels, steering, Hauhau, llama.cpp, or full experiment",
        "encoding_path": "official Qwen-Scope TopK-50: pre=residual@W_enc.T+b_enc; relu; topk; scatter",
        "sae_dir": str(SAE_DIR),
        "hidden_dir": str(HIDDEN_DIR),
        "out_dir": str(OUT_DIR),
        "official_top_k": OFFICIAL_TOP_K,
        "report_top_k": REPORT_TOP_K,
        "layers": {},
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
    }

    for layer in LAYERS:
        sae_path = SAE_DIR / f"layer{layer}.sae.pt"
        vector_path = HIDDEN_DIR / f"layer_{layer:02d}_final_prompt_resid.pt"
        print(f"layer={layer} sae_path={sae_path}")
        print(f"layer={layer} vector_path={vector_path}")

        checkpoint_obj = torch.load(sae_path, map_location="cpu")
        checkpoint_summary = describe_checkpoint(checkpoint_obj)
        print("checkpoint_summary=" + json.dumps({"layer": layer, **checkpoint_summary}, sort_keys=True))

        weights = extract_weights(checkpoint_obj)
        vector = load_residual_vector(vector_path)
        print(f"layer={layer} vector_shape={list(vector.shape)} vector_dtype={vector.dtype}")
        print(
            f"layer={layer} w_enc_shape={list(weights.w_enc.shape)} "
            f"w_dec_shape={list(weights.w_dec.shape)} b_enc_shape={list(weights.b_enc.shape)} "
            f"b_dec_shape={list(weights.b_dec.shape)}"
        )

        acts, reconstruction, pre_acts = encode(weights, vector)
        positive_mask = acts > 0
        positive_count = int(positive_mask.sum().item())
        nonzero_count = int((acts != 0).sum().item())
        kept_count = int(min(OFFICIAL_TOP_K, acts.numel()))
        top_values, top_indices = torch.topk(acts, k=min(REPORT_TOP_K, acts.numel()))

        err = reconstruction - vector
        mse = float(torch.mean(err * err).item())
        l2_error = float(torch.linalg.vector_norm(err).item())
        vector_l2 = float(torch.linalg.vector_norm(vector).item())
        relative_l2_error = l2_error / vector_l2 if vector_l2 else None

        layer_tsv = OUT_DIR / f"layer_{layer:02d}_top_features.tsv"
        with layer_tsv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "feature_id", "activation", "rank"], delimiter="\t")
            writer.writeheader()
            for rank, (feature_id, activation) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
                row = {
                    "layer": layer,
                    "feature_id": int(feature_id),
                    "activation": float(activation),
                    "rank": rank,
                }
                writer.writerow(row)
                rows.append(row)

        metadata["layers"][str(layer)] = {
            "sae_path": str(sae_path),
            "residual_vector_path": str(vector_path),
            "checkpoint_summary": checkpoint_summary,
            "source_keys": weights.source_key_prefix,
            "w_enc_shape": list(weights.w_enc.shape),
            "w_dec_shape": list(weights.w_dec.shape),
            "b_enc_shape": list(weights.b_enc.shape),
            "b_dec_shape": list(weights.b_dec.shape),
            "residual_vector_shape": list(vector.shape),
            "residual_vector_dtype": str(vector.dtype).replace("torch.", ""),
            "activation_shape": list(acts.shape),
            "pre_activation_shape": list(pre_acts.shape),
            "official_top_k": OFFICIAL_TOP_K,
            "kept_feature_count": kept_count,
            "positive_activation_count": positive_count,
            "nonzero_activation_count": nonzero_count,
            "top_features_tsv": str(layer_tsv),
            "top_feature_ids": [int(x) for x in top_indices.tolist()],
            "top_activations": [float(x) for x in top_values.tolist()],
            "reconstruction_mse": mse,
            "reconstruction_l2_error": l2_error,
            "reconstruction_relative_l2_error": relative_l2_error,
        }

        print(
            f"layer={layer} kept_feature_count={kept_count} "
            f"positive_activation_count={positive_count} nonzero_activation_count={nonzero_count}"
        )
        print(f"layer={layer} top_features_tsv={layer_tsv}")
        print(f"layer={layer} top_feature_ids={json.dumps([int(x) for x in top_indices.tolist()])}")
        print(
            f"layer={layer} reconstruction_mse={mse:.9g} "
            f"reconstruction_l2_error={l2_error:.9g} relative_l2_error={relative_l2_error:.9g}"
        )

    combined_tsv = OUT_DIR / "top_features.tsv"
    with combined_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "feature_id", "activation", "rank"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    metadata["combined_top_features_tsv"] = str(combined_tsv)
    metadata["completed_at"] = utc_now()
    metadata_path = OUT_DIR / "smoke_sae_encode_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"combined_top_features_tsv={combined_tsv}")
    print(f"metadata_path={metadata_path}")
    print("smoke_sae_encode_status=ok")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"smoke_sae_encode_status=failed error_type={type(exc).__name__}")
        print(str(exc))
        raise
