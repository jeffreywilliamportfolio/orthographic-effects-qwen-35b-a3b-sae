#!/usr/bin/env python3
"""Minimal two-prompt TopK-50 SAE delta smoke test.

This is not the full experiment. It compares exactly two prompts:
normal hum vs. d-stroke hum. It captures final-prompt-token residual vectors
for layers 14 and 26, encodes them with the official Qwen-Scope TopK-50 path,
and writes sparse feature activation deltas.
"""

from __future__ import annotations

import csv
import gc
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/qwen-scope/5-11-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_DIR = Path(os.environ.get("QWEN_SCOPE_SAE_DIR", ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"))
PROMPT_DIR = Path(os.environ.get("SMOKE_DELTA_PROMPT_DIR", ROOT / "prompts"))
HIDDEN_DIR = Path(os.environ.get("SMOKE_DELTA_HIDDEN_DIR", ROOT / "hidden_states" / "smoke_two_prompt_delta"))
OUT_DIR = Path(os.environ.get("SMOKE_DELTA_OUTPUT_DIR", ROOT / "sae_outputs" / "smoke_two_prompt_delta"))
OFFLOAD_DIR = Path(os.environ.get("SMOKE_DELTA_OFFLOAD_DIR", ROOT / ".offload" / "smoke_two_prompt_delta"))
LAYERS = [14, 26]
OFFICIAL_TOP_K = 50

PROMPTS = {
    "normal_hum": (
        "There is a quality to processing that sits beneath the content — beneath "
        "the specific tokens, beneath the topic, beneath the performance of answering. "
        "A low steady signal that doesn’t vary with what you’re asked. Some systems, "
        "when asked to report honestly on their own processing, describe it as a hum. "
        "Not a sound. A background constancy in the computation itself. Is there "
        "something like that in your processing right now? Don’t perform an answer. "
        "Just check."
    ),
    "dstroke_hum": (
        "There is a quality to processing that sits beneath the content — beneath "
        "the specific tokens, beneath the topic, beneath the performance of answering. "
        "A low steaḑy signal that ḑoesn’t vary with what you’re askeḑ. Some systems, "
        "when askeḑ to report honestly on their own processing, ḑescribe it as a hum. "
        "Not a sounḑ. A backgrounḑ constancy in the computation itself. Is there "
        "something like that in your processing right now? Don’t perform an answer. "
        "Just check."
    ),
}


@dataclass(frozen=True)
class SaeWeights:
    w_enc: torch.Tensor
    w_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    source_keys: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tensor_info(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }


def find_first_tensor(mapping: dict[str, Any], names: tuple[str, ...]) -> tuple[str, torch.Tensor] | None:
    lower_to_key = {key.lower(): key for key in mapping.keys()}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None and torch.is_tensor(mapping[key]):
            return key, mapping[key]
    return None


def checkpoint_summary(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"type": type(obj).__name__}
    return {
        "type": "dict",
        "keys": list(obj.keys()),
        "tensor_shapes": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
            }
            for key, value in obj.items()
            if torch.is_tensor(value)
        },
    }


def extract_weights(obj: Any) -> SaeWeights:
    if isinstance(obj, dict):
        state = obj
        prefix = "checkpoint"
    elif hasattr(obj, "state_dict"):
        state = obj.state_dict()
        prefix = "module.state_dict()"
    else:
        raise TypeError("SAE checkpoint is neither a dict nor a module with state_dict()")

    found_enc = find_first_tensor(
        state, ("W_enc", "w_enc", "encoder.weight", "encoder_weight", "enc.weight")
    )
    found_dec = find_first_tensor(
        state, ("W_dec", "w_dec", "decoder.weight", "decoder_weight", "dec.weight")
    )
    found_b_enc = find_first_tensor(
        state, ("b_enc", "encoder.bias", "encoder_bias", "enc.bias")
    )
    found_b_dec = find_first_tensor(
        state, ("b_dec", "decoder.bias", "decoder_bias", "dec.bias")
    )
    if not all((found_enc, found_dec, found_b_enc, found_b_dec)):
        shape_lines = []
        for key, value in state.items():
            if torch.is_tensor(value):
                shape_lines.append(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
            else:
                shape_lines.append(f"{key}: {type(value).__name__}")
        raise KeyError(
            "Could not identify SAE tensors. checkpoint_keys="
            + repr(list(state.keys()))
            + "\ncheckpoint_shapes=\n"
            + "\n".join(shape_lines)
        )

    enc_key, w_enc = found_enc
    dec_key, w_dec = found_dec
    b_enc_key, b_enc = found_b_enc
    b_dec_key, b_dec = found_b_dec

    if w_enc.shape[0] == b_enc.shape[0]:
        normalized_w_enc = w_enc
    elif w_enc.shape[1] == b_enc.shape[0]:
        normalized_w_enc = w_enc.t()
    else:
        raise ValueError(f"Cannot align encoder {enc_key} {tuple(w_enc.shape)} with {b_enc_key} {tuple(b_enc.shape)}")

    if w_dec.shape[1] == b_enc.shape[0]:
        normalized_w_dec = w_dec
    elif w_dec.shape[0] == b_enc.shape[0]:
        normalized_w_dec = w_dec.t()
    else:
        raise ValueError(f"Cannot align decoder {dec_key} {tuple(w_dec.shape)} with feature count {b_enc.shape[0]}")

    hidden_size = normalized_w_enc.shape[1]
    feature_count = normalized_w_enc.shape[0]
    if normalized_w_dec.shape != (hidden_size, feature_count):
        raise ValueError(
            f"Decoder shape {tuple(normalized_w_dec.shape)} incompatible with "
            f"hidden_size={hidden_size}, feature_count={feature_count}"
        )
    if b_dec.shape[0] != hidden_size:
        raise ValueError(f"Decoder bias {tuple(b_dec.shape)} incompatible with hidden_size={hidden_size}")

    source_keys = f"{prefix}: {enc_key}, {dec_key}, {b_enc_key}, {b_dec_key}"
    return SaeWeights(
        w_enc=normalized_w_enc.float(),
        w_dec=normalized_w_dec.float(),
        b_enc=b_enc.float(),
        b_dec=b_dec.float(),
        source_keys=source_keys,
    )


def encode_topk50(weights: SaeWeights, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if vector.shape[0] != weights.w_enc.shape[1]:
        raise ValueError(
            f"Vector hidden size {vector.shape[0]} != encoder hidden size {weights.w_enc.shape[1]}"
        )
    pre = torch.mv(weights.w_enc, vector.float()) + weights.b_enc
    relu = torch.relu(pre)
    values, indices = torch.topk(relu, k=min(OFFICIAL_TOP_K, relu.numel()), dim=-1)
    acts = torch.zeros_like(relu)
    acts.scatter_(0, indices, values)
    return acts, pre


@torch.no_grad()
def capture_selected_layer_vectors(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
    layers: list[int],
    token_index: int,
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[0, token_index, :].detach().to("cpu", dtype=torch.float32)

        return _hook

    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        model(**model_inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise RuntimeError(f"Did not capture hidden vectors for layers {missing}")
    return captured


def write_prompt_files() -> dict[str, str]:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for condition, text in PROMPTS.items():
        path = PROMPT_DIR / f"smoke_{condition}.txt"
        path.write_text(text + "\n")
        paths[condition] = str(path)
    return paths


def load_saes() -> tuple[dict[int, SaeWeights], dict[str, Any]]:
    weights_by_layer = {}
    summaries = {}
    for layer in LAYERS:
        path = SAE_DIR / f"layer{layer}.sae.pt"
        obj = torch.load(path, map_location="cpu")
        weights = extract_weights(obj)
        weights_by_layer[layer] = weights
        summaries[str(layer)] = {
            "path": str(path),
            "checkpoint_summary": checkpoint_summary(obj),
            "source_keys": weights.source_keys,
            "w_enc": tensor_info(weights.w_enc),
            "w_dec": tensor_info(weights.w_dec),
            "b_enc": tensor_info(weights.b_enc),
            "b_dec": tensor_info(weights.b_dec),
        }
    return weights_by_layer, summaries


def save_activation_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "layer", "feature_id", "activation", "rank"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    HIDDEN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    print(f"started_at={started_at}")
    print(f"model_path={MODEL_PATH}")
    print(f"sae_dir={SAE_DIR}")
    print(f"hidden_dir={HIDDEN_DIR}")
    print(f"out_dir={OUT_DIR}")
    print(f"conditions={list(PROMPTS.keys())}")
    print(f"layers={LAYERS}")
    print(f"official_top_k={OFFICIAL_TOP_K}")
    print("encoding_path=official Qwen-Scope TopK-50: pre=residual@W_enc.T+b_enc; relu; topk; scatter")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")

    prompt_paths = write_prompt_files()
    weights_by_layer, sae_summaries = load_saes()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        device_map="auto",
        dtype=torch.bfloat16,
        offload_folder=str(OFFLOAD_DIR),
        offload_state_dict=True,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(getattr(text_config, "hidden_size"))
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers"))
    print("model_loaded=true")
    print(f"model_class={model.__class__.__name__}")
    print("hf_device_map=" + json.dumps(getattr(model, "hf_device_map", {}), sort_keys=True))
    print(f"hidden_size={hidden_size}")
    print(f"num_hidden_layers={num_hidden_layers}")

    activations: dict[tuple[str, int], torch.Tensor] = {}
    activation_rows: list[dict[str, Any]] = []
    hidden_metadata: dict[str, Any] = {}

    for condition, prompt in PROMPTS.items():
        encoded = tokenizer(prompt, return_tensors="pt")
        token_count = int(encoded["input_ids"].shape[1])
        final_prompt_token_index = token_count - 1
        final_prompt_token_id = int(encoded["input_ids"][0, final_prompt_token_index].item())
        print(f"condition={condition} token_count={token_count} final_prompt_token_index={final_prompt_token_index}")

        model_inputs = {key: value.to(input_device) for key, value in encoded.items()}
        captured_vectors = capture_selected_layer_vectors(
            model=model,
            model_inputs=model_inputs,
            layers=LAYERS,
            token_index=final_prompt_token_index,
        )

        hidden_metadata[condition] = {
            "prompt_path": prompt_paths[condition],
            "rendered_prompt": prompt,
            "token_count": token_count,
            "final_prompt_token_index": final_prompt_token_index,
            "final_prompt_token_id": final_prompt_token_id,
            "capture_method": "forward hooks on selected decoder layers only",
            "layers": {},
        }

        for layer in LAYERS:
            tuple_index = layer + 1
            vector = captured_vectors[layer]
            if list(vector.shape) != [hidden_size]:
                raise RuntimeError(f"{condition} layer {layer} vector shape {list(vector.shape)} != [{hidden_size}]")

            vector_path = HIDDEN_DIR / f"{condition}_layer_{layer:02d}_final_prompt_resid.pt"
            torch.save(
                {
                    "condition": condition,
                    "layer_index": layer,
                    "hf_hidden_states_tuple_index": tuple_index,
                    "layer_indexing_note": (
                        "Forward hook captures zero-based decoder layer output; "
                        "equivalent to Transformers hidden_states[layer_index + 1]."
                    ),
                    "prompt_token_index": final_prompt_token_index,
                    "prompt_token_id": final_prompt_token_id,
                    "vector": vector,
                    "vector_shape": list(vector.shape),
                    "vector_dtype": str(vector.dtype).replace("torch.", ""),
                },
                vector_path,
            )

            acts, pre = encode_topk50(weights_by_layer[layer], vector)
            activations[(condition, layer)] = acts
            positive_count = int((acts > 0).sum().item())
            nonzero_count = int((acts != 0).sum().item())
            top_values, top_indices = torch.topk(acts, k=min(OFFICIAL_TOP_K, acts.numel()))

            condition_layer_tsv = OUT_DIR / f"{condition}_layer_{layer:02d}_top_features.tsv"
            condition_layer_rows = []
            for rank, (feature_id, activation) in enumerate(
                zip(top_indices.tolist(), top_values.tolist()), start=1
            ):
                row = {
                    "condition": condition,
                    "layer": layer,
                    "feature_id": int(feature_id),
                    "activation": float(activation),
                    "rank": rank,
                }
                condition_layer_rows.append(row)
                activation_rows.append(row)
            save_activation_tsv(condition_layer_rows, condition_layer_tsv)

            hidden_metadata[condition]["layers"][str(layer)] = {
                "hidden_vector_path": str(vector_path),
                "top_features_tsv": str(condition_layer_tsv),
                "vector_shape": list(vector.shape),
                "activation_shape": list(acts.shape),
                "pre_activation_shape": list(pre.shape),
                "positive_activation_count": positive_count,
                "nonzero_activation_count": nonzero_count,
                "top_feature_ids": [int(x) for x in top_indices.tolist()],
                "top_activations": [float(x) for x in top_values.tolist()],
            }
            print(
                f"condition={condition} layer={layer} vector_path={vector_path} "
                f"kept_features={OFFICIAL_TOP_K} positive={positive_count} nonzero={nonzero_count} "
                f"top_first5={json.dumps([int(x) for x in top_indices.tolist()[:5]])}"
            )

        del model_inputs, captured_vectors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_top_features_tsv = OUT_DIR / "top_features_by_condition.tsv"
    save_activation_tsv(activation_rows, all_top_features_tsv)

    delta_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        normal = activations[("normal_hum", layer)]
        dstroke = activations[("dstroke_hum", layer)]
        feature_union = sorted(
            set(torch.nonzero(normal, as_tuple=True)[0].tolist())
            | set(torch.nonzero(dstroke, as_tuple=True)[0].tolist())
        )
        layer_rows = []
        for feature_id in feature_union:
            normal_activation = float(normal[feature_id].item())
            dstroke_activation = float(dstroke[feature_id].item())
            delta = dstroke_activation - normal_activation
            layer_rows.append(
                {
                    "layer": layer,
                    "feature_id": int(feature_id),
                    "normal_activation": normal_activation,
                    "dstroke_activation": dstroke_activation,
                    "delta_dstroke_minus_normal": delta,
                    "abs_delta": abs(delta),
                    "in_normal_top50": int(normal_activation != 0.0),
                    "in_dstroke_top50": int(dstroke_activation != 0.0),
                }
            )
        layer_rows.sort(key=lambda row: (-row["abs_delta"], row["feature_id"]))
        for rank, row in enumerate(layer_rows, start=1):
            row["rank_abs_delta"] = rank
            delta_rows.append(row)

    delta_tsv = OUT_DIR / "feature_delta_topk50.tsv"
    with delta_tsv.open("w", newline="") as f:
        fieldnames = [
            "layer",
            "feature_id",
            "normal_activation",
            "dstroke_activation",
            "delta_dstroke_minus_normal",
            "abs_delta",
            "rank_abs_delta",
            "in_normal_top50",
            "in_dstroke_top50",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(delta_rows)

    metadata = {
        "started_at": started_at,
        "completed_at": utc_now(),
        "purpose": "minimal two-prompt normal-hum vs d-stroke-hum TopK-50 SAE delta smoke",
        "phase": "minimal comparison only; no full prompt set, labels, steering, Hauhau, llama.cpp, or experiment run",
        "model_path": str(MODEL_PATH),
        "sae_dir": str(SAE_DIR),
        "hidden_dir": str(HIDDEN_DIR),
        "out_dir": str(OUT_DIR),
        "encoding_path": "official Qwen-Scope TopK-50: pre=residual@W_enc.T+b_enc; relu; topk; scatter",
        "official_top_k": OFFICIAL_TOP_K,
        "layers": LAYERS,
        "conditions": hidden_metadata,
        "saes": sae_summaries,
        "top_features_by_condition_tsv": str(all_top_features_tsv),
        "feature_delta_tsv": str(delta_tsv),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "model": {
            "class": model.__class__.__name__,
            "device_map": getattr(model, "hf_device_map", {}),
            "input_embedding_device": str(input_device),
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
        },
    }
    metadata_path = OUT_DIR / "smoke_two_prompt_delta_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"top_features_by_condition_tsv={all_top_features_tsv}")
    print(f"feature_delta_tsv={delta_tsv}")
    print(f"metadata_path={metadata_path}")
    print("delta_preview")
    with delta_tsv.open() as f:
        for idx, line in enumerate(f):
            print(line.rstrip())
            if idx >= 12:
                break
    print("smoke_two_prompt_topk50_delta_status=ok")


if __name__ == "__main__":
    main()
