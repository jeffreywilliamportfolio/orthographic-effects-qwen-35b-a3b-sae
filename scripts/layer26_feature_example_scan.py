#!/usr/bin/env python3
"""Layer-26 high-activation example scan for selected Qwen-Scope features."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/qwen-scope/5-11-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_PATH = ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50" / "layer26.sae.pt"
SEED_BANK_PATH = ROOT / "prompts" / "layer26_feature_seed_bank.tsv"
SCRIPT_PATH = ROOT / "scripts" / "layer26_feature_example_scan.py"
OUT_DIR = ROOT / "sae_outputs" / "layer26_feature_example_scan"
PROVENANCE_PATH = ROOT / "provenance" / "layer26_feature_example_scan_20260511.txt"
OFFLOAD_DIR = ROOT / ".offload" / "layer26_feature_example_scan"
TRACKED_FEATURE_IDS = [23977, 2722, 9745, 7108, 31784]
LAYER_INDEX = 26
TOP_K = 50
MAX_NEW_TOKENS = 24


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def load_seed_bank(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = ["prompt_id", "category", "prompt_text"]
        if reader.fieldnames != expected:
            raise ValueError(f"Seed bank schema mismatch: {reader.fieldnames} != {expected}")
        rows = list(reader)
    if len(rows) != 30:
        raise ValueError(f"Expected 30 seed prompts, found {len(rows)}")
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Seed bank prompt_id values are not unique")
    return rows


def load_sae(path: Path) -> dict[str, Any]:
    try:
        sae = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sae = torch.load(path, map_location="cpu")
    if not isinstance(sae, dict):
        raise TypeError(f"Expected SAE checkpoint dict at {path}, got {type(sae).__name__}")
    for key in ("W_enc", "b_enc"):
        if key not in sae or not torch.is_tensor(sae[key]):
            shapes = {
                name: list(value.shape)
                for name, value in sae.items()
                if torch.is_tensor(value)
            }
            raise KeyError(f"SAE missing tensor key {key}; tensor_shapes={shapes}")
    w_enc = sae["W_enc"]
    b_enc = sae["b_enc"]
    if w_enc.ndim != 2 or b_enc.ndim != 1:
        raise ValueError(f"Unexpected SAE shapes: W_enc={tuple(w_enc.shape)}, b_enc={tuple(b_enc.shape)}")
    if w_enc.shape[0] == b_enc.shape[0]:
        w_enc_t = w_enc.T.to(dtype=torch.float32).contiguous()
    elif w_enc.shape[1] == b_enc.shape[0]:
        w_enc_t = w_enc.to(dtype=torch.float32).contiguous()
    else:
        raise ValueError(f"Cannot align W_enc={tuple(w_enc.shape)} with b_enc={tuple(b_enc.shape)}")
    return {
        "W_enc_source_shape": list(w_enc.shape),
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


def encode_topk50(vector: torch.Tensor, sae: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
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
    return sparse, pre


def capture_layer_vector(model: torch.nn.Module, layers: torch.nn.ModuleList, encoded: dict[str, torch.Tensor], final_index: int) -> torch.Tensor:
    buf: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
        hidden = out[0] if isinstance(out, tuple) else out
        buf["vector"] = hidden[0, final_index, :].detach().to("cpu", dtype=torch.float32)

    handle = layers[LAYER_INDEX].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if "vector" not in buf:
        raise RuntimeError(f"Layer {LAYER_INDEX} hook did not capture a vector")
    return buf["vector"]


def summarize_evidence(rows: list[dict[str, Any]], seed_rows: list[dict[str, str]], path: Path) -> None:
    category_by_prompt = {row["prompt_id"]: row["category"] for row in seed_rows}
    all_categories = sorted({row["category"] for row in seed_rows})
    lines = [
        "# Layer 26 Feature Evidence Summary",
        "",
        "Evidence-only summary from the 30-prompt seed scan. No semantic labels are assigned here.",
        "",
    ]
    by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[int(row["feature_id"])].append(row)

    for feature_id in TRACKED_FEATURE_IDS:
        feature_rows = by_feature[feature_id]
        hits = [row for row in feature_rows if row["appeared_in_topk50"] == "1"]
        hit_categories = sorted({row["category"] for row in hits})
        miss_categories = [category for category in all_categories if category not in hit_categories]
        strongest = sorted(hits, key=lambda row: float(row["activation"]), reverse=True)[:5]
        normal_hits = [row for row in hits if row["category"] != "Unicode d-stroke perturbation"]
        dstroke_hits = [row for row in hits if row["category"] == "Unicode d-stroke perturbation"]

        if hits and len(dstroke_hits) == len(hits):
            scope_sentence = f"Feature {feature_id} appeared only in d-stroke perturbation prompts in this seed scan."
        elif hits and dstroke_hits and normal_hits:
            scope_sentence = f"Feature {feature_id} appeared in d-stroke perturbation prompts and also in other categories in this seed scan."
        elif hits and normal_hits and not dstroke_hits:
            scope_sentence = f"Feature {feature_id} appeared in normal-prompt categories and did not appear in d-stroke perturbation prompts in this seed scan."
        else:
            scope_sentence = f"Feature {feature_id} did not appear in TopK-50 for these seed prompts."

        lines.extend([
            f"## Feature {feature_id}",
            "",
            scope_sentence,
            "",
            "Appeared in prompt categories: " + (", ".join(hit_categories) if hit_categories else "none") + ".",
            "Did not appear in prompt categories: " + (", ".join(miss_categories) if miss_categories else "none") + ".",
            "",
            "Strongest activation examples:",
        ])
        if strongest:
            for row in strongest:
                lines.append(
                    f"- {row['prompt_id']} ({row['category']}): activation={float(row['activation']):.6g}, rank={row['rank']}"
                )
        else:
            lines.append("- none")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started_at = utc_now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    seed_rows = load_seed_bank(SEED_BANK_PATH)
    sae = load_sae(SAE_PATH)

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
    layers = decoder_layers(model)
    if LAYER_INDEX < 0 or LAYER_INDEX >= len(layers):
        raise ValueError(f"Layer index {LAYER_INDEX} outside 0..{len(layers) - 1}")

    input_device = model.get_input_embeddings().weight.device
    tracked_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    prompt_metadata: list[dict[str, Any]] = []

    for idx, seed in enumerate(seed_rows, start=1):
        prompt_id = seed["prompt_id"]
        category = seed["category"]
        prompt_text = seed["prompt_text"]
        encoded_cpu = tokenizer(prompt_text, return_tensors="pt")
        prompt_token_count = int(encoded_cpu["input_ids"].shape[1])
        final_index = prompt_token_count - 1
        final_token_id = int(encoded_cpu["input_ids"][0, final_index].item())
        final_prompt_token_string = clean_cell(tokenizer.decode([final_token_id]))
        encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}

        vector = capture_layer_vector(model, layers, encoded, final_index)
        sparse, pre = encode_topk50(vector, sae)
        values, indices = torch.topk(sparse, k=min(TOP_K, sparse.numel()), dim=-1)
        feature_rank = {int(fid): rank for rank, fid in enumerate(indices.tolist(), start=1)}
        feature_activation = {int(fid): float(val) for fid, val in zip(indices.tolist(), values.tolist())}

        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_token_ids = generated_ids[0, prompt_token_count:]
        generated_text_short = clean_cell(tokenizer.decode(new_token_ids, skip_special_tokens=True))

        for rank, (feature_id, activation) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            topk_rows.append({
                "prompt_id": prompt_id,
                "category": category,
                "feature_id": int(feature_id),
                "activation": float(activation),
                "rank": rank,
            })

        for feature_id in TRACKED_FEATURE_IDS:
            appeared = feature_id in feature_rank
            tracked_rows.append({
                "prompt_id": prompt_id,
                "category": category,
                "feature_id": feature_id,
                "appeared_in_topk50": "1" if appeared else "0",
                "activation": feature_activation.get(feature_id, 0.0),
                "rank": feature_rank.get(feature_id, ""),
                "prompt_token_count": prompt_token_count,
                "final_prompt_token_string": final_prompt_token_string,
                "generated_text_short": generated_text_short,
            })

        generated_rows.append({
            "prompt_id": prompt_id,
            "category": category,
            "prompt_text": prompt_text,
            "prompt_token_count": prompt_token_count,
            "final_prompt_token_string": final_prompt_token_string,
            "generated_text_short": generated_text_short,
        })
        prompt_metadata.append({
            "prompt_id": prompt_id,
            "category": category,
            "prompt_token_count": prompt_token_count,
            "final_prompt_token_string": final_prompt_token_string,
            "top_feature_ids": [int(x) for x in indices[:TOP_K].tolist()],
            "top_feature_activations": [float(x) for x in values[:TOP_K].tolist()],
            "tracked_hits": {
                str(feature_id): {
                    "appeared_in_topk50": feature_id in feature_rank,
                    "activation": feature_activation.get(feature_id, 0.0),
                    "rank": feature_rank.get(feature_id),
                }
                for feature_id in TRACKED_FEATURE_IDS
            },
            "residual_vector_shape": list(vector.shape),
            "pre_activation_shape": list(pre.shape),
        })
        print(
            f"prompt {idx:02d}/30 {prompt_id} category={category!r} "
            f"tracked_hits={sum(1 for feature_id in TRACKED_FEATURE_IDS if feature_id in feature_rank)}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tracked_path = OUT_DIR / "tracked_feature_hits.tsv"
    topk_path = OUT_DIR / "topk_features_by_prompt.tsv"
    generated_path = OUT_DIR / "generated_text_by_prompt.tsv"
    metadata_path = OUT_DIR / "layer26_feature_example_scan_metadata.json"
    summary_path = OUT_DIR / "layer26_feature_evidence_summary.md"

    with tracked_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "category",
            "feature_id",
            "appeared_in_topk50",
            "activation",
            "rank",
            "prompt_token_count",
            "final_prompt_token_string",
            "generated_text_short",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tracked_rows)

    with topk_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["prompt_id", "category", "feature_id", "activation", "rank"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(topk_rows)

    with generated_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "category",
            "prompt_text",
            "prompt_token_count",
            "final_prompt_token_string",
            "generated_text_short",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(generated_rows)

    summarize_evidence(tracked_rows, seed_rows, summary_path)
    completed_at = utc_now()

    metadata = {
        "started_at": started_at,
        "completed_at": completed_at,
        "purpose": "high-activation example collection for priority layer-26 Qwen-Scope features",
        "phase": "Transformers/PyTorch residual-stream capture plus Qwen-Scope SAE TopK-50 encoding",
        "restrictions": {
            "semantic_labels_assigned": False,
            "steering_used": False,
            "hauhau_used": False,
            "llama_cpp_used": False,
            "full_experiment_run": False,
        },
        "model_path": str(MODEL_PATH),
        "sae_path": str(SAE_PATH),
        "seed_bank_path": str(SEED_BANK_PATH),
        "script_path": str(SCRIPT_PATH),
        "output_dir": str(OUT_DIR),
        "tracked_feature_ids": TRACKED_FEATURE_IDS,
        "layer_index": LAYER_INDEX,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_count": len(seed_rows),
        "selected_layer_hooks_used": True,
        "hidden_state_capture_method": "forward hook on model.model.layers[26]; no output_hidden_states=True request",
        "encoding_path": "official Qwen-Scope TopK-50: pre=hidden@W_enc.T+b_enc; relu; topk; scatter",
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
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
            "decoder_layer_count": len(layers),
        },
        "sae": {
            "W_enc_source_shape": sae["W_enc_source_shape"],
            "b_enc_shape": sae["b_enc_shape"],
            "W_enc_transposed_shape": list(sae["_W_enc"].shape),
        },
        "outputs_written": {
            "tracked_feature_hits": str(tracked_path),
            "topk_features_by_prompt": str(topk_path),
            "generated_text_by_prompt": str(generated_path),
            "metadata": str(metadata_path),
            "evidence_summary": str(summary_path),
            "provenance": str(PROVENANCE_PATH),
        },
        "prompts": prompt_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance_lines = [
        f"timestamp={completed_at}",
        f"seed_bank_path={SEED_BANK_PATH}",
        f"script_path={SCRIPT_PATH}",
        "tracked_feature_ids=" + ",".join(str(x) for x in TRACKED_FEATURE_IDS),
        f"model_path={MODEL_PATH}",
        f"sae_path={SAE_PATH}",
        f"outputs_written={tracked_path},{topk_path},{generated_path},{metadata_path},{summary_path}",
        f"prompt_count={len(seed_rows)}",
        "selected_layer_hooks_used=true; hook_path=model.model.layers[26]; output_hidden_states_not_requested=true",
        "confirmation=no steering, no Hauhau, no llama.cpp, no full experiment, and no semantic labels were used",
    ]
    PROVENANCE_PATH.write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

    print(f"tracked_feature_hits={tracked_path}")
    print(f"topk_features_by_prompt={topk_path}")
    print(f"generated_text_by_prompt={generated_path}")
    print(f"metadata={metadata_path}")
    print(f"evidence_summary={summary_path}")
    print(f"provenance={PROVENANCE_PATH}")
    print("layer26_feature_example_scan_status=ok")


if __name__ == "__main__":
    main()
