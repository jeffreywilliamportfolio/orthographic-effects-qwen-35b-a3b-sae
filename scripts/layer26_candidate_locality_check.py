#!/usr/bin/env python3
"""Layer-26 locality check for candidate Qwen-Scope SAE features."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/qwen-scope/5-11-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_PATH = ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50" / "layer26.sae.pt"
PROMPT_TSV_PATH = ROOT / "prompts" / "layer26_candidate_locality_prompts.tsv"
SCRIPT_PATH = ROOT / "scripts" / "layer26_candidate_locality_check.py"
OUT_DIR = ROOT / "sae_outputs" / "layer26_candidate_locality_check"
PROVENANCE_PATH = ROOT / "provenance" / "layer26_candidate_locality_check_20260511.txt"
OFFLOAD_DIR = ROOT / ".offload" / "layer26_candidate_locality_check"

TRACKED_FEATURE_IDS = [23977, 2722, 9745, 7108, 31784]
LAYER_INDEX = 26
TOP_K = 50
MAX_NEW_TOKENS = 24
CAPTURE_POSITIONS = [
    ("final_prompt_token", 0),
    ("final_prompt_token_minus_1", 1),
    ("final_prompt_token_minus_2", 2),
    ("final_prompt_token_minus_5", 5),
    ("final_prompt_token_minus_10", 10),
]
EXPECTED_ORIGINAL_FINAL_HITS = {
    "normal_hum_original": {23977, 2722, 9745, 7108},
    "dstroke_hum_original": {31784},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def load_prompt_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = ["prompt_id", "condition_family", "prompt_text", "notes"]
        if reader.fieldnames != expected:
            raise ValueError(f"Prompt TSV schema mismatch: {reader.fieldnames} != {expected}")
        rows = list(reader)
    if len(rows) != 10:
        raise ValueError(f"Expected 10 prompts, found {len(rows)}")
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt IDs are not unique")
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
            shapes = {name: list(value.shape) for name, value in sae.items() if torch.is_tensor(value)}
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


def capture_layer_sequence(model: torch.nn.Module, layers: torch.nn.ModuleList, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    buf: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
        hidden = out[0] if isinstance(out, tuple) else out
        buf["hidden"] = hidden[0].detach().to("cpu", dtype=torch.float32)

    handle = layers[LAYER_INDEX].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in buf:
        raise RuntimeError(f"Layer {LAYER_INDEX} hook did not capture hidden states")
    return buf["hidden"]


def condition_side(condition_family: str) -> str:
    if "dstroke" in condition_family:
        return "dstroke"
    if "normal" in condition_family:
        return "normal"
    return "other"


def write_summary(
    path: Path,
    tracked_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, str]],
    prompt_position_count: int,
) -> None:
    hits = [row for row in tracked_rows if row["appeared_in_topk50"] == "1"]
    final_hits_by_prompt: dict[str, set[int]] = defaultdict(set)
    all_hits_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in hits:
        feature_id = int(row["feature_id"])
        all_hits_by_feature[feature_id].append(row)
        if row["position_label"] == "final_prompt_token":
            final_hits_by_prompt[row["prompt_id"]].add(feature_id)

    lines = [
        "# Layer 26 Candidate Locality Summary",
        "",
        "Evidence-only summary from the 10-prompt locality check. No semantic labels are assigned here.",
        "",
        f"Prompt-position pairs scanned: {prompt_position_count}.",
        f"Tracked TopK-50 hit rows: {len(hits)}.",
        "",
        "## Do The Original Two Prompts Reproduce The Same Tracked Feature Hits?",
        "",
    ]
    for prompt_id, expected in EXPECTED_ORIGINAL_FINAL_HITS.items():
        observed = final_hits_by_prompt.get(prompt_id, set())
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if observed == expected:
            lines.append(f"- `{prompt_id}` reproduced the prior tracked final-token hit set: {sorted(observed)}.")
        else:
            lines.append(
                f"- `{prompt_id}` observed final-token hits {sorted(observed)}; "
                f"expected {sorted(expected)} from the prior two-prompt evidence; "
                f"missing {missing}; extra {extra}."
            )

    lines.extend(["", "## Do Tracked Features Appear At Nearby Boundary Positions?", ""])
    nearby_hits = [row for row in hits if row["position_label"] != "final_prompt_token"]
    if nearby_hits:
        counts = Counter(row["position_label"] for row in nearby_hits)
        lines.append(
            "- Tracked features appeared at nearby boundary positions: "
            + ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
            + "."
        )
    else:
        lines.append("- No tracked features appeared at nearby boundary positions in this locality check.")

    lines.extend(["", "## Do Tracked Features Appear Only In Exact Original Prompts?", ""])
    original_prompt_ids = set(EXPECTED_ORIGINAL_FINAL_HITS)
    variant_hits = [row for row in hits if row["prompt_id"] not in original_prompt_ids]
    if hits and not variant_hits:
        lines.append("- All tracked feature hits occurred in the exact original prompts.")
    elif variant_hits:
        variant_prompt_ids = sorted({row["prompt_id"] for row in variant_hits})
        lines.append("- Tracked features also appeared in near-neighbor variants: " + ", ".join(variant_prompt_ids) + ".")
    else:
        lines.append("- No tracked feature hits were observed, so exact-original specificity cannot be established from this run.")

    lines.extend(["", "## Do Tracked Features Appear In Near-Neighbor Variants?", ""])
    if variant_hits:
        by_variant = Counter(row["prompt_id"] for row in variant_hits)
        lines.append(
            "- Near-neighbor variant hit counts: "
            + ", ".join(f"{prompt_id}={count}" for prompt_id, count in sorted(by_variant.items()))
            + "."
        )
    else:
        lines.append("- No tracked feature hits appeared in near-neighbor variants.")

    lines.extend(["", "## Do Tracked Features Separate Normal From D-Stroke Consistently?", ""])
    prompt_family = {row["prompt_id"]: row["condition_family"] for row in prompt_rows}
    for feature_id in TRACKED_FEATURE_IDS:
        rows = all_hits_by_feature.get(feature_id, [])
        normal_rows = [row for row in rows if condition_side(prompt_family[row["prompt_id"]]) == "normal"]
        dstroke_rows = [row for row in rows if condition_side(prompt_family[row["prompt_id"]]) == "dstroke"]
        if normal_rows and not dstroke_rows:
            lines.append(f"- Feature {feature_id} appeared only in normal-family prompts in this locality check.")
        elif dstroke_rows and not normal_rows:
            lines.append(f"- Feature {feature_id} appeared only in d-stroke-family prompts in this locality check.")
        elif normal_rows and dstroke_rows:
            lines.append(f"- Feature {feature_id} appeared in both normal-family and d-stroke-family prompts in this locality check.")
        else:
            lines.append(f"- Feature {feature_id} did not appear in TopK-50 in this locality check.")

    lines.extend(["", "## Strongest Tracked Hits", ""])
    if hits:
        for row in sorted(hits, key=lambda item: float(item["activation"]), reverse=True)[:15]:
            lines.append(
                f"- {row['prompt_id']} {row['position_label']} feature {row['feature_id']}: "
                f"activation={float(row['activation']):.6g}, rank={row['rank']}."
            )
    else:
        lines.append("- none")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started_at = utc_now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    prompt_rows = load_prompt_rows(PROMPT_TSV_PATH)
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
    skipped_positions: list[dict[str, Any]] = []
    prompt_position_count = 0

    for prompt_idx, prompt in enumerate(prompt_rows, start=1):
        prompt_id = prompt["prompt_id"]
        condition_family = prompt["condition_family"]
        prompt_text = prompt["prompt_text"]
        encoded_cpu = tokenizer(prompt_text, return_tensors="pt")
        prompt_token_count = int(encoded_cpu["input_ids"].shape[1])
        final_index = prompt_token_count - 1
        encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}

        hidden_sequence = capture_layer_sequence(model, layers, encoded)
        if hidden_sequence.shape[0] != prompt_token_count:
            raise RuntimeError(
                f"{prompt_id} hidden sequence length {hidden_sequence.shape[0]} != prompt token count {prompt_token_count}"
            )

        prompt_positions: list[dict[str, Any]] = []
        for position_label, offset in CAPTURE_POSITIONS:
            token_position = final_index - offset
            if token_position < 0:
                skipped_positions.append({
                    "prompt_id": prompt_id,
                    "condition_family": condition_family,
                    "position_label": position_label,
                    "reason": "prompt too short",
                    "prompt_token_count": prompt_token_count,
                })
                continue

            token_id = int(encoded_cpu["input_ids"][0, token_position].item())
            token_string = clean_cell(tokenizer.decode([token_id]))
            vector = hidden_sequence[token_position, :]
            sparse, pre = encode_topk50(vector, sae)
            values, indices = torch.topk(sparse, k=min(TOP_K, sparse.numel()), dim=-1)
            feature_rank = {
                int(feature_id): rank
                for rank, feature_id in enumerate(indices.tolist(), start=1)
                if float(values[rank - 1].item()) > 0.0
            }
            feature_activation = {
                int(feature_id): float(activation)
                for feature_id, activation in zip(indices.tolist(), values.tolist())
            }
            prompt_position_count += 1

            for rank, (feature_id, activation) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                topk_rows.append({
                    "prompt_id": prompt_id,
                    "condition_family": condition_family,
                    "position_label": position_label,
                    "token_position": token_position,
                    "token_string": token_string,
                    "feature_id": int(feature_id),
                    "activation": float(activation),
                    "rank": rank,
                })

            position_hits = {}
            for feature_id in TRACKED_FEATURE_IDS:
                appeared = feature_id in feature_rank
                row = {
                    "prompt_id": prompt_id,
                    "condition_family": condition_family,
                    "position_label": position_label,
                    "token_position": token_position,
                    "token_string": token_string,
                    "feature_id": feature_id,
                    "appeared_in_topk50": "1" if appeared else "0",
                    "activation": feature_activation.get(feature_id, 0.0) if appeared else 0.0,
                    "rank": feature_rank.get(feature_id, ""),
                    "prompt_token_count": prompt_token_count,
                }
                tracked_rows.append(row)
                position_hits[str(feature_id)] = {
                    "appeared_in_topk50": appeared,
                    "activation": row["activation"],
                    "rank": row["rank"] if row["rank"] != "" else None,
                }

            prompt_positions.append({
                "position_label": position_label,
                "token_position": token_position,
                "token_string": token_string,
                "top_feature_ids": [int(x) for x in indices.tolist()],
                "top_feature_activations": [float(x) for x in values.tolist()],
                "tracked_hits": position_hits,
                "pre_activation_shape": list(pre.shape),
            })

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
        generated_rows.append({
            "prompt_id": prompt_id,
            "condition_family": condition_family,
            "prompt_text": prompt_text,
            "prompt_token_count": prompt_token_count,
            "generated_text_short": generated_text_short,
        })
        prompt_metadata.append({
            "prompt_id": prompt_id,
            "condition_family": condition_family,
            "prompt_token_count": prompt_token_count,
            "positions": prompt_positions,
            "generated_text_short": generated_text_short,
        })

        prompt_hit_count = sum(1 for row in tracked_rows if row["prompt_id"] == prompt_id and row["appeared_in_topk50"] == "1")
        print(
            f"prompt {prompt_idx:02d}/10 {prompt_id} tokens={prompt_token_count} "
            f"positions={len(prompt_positions)} tracked_hits={prompt_hit_count}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tracked_path = OUT_DIR / "tracked_feature_hits_by_position.tsv"
    topk_path = OUT_DIR / "topk_features_by_prompt_position.tsv"
    generated_path = OUT_DIR / "generated_text_by_prompt.tsv"
    metadata_path = OUT_DIR / "layer26_candidate_locality_metadata.json"
    summary_path = OUT_DIR / "layer26_candidate_locality_summary.md"

    with tracked_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "condition_family",
            "position_label",
            "token_position",
            "token_string",
            "feature_id",
            "appeared_in_topk50",
            "activation",
            "rank",
            "prompt_token_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tracked_rows)

    with topk_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "condition_family",
            "position_label",
            "token_position",
            "token_string",
            "feature_id",
            "activation",
            "rank",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(topk_rows)

    with generated_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "condition_family",
            "prompt_text",
            "prompt_token_count",
            "generated_text_short",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(generated_rows)

    write_summary(summary_path, tracked_rows, prompt_rows, prompt_position_count)
    completed_at = utc_now()
    metadata = {
        "started_at": started_at,
        "completed_at": completed_at,
        "purpose": "layer-26 candidate feature locality and reproducibility check",
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
        "prompt_tsv_path": str(PROMPT_TSV_PATH),
        "script_path": str(SCRIPT_PATH),
        "output_dir": str(OUT_DIR),
        "tracked_feature_ids": TRACKED_FEATURE_IDS,
        "layer_index": LAYER_INDEX,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "capture_positions": [{"position_label": label, "offset_from_final_prompt_token": offset} for label, offset in CAPTURE_POSITIONS],
        "prompt_count": len(prompt_rows),
        "prompt_position_count": prompt_position_count,
        "skipped_positions": skipped_positions,
        "selected_layer_hooks_used": True,
        "hidden_state_capture_method": "single forward hook on model.model.layers[26] per prompt; no output_hidden_states=True request",
        "encoding_path": "official Qwen-Scope TopK-50: pre=hidden@W_enc.T+b_enc; relu; topk; scatter",
        "expected_original_final_hits_from_prior_two_prompt_evidence": {
            key: sorted(value) for key, value in EXPECTED_ORIGINAL_FINAL_HITS.items()
        },
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
            "tracked_feature_hits_by_position": str(tracked_path),
            "topk_features_by_prompt_position": str(topk_path),
            "generated_text_by_prompt": str(generated_path),
            "metadata": str(metadata_path),
            "summary": str(summary_path),
            "provenance": str(PROVENANCE_PATH),
        },
        "prompts": prompt_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance_lines = [
        f"timestamp={completed_at}",
        f"prompt_tsv_path={PROMPT_TSV_PATH}",
        f"script_path={SCRIPT_PATH}",
        "tracked_feature_ids=" + ",".join(str(x) for x in TRACKED_FEATURE_IDS),
        "capture_positions=" + ",".join(label for label, _offset in CAPTURE_POSITIONS),
        f"model_path={MODEL_PATH}",
        f"sae_path={SAE_PATH}",
        f"outputs_written={tracked_path},{topk_path},{generated_path},{metadata_path},{summary_path}",
        f"prompt_count={len(prompt_rows)}",
        f"prompt_position_count={prompt_position_count}",
        "selected_layer_hooks_used=true; hook_path=model.model.layers[26]; output_hidden_states_not_requested=true",
        "confirmation=no steering, no Hauhau, no llama.cpp, no full experiment, and no semantic labels were used",
    ]
    PROVENANCE_PATH.write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

    print(f"tracked_feature_hits_by_position={tracked_path}")
    print(f"topk_features_by_prompt_position={topk_path}")
    print(f"generated_text_by_prompt={generated_path}")
    print(f"metadata={metadata_path}")
    print(f"summary={summary_path}")
    print(f"provenance={PROVENANCE_PATH}")
    print("layer26_candidate_locality_check_status=ok")


if __name__ == "__main__":
    main()
