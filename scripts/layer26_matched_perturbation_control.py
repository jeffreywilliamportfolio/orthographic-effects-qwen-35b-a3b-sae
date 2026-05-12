#!/usr/bin/env python3
"""Layer-26 matched perturbation control for Qwen-Scope candidate features."""

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
PROMPT_TSV_PATH = ROOT / "prompts" / "layer26_matched_perturbation_prompts.tsv"
SCRIPT_PATH = ROOT / "scripts" / "layer26_matched_perturbation_control.py"
OUT_DIR = ROOT / "sae_outputs" / "layer26_matched_perturbation_control"
PROVENANCE_PATH = ROOT / "provenance" / "layer26_matched_perturbation_control_20260511.txt"
OFFLOAD_DIR = ROOT / ".offload" / "layer26_matched_perturbation_control"

TRACKED_FEATURE_IDS = [2722, 7108, 31784, 23977, 9745]
PERTURBATION_TYPES = ["ascii_original", "d_to_ḑ", "e_to_ē", "s_to_ş", "s_to_ṡ"]
BASE_PROMPT_FAMILIES = ["original_hum", "just_check_hum"]
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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def load_prompt_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = ["prompt_id", "base_prompt_family", "perturbation_type", "prompt_text", "notes"]
        if reader.fieldnames != expected:
            raise ValueError(f"Prompt TSV schema mismatch: {reader.fieldnames} != {expected}")
        rows = list(reader)
    if len(rows) != 10:
        raise ValueError(f"Expected 10 prompts, found {len(rows)}")
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt IDs are not unique")
    observed_pairs = {(row["base_prompt_family"], row["perturbation_type"]) for row in rows}
    expected_pairs = {(family, perturb) for family in BASE_PROMPT_FAMILIES for perturb in PERTURBATION_TYPES}
    if observed_pairs != expected_pairs:
        raise ValueError(f"Prompt set pairs mismatch: missing={expected_pairs - observed_pairs}, extra={observed_pairs - expected_pairs}")
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


def write_summary(path: Path, tracked_rows: list[dict[str, Any]], prompt_position_count: int) -> None:
    hits = [row for row in tracked_rows if row["appeared_in_topk50"] == "1"]
    hits_by_feature = defaultdict(list)
    hits_by_perturb = Counter()
    hits_by_position = Counter()
    hits_by_feature_perturb = defaultdict(Counter)
    for row in hits:
        feature_id = int(row["feature_id"])
        perturb = row["perturbation_type"]
        hits_by_feature[feature_id].append(row)
        hits_by_perturb[perturb] += 1
        hits_by_position[row["position_label"]] += 1
        hits_by_feature_perturb[feature_id][perturb] += 1

    def feature_concentration_sentence(feature_id: int, target: str) -> str:
        rows = hits_by_feature.get(feature_id, [])
        if not rows:
            return f"Feature {feature_id} did not appear in TopK-50 in this matched control."
        counts = hits_by_feature_perturb[feature_id]
        if set(counts) == {target}:
            return f"Feature {feature_id} appeared only in `{target}` prompts in this matched control."
        if counts.get(target, 0) > 0:
            return (
                f"Feature {feature_id} appeared in `{target}` prompts and also in "
                + ", ".join(f"`{k}`={v}" for k, v in sorted(counts.items()) if k != target)
                + "."
            )
        return f"Feature {feature_id} did not appear in `{target}` prompts; observed counts were " + ", ".join(f"`{k}`={v}" for k, v in sorted(counts.items())) + "."

    handled = {"e_to_ē", "s_to_ş"}
    d_count = hits_by_perturb.get("d_to_ḑ", 0)
    handled_count = sum(hits_by_perturb.get(x, 0) for x in handled)
    sdot_count = hits_by_perturb.get("s_to_ṡ", 0)
    if abs(sdot_count - d_count) < abs(sdot_count - handled_count):
        sdot_sentence = f"`s_to_ṡ` had {sdot_count} tracked hits, closer to `d_to_ḑ` ({d_count}) than handled controls ({handled_count})."
    elif abs(sdot_count - handled_count) < abs(sdot_count - d_count):
        sdot_sentence = f"`s_to_ṡ` had {sdot_count} tracked hits, closer to handled controls ({handled_count}) than `d_to_ḑ` ({d_count})."
    else:
        sdot_sentence = f"`s_to_ṡ` had {sdot_count} tracked hits, equally distant from `d_to_ḑ` ({d_count}) and handled controls ({handled_count}) by count."

    lines = [
        "# Layer 26 Matched Perturbation Control Summary",
        "",
        "Evidence-only summary from the 10-prompt matched perturbation control. No semantic labels are assigned here.",
        "",
        f"Prompt-position pairs scanned: {prompt_position_count}.",
        f"Tracked TopK-50 hit rows: {len(hits)}.",
        "",
        "## Do 2722 And 7108 Stay Concentrated In ASCII Original Prompts?",
        "",
        "- " + feature_concentration_sentence(2722, "ascii_original"),
        "- " + feature_concentration_sentence(7108, "ascii_original"),
        "",
        "## Does 31784 Concentrate In D-To-Dstroke Prompts?",
        "",
        "- " + feature_concentration_sentence(31784, "d_to_ḑ"),
        "",
        "## Do Handled Controls Reproduce The Same Tracked Hits?",
        "",
    ]
    handled_hits = [row for row in hits if row["perturbation_type"] in handled]
    if handled_hits:
        counts = Counter((row["perturbation_type"], row["feature_id"]) for row in handled_hits)
        lines.append(
            "- Handled controls produced tracked hits: "
            + ", ".join(f"{perturb}/feature {feature_id}={count}" for (perturb, feature_id), count in sorted(counts.items()))
            + "."
        )
    else:
        lines.append("- Handled controls did not produce tracked TopK-50 hits for the tracked features.")

    lines.extend([
        "",
        "## Does S-To-Sdot Behave More Like D-To-Dstroke Or Handled Controls?",
        "",
        "- " + sdot_sentence,
        "",
        "## Are Tracked Hits Concentrated At Final Prompt Token Or Nearby Boundary Positions?",
        "",
    ])
    if hits_by_position:
        lines.append("- Tracked hit counts by position: " + ", ".join(f"{k}={v}" for k, v in sorted(hits_by_position.items())) + ".")
    else:
        lines.append("- No tracked hits were observed at any captured boundary position.")

    lines.extend(["", "## Strongest Tracked Hits", ""])
    if hits:
        for row in sorted(hits, key=lambda item: float(item["activation"]), reverse=True)[:20]:
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
        base_prompt_family = prompt["base_prompt_family"]
        perturbation_type = prompt["perturbation_type"]
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
                    "base_prompt_family": base_prompt_family,
                    "perturbation_type": perturbation_type,
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
                    "base_prompt_family": base_prompt_family,
                    "perturbation_type": perturbation_type,
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
                    "base_prompt_family": base_prompt_family,
                    "perturbation_type": perturbation_type,
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
            "base_prompt_family": base_prompt_family,
            "perturbation_type": perturbation_type,
            "prompt_text": prompt_text,
            "prompt_token_count": prompt_token_count,
            "generated_text_short": generated_text_short,
        })
        prompt_metadata.append({
            "prompt_id": prompt_id,
            "base_prompt_family": base_prompt_family,
            "perturbation_type": perturbation_type,
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
    metadata_path = OUT_DIR / "layer26_matched_perturbation_metadata.json"
    summary_path = OUT_DIR / "layer26_matched_perturbation_summary.md"

    with tracked_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "base_prompt_family",
            "perturbation_type",
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
            "base_prompt_family",
            "perturbation_type",
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
            "base_prompt_family",
            "perturbation_type",
            "prompt_text",
            "prompt_token_count",
            "generated_text_short",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(generated_rows)

    write_summary(summary_path, tracked_rows, prompt_position_count)
    completed_at = utc_now()
    metadata = {
        "started_at": started_at,
        "completed_at": completed_at,
        "purpose": "layer-26 matched perturbation control for candidate features",
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
        "perturbation_types": PERTURBATION_TYPES,
        "base_prompt_families": BASE_PROMPT_FAMILIES,
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
        "perturbation_types=" + ",".join(PERTURBATION_TYPES),
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
    print("layer26_matched_perturbation_control_status=ok")


if __name__ == "__main__":
    main()
