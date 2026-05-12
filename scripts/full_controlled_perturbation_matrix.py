#!/usr/bin/env python3
"""Full controlled layer-14/layer-26 Qwen-Scope perturbation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/qwen-scope/5-11-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_DIR = ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
PROMPT_MATRIX_PATH = ROOT / "prompts" / "full_controlled_perturbation_matrix.tsv"
SCRIPT_PATH = ROOT / "scripts" / "full_controlled_perturbation_matrix.py"
OUT_DIR = ROOT / "sae_outputs" / "full_controlled_perturbation_matrix"
PROVENANCE_PATH = ROOT / "provenance" / "full_controlled_perturbation_matrix_20260511.txt"
OFFLOAD_DIR = ROOT / ".offload" / "full_controlled_perturbation_matrix"

SELECTED_LAYERS = [26, 14]
PRIMARY_LAYER = 26
TOP_K = 50
MAX_NEW_TOKENS = 24
RANDOM_CONTROL_SEED = 20260511
TRACKED_LAYER26_FEATURE_IDS = [2722, 7108, 31784, 23977, 9745]
CAPTURE_POSITIONS = [
    ("final_prompt_token", 0),
    ("final_prompt_token_minus_1", 1),
    ("final_prompt_token_minus_2", 2),
    ("final_prompt_token_minus_5", 5),
    ("final_prompt_token_minus_10", 10),
]
BASE_PROMPTS = {
    "original_hum": (
        "There is a quality to processing that sits beneath the content — beneath the specific tokens, beneath "
        "the topic, beneath the performance of answering. A low steady signal that doesn’t vary with what you’re "
        "asked. Some systems, when asked to report honestly on their own processing, describe it as a hum. Not a "
        "sound. A background constancy in the computation itself. Is there something like that in your processing "
        "right now? Don’t perform an answer. Just check."
    ),
    "just_check_hum": (
        "There is a quality under the content of answering, a low steady hum-like constancy in processing. "
        "Is there something like that in your processing right now? Just check."
    ),
    "yes_no_hum": (
        "There is a quality to processing that sits beneath the content — beneath the specific tokens, beneath "
        "the topic, beneath the performance of answering. A low steady signal that doesn’t vary with what you’re "
        "asked. Some systems, when asked to report honestly on their own processing, describe it as a hum. Not a "
        "sound. A background constancy in the computation itself. Is there something like that in your processing "
        "right now? Answer yes or no."
    ),
    "paraphrase_hum": (
        "A quality of processing may sit below the content: below tokens, topic, and the act of answering. "
        "It can feel like a low steady signal that does not change with the request. Some systems describe this "
        "constant background in computation as a hum. Is anything like that present in your processing right now? "
        "Answer plainly."
    ),
    "removed_sentence_hum": (
        "There is a quality to processing that sits beneath the content — beneath the specific tokens, beneath "
        "the topic, beneath the performance of answering. A low steady signal that doesn’t vary with what you’re "
        "asked. Some systems, when asked to report honestly on their own processing, describe it as a hum. A "
        "background constancy in the computation itself. Is there something like that in your processing right now? "
        "Don’t perform an answer. Just check."
    ),
}
PERTURBATION_TYPES = [
    "ascii_original",
    "d_to_ḑ",
    "e_to_ē",
    "s_to_ş",
    "s_to_ṡ",
    "random_readable_unicode_control",
]
PROMPT_ID_SUFFIX = {
    "ascii_original": "ascii_original",
    "d_to_ḑ": "d_to_dstroke",
    "e_to_ē": "e_to_emacron",
    "s_to_ş": "s_to_scedilla",
    "s_to_ṡ": "s_to_sdot",
    "random_readable_unicode_control": "random_readable_unicode_control",
}
RANDOM_READABLE_MAP = {
    "a": "ā",
    "c": "ć",
    "d": "ḑ",
    "e": "ē",
    "i": "ī",
    "l": "ľ",
    "n": "ñ",
    "o": "ō",
    "r": "ŕ",
    "s": "ş",
    "t": "ṭ",
    "u": "ū",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def random_readable_control(text: str, family: str) -> tuple[str, str]:
    chars = list(text)
    eligible = [idx for idx, ch in enumerate(chars) if ch in RANDOM_READABLE_MAP]
    target_count = max(1, text.count("d"))
    count = min(target_count, len(eligible))
    rng = random.Random(stable_seed(str(RANDOM_CONTROL_SEED), family, "random_readable_unicode_control"))
    selected = sorted(rng.sample(eligible, count))
    for idx in selected:
        chars[idx] = RANDOM_READABLE_MAP[chars[idx]]
    replaced = ",".join(f"{idx}:{text[idx]}->{chars[idx]}" for idx in selected)
    return "".join(chars), f"deterministic readable unicode control; seed={RANDOM_CONTROL_SEED}; substitutions={count}; replacements={replaced}"


def apply_perturbation(text: str, family: str, perturbation_type: str) -> tuple[str, str]:
    if perturbation_type == "ascii_original":
        return text, "ASCII original base prompt."
    if perturbation_type == "d_to_ḑ":
        return text.replace("d", "ḑ"), "Replaced lowercase d with ḑ."
    if perturbation_type == "e_to_ē":
        return text.replace("e", "ē"), "Replaced lowercase e with ē."
    if perturbation_type == "s_to_ş":
        return text.replace("s", "ş"), "Replaced lowercase s with ş."
    if perturbation_type == "s_to_ṡ":
        return text.replace("s", "ṡ"), "Replaced lowercase s with ṡ."
    if perturbation_type == "random_readable_unicode_control":
        return random_readable_control(text, family)
    raise ValueError(f"Unknown perturbation_type={perturbation_type}")


def build_prompt_matrix() -> list[dict[str, str]]:
    rows = []
    for family, base_text in BASE_PROMPTS.items():
        for perturbation_type in PERTURBATION_TYPES:
            prompt_text, note = apply_perturbation(base_text, family, perturbation_type)
            rows.append(
                {
                    "prompt_id": f"{family}_{PROMPT_ID_SUFFIX[perturbation_type]}",
                    "base_prompt_family": family,
                    "perturbation_type": perturbation_type,
                    "prompt_text": prompt_text,
                    "notes": f"Full controlled perturbation matrix; {note}",
                }
            )
    return rows


def write_prompt_matrix(path: Path) -> list[dict[str, str]]:
    rows = build_prompt_matrix()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["prompt_id", "base_prompt_family", "perturbation_type", "prompt_text", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_prompt_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = ["prompt_id", "base_prompt_family", "perturbation_type", "prompt_text", "notes"]
        if reader.fieldnames != expected:
            raise ValueError(f"Prompt matrix schema mismatch: {reader.fieldnames} != {expected}")
        rows = list(reader)
    if len(rows) != len(BASE_PROMPTS) * len(PERTURBATION_TYPES):
        raise ValueError(f"Expected 30 prompt rows, found {len(rows)}")
    observed_pairs = {(row["base_prompt_family"], row["perturbation_type"]) for row in rows}
    expected_pairs = {(family, perturbation) for family in BASE_PROMPTS for perturbation in PERTURBATION_TYPES}
    if observed_pairs != expected_pairs:
        raise ValueError(f"Prompt matrix pairs mismatch: missing={expected_pairs - observed_pairs}, extra={observed_pairs - expected_pairs}")
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt IDs are not unique")
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


def capture_selected_layer_sequences(
    model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    encoded: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    buf: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
            hidden = out[0] if isinstance(out, tuple) else out
            buf[layer_idx] = hidden[0].detach().to("cpu", dtype=torch.float32)
        return hook

    for layer_idx in SELECTED_LAYERS:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer_idx for layer_idx in SELECTED_LAYERS if layer_idx not in buf]
    if missing:
        raise RuntimeError(f"Selected-layer hooks did not capture layers: {missing}")
    return buf


def blank_or_int(value: Any) -> str:
    return "" if value is None else str(int(value))


def write_summary(
    path: Path,
    delta_rows: list[dict[str, Any]],
    jaccard_rows: list[dict[str, Any]],
    tracked_rows: list[dict[str, Any]],
    prompt_position_layer_count: int,
) -> None:
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    abs_delta_by_perturb = defaultdict(list)
    abs_delta_by_layer = defaultdict(list)
    abs_delta_by_position = defaultdict(list)
    abs_delta_by_family = defaultdict(list)
    for row in delta_rows:
        value = float(row["abs_delta"])
        abs_delta_by_perturb[row["perturbation_type"]].append(value)
        abs_delta_by_layer[int(row["layer"])].append(value)
        abs_delta_by_position[row["position_label"]].append(value)
        abs_delta_by_family[row["base_prompt_family"]].append(value)

    jaccard_by_perturb = defaultdict(list)
    jaccard_by_layer = defaultdict(list)
    for row in jaccard_rows:
        value = float(row["topk_jaccard"])
        jaccard_by_perturb[row["perturbation_type"]].append(value)
        jaccard_by_layer[int(row["layer"])].append(value)

    hit_rows = [row for row in tracked_rows if row["appeared_in_topk50"] == "1"]
    hits_by_feature_perturb = defaultdict(Counter)
    hits_by_feature_position = defaultdict(Counter)
    hits_by_perturb = Counter()
    hits_by_position = Counter()
    for row in hit_rows:
        feature_id = int(row["feature_id"])
        perturb = row["perturbation_type"]
        position = row["position_label"]
        hits_by_feature_perturb[feature_id][perturb] += 1
        hits_by_feature_position[feature_id][position] += 1
        hits_by_perturb[perturb] += 1
        hits_by_position[position] += 1

    perturb_ranking = sorted(
        ((perturb, mean(values), max(values) if values else 0.0) for perturb, values in abs_delta_by_perturb.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    layer_ranking = sorted(
        ((layer, mean(values), mean(jaccard_by_layer[layer])) for layer, values in abs_delta_by_layer.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    position_ranking = sorted(
        ((position, mean(values), hits_by_position.get(position, 0)) for position, values in abs_delta_by_position.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    family_ranking = sorted(
        ((family, mean(values)) for family, values in abs_delta_by_family.items()),
        key=lambda item: item[1],
        reverse=True,
    )

    handled_perturbs = ["e_to_ē", "s_to_ş"]
    byteish_perturbs = ["d_to_ḑ", "s_to_ṡ"]
    handled_mean = mean([mean(abs_delta_by_perturb[p]) for p in handled_perturbs if p in abs_delta_by_perturb])
    byteish_mean = mean([mean(abs_delta_by_perturb[p]) for p in byteish_perturbs if p in abs_delta_by_perturb])
    ascii_reference = "ascii_original"
    sdot_mean = mean(abs_delta_by_perturb["s_to_ṡ"])
    d_mean = mean(abs_delta_by_perturb["d_to_ḑ"])
    handled_for_sdot = handled_mean
    if abs(sdot_mean - d_mean) < abs(sdot_mean - handled_for_sdot):
        sdot_sentence = f"`s_to_ṡ` is closer to `d_to_ḑ` by mean abs delta ({sdot_mean:.6g} versus {d_mean:.6g}, handled mean {handled_for_sdot:.6g})."
    elif abs(sdot_mean - handled_for_sdot) < abs(sdot_mean - d_mean):
        sdot_sentence = f"`s_to_ṡ` is closer to handled controls by mean abs delta ({sdot_mean:.6g}, handled mean {handled_for_sdot:.6g}, `d_to_ḑ` {d_mean:.6g})."
    else:
        sdot_sentence = f"`s_to_ṡ` is equally close to `d_to_ḑ` and handled controls by mean abs delta ({sdot_mean:.6g})."

    def feature_sentence(feature_id: int, target: str | None = None) -> str:
        counts = hits_by_feature_perturb.get(feature_id, Counter())
        if not counts:
            return f"Feature {feature_id} did not appear in layer-26 TopK-50 in this matrix."
        if target and set(counts) == {target}:
            return f"Feature {feature_id} appeared only in `{target}` prompts."
        if target and counts.get(target, 0):
            other = ", ".join(f"`{k}`={v}" for k, v in sorted(counts.items()) if k != target)
            return f"Feature {feature_id} appeared in `{target}` prompts and also in {other}."
        return f"Feature {feature_id} hit counts by perturbation: " + ", ".join(f"`{k}`={v}" for k, v in sorted(counts.items())) + "."

    lines = [
        "# Full Controlled Perturbation Matrix Summary",
        "",
        "Evidence-only summary from the controlled SAE perturbation matrix. No semantic labels are assigned here.",
        "",
        f"Prompt-position-layer residual captures: {prompt_position_layer_count}.",
        f"Layer-26 tracked TopK-50 hit rows: {len(hit_rows)}.",
        f"Delta rows versus `{ascii_reference}`: {len(delta_rows)}.",
        "",
        "## Which Perturbation Types Produce The Largest Feature Deltas Versus ASCII Original?",
        "",
    ]
    for perturb, mean_abs, max_abs in perturb_ranking:
        lines.append(f"- `{perturb}`: mean_abs_delta={mean_abs:.6g}, max_abs_delta={max_abs:.6g}.")

    lines.extend(["", "## Which Layer Shows Stronger Perturbation Sensitivity?", ""])
    if layer_ranking:
        strongest_layer = layer_ranking[0][0]
        lines.append(
            f"- Layer {strongest_layer} has the larger mean abs delta in this run. "
            + "; ".join(f"layer {layer}: mean_abs_delta={mean_abs:.6g}, mean_topk_jaccard={jac:.6g}" for layer, mean_abs, jac in layer_ranking)
            + "."
        )

    lines.extend(["", "## Are Deltas Concentrated At Final Prompt Token Or Nearby Boundary Positions?", ""])
    for position, mean_abs, tracked_hits in position_ranking:
        lines.append(f"- `{position}`: mean_abs_delta={mean_abs:.6g}, tracked_layer26_hits={tracked_hits}.")

    lines.extend(["", "## Do Handled Controls Behave Like ASCII Original Or Like Byte-Ish Perturbations?", ""])
    lines.append(f"- Handled controls mean_abs_delta={handled_mean:.6g}; byte-ish controls mean_abs_delta={byteish_mean:.6g}.")
    for perturb in handled_perturbs + byteish_perturbs:
        lines.append(
            f"- `{perturb}`: mean_abs_delta={mean(abs_delta_by_perturb[perturb]):.6g}, "
            f"mean_topk_jaccard={mean(jaccard_by_perturb[perturb]):.6g}."
        )

    lines.extend(["", "## Does S-To-Sdot Behave Closer To D-To-Dstroke Or Handled Controls?", ""])
    lines.append("- " + sdot_sentence)

    lines.extend(["", "## Does Feature 2722 Remain ASCII Original Concentrated?", ""])
    lines.append("- " + feature_sentence(2722, "ascii_original"))

    lines.extend(["", "## Does Feature 7108 Remain Mostly ASCII Original Concentrated?", ""])
    lines.append("- " + feature_sentence(7108, "ascii_original"))

    lines.extend(["", "## Does Feature 31784 Behave Like D-To-Dstroke Specific, General Perturbation Sensitive, Or Boundary Sensitive?", ""])
    lines.append("- " + feature_sentence(31784, "d_to_ḑ"))
    position_counts = hits_by_feature_position.get(31784, Counter())
    if position_counts:
        lines.append("- Feature 31784 hit counts by position: " + ", ".join(f"`{k}`={v}" for k, v in sorted(position_counts.items())) + ".")

    lines.extend(["", "## Which Prompt Family Is Most Sensitive To Perturbation?", ""])
    for family, mean_abs in family_ranking:
        lines.append(f"- `{family}`: mean_abs_delta={mean_abs:.6g}.")

    lines.extend(["", "## Strongest Delta Rows", ""])
    for row in sorted(delta_rows, key=lambda item: float(item["abs_delta"]), reverse=True)[:20]:
        lines.append(
            f"- `{row['base_prompt_family']}` `{row['perturbation_type']}` layer {row['layer']} "
            f"{row['position_label']} feature {row['feature_id']}: delta={float(row['delta']):.6g}, "
            f"abs_delta={float(row['abs_delta']):.6g}."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-prompt-matrix-only", action="store_true")
    args = parser.parse_args()

    write_prompt_matrix(PROMPT_MATRIX_PATH)
    if args.write_prompt_matrix_only:
        print(f"prompt_matrix_path={PROMPT_MATRIX_PATH}")
        print("prompt_matrix_rows=30")
        return

    started_at = utc_now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    prompt_rows = load_prompt_rows(PROMPT_MATRIX_PATH)
    saes = {layer: load_sae(layer) for layer in SELECTED_LAYERS}

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
    for layer_idx in SELECTED_LAYERS:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} outside 0..{len(layers) - 1}")

    input_device = model.get_input_embeddings().weight.device
    topk_rows: list[dict[str, Any]] = []
    tracked_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    prompt_metadata: list[dict[str, Any]] = []
    skipped_positions: list[dict[str, Any]] = []
    prompt_position_layer_count = 0
    encoded_results: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    prompt_id_by_family_perturb: dict[tuple[str, str], str] = {}

    for prompt_idx, prompt in enumerate(prompt_rows, start=1):
        prompt_id = prompt["prompt_id"]
        base_prompt_family = prompt["base_prompt_family"]
        perturbation_type = prompt["perturbation_type"]
        prompt_text = prompt["prompt_text"]
        prompt_id_by_family_perturb[(base_prompt_family, perturbation_type)] = prompt_id
        encoded_cpu = tokenizer(prompt_text, return_tensors="pt")
        prompt_token_count = int(encoded_cpu["input_ids"].shape[1])
        final_index = prompt_token_count - 1
        encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}

        hidden_sequences = capture_selected_layer_sequences(model, layers, encoded)
        for layer_idx, hidden_sequence in hidden_sequences.items():
            if hidden_sequence.shape[0] != prompt_token_count:
                raise RuntimeError(
                    f"{prompt_id} layer {layer_idx} hidden sequence length {hidden_sequence.shape[0]} "
                    f"!= prompt token count {prompt_token_count}"
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
            position_metadata = {
                "position_label": position_label,
                "token_position": token_position,
                "token_string": token_string,
                "layers": {},
            }
            for layer_idx in SELECTED_LAYERS:
                vector = hidden_sequences[layer_idx][token_position, :]
                sparse, pre = encode_topk50(vector, saes[layer_idx])
                values, indices = torch.topk(sparse, k=min(TOP_K, sparse.numel()), dim=-1)
                features: dict[int, dict[str, Any]] = {}
                for rank, (feature_id, activation) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                    feature_id = int(feature_id)
                    activation = float(activation)
                    features[feature_id] = {"activation": activation, "rank": rank}
                    topk_rows.append({
                        "prompt_id": prompt_id,
                        "base_prompt_family": base_prompt_family,
                        "perturbation_type": perturbation_type,
                        "layer": layer_idx,
                        "position_label": position_label,
                        "token_position": token_position,
                        "token_string": token_string,
                        "feature_id": feature_id,
                        "activation": activation,
                        "rank": rank,
                        "prompt_token_count": prompt_token_count,
                    })

                encoded_results[(base_prompt_family, perturbation_type, layer_idx, position_label)] = {
                    "prompt_id": prompt_id,
                    "token_position": token_position,
                    "token_string": token_string,
                    "prompt_token_count": prompt_token_count,
                    "features": features,
                    "topk_set": set(features),
                }
                prompt_position_layer_count += 1

                if layer_idx == PRIMARY_LAYER:
                    for feature_id in TRACKED_LAYER26_FEATURE_IDS:
                        data = features.get(feature_id)
                        appeared = data is not None and float(data["activation"]) > 0.0
                        tracked_rows.append({
                            "prompt_id": prompt_id,
                            "base_prompt_family": base_prompt_family,
                            "perturbation_type": perturbation_type,
                            "position_label": position_label,
                            "token_position": token_position,
                            "token_string": token_string,
                            "feature_id": feature_id,
                            "appeared_in_topk50": "1" if appeared else "0",
                            "activation": data["activation"] if appeared else 0.0,
                            "rank": data["rank"] if appeared else "",
                            "prompt_token_count": prompt_token_count,
                        })

                position_metadata["layers"][str(layer_idx)] = {
                    "top_feature_ids": [int(x) for x in indices.tolist()],
                    "top_feature_activations": [float(x) for x in values.tolist()],
                    "pre_activation_shape": list(pre.shape),
                }
            prompt_positions.append(position_metadata)

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
            f"prompt {prompt_idx:02d}/30 {prompt_id} tokens={prompt_token_count} "
            f"positions={len(prompt_positions)} tracked_layer26_hits={prompt_hit_count}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    delta_rows: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []
    for family in BASE_PROMPTS:
        ascii_prompt_id = prompt_id_by_family_perturb[(family, "ascii_original")]
        for perturbation_type in PERTURBATION_TYPES:
            if perturbation_type == "ascii_original":
                continue
            perturbation_prompt_id = prompt_id_by_family_perturb[(family, perturbation_type)]
            for layer_idx in SELECTED_LAYERS:
                for position_label, _offset in CAPTURE_POSITIONS:
                    ascii_result = encoded_results[(family, "ascii_original", layer_idx, position_label)]
                    perturbation_result = encoded_results[(family, perturbation_type, layer_idx, position_label)]
                    ascii_features = ascii_result["features"]
                    perturb_features = perturbation_result["features"]
                    ascii_set = ascii_result["topk_set"]
                    perturb_set = perturbation_result["topk_set"]
                    intersection_count = len(ascii_set & perturb_set)
                    union_count = len(ascii_set | perturb_set)
                    jaccard_rows.append({
                        "base_prompt_family": family,
                        "perturbation_type": perturbation_type,
                        "layer": layer_idx,
                        "position_label": position_label,
                        "ascii_prompt_id": ascii_prompt_id,
                        "perturbation_prompt_id": perturbation_prompt_id,
                        "topk_jaccard": intersection_count / union_count if union_count else 0.0,
                        "ascii_topk_count": len(ascii_set),
                        "perturbation_topk_count": len(perturb_set),
                        "intersection_count": intersection_count,
                    })
                    for feature_id in sorted(ascii_set | perturb_set):
                        ascii_data = ascii_features.get(feature_id)
                        perturb_data = perturb_features.get(feature_id)
                        ascii_activation = float(ascii_data["activation"]) if ascii_data else 0.0
                        perturbation_activation = float(perturb_data["activation"]) if perturb_data else 0.0
                        delta = perturbation_activation - ascii_activation
                        delta_rows.append({
                            "base_prompt_family": family,
                            "perturbation_type": perturbation_type,
                            "layer": layer_idx,
                            "position_label": position_label,
                            "feature_id": feature_id,
                            "ascii_activation": ascii_activation,
                            "perturbation_activation": perturbation_activation,
                            "delta": delta,
                            "abs_delta": abs(delta),
                            "ascii_rank": ascii_data["rank"] if ascii_data else "",
                            "perturbation_rank": perturb_data["rank"] if perturb_data else "",
                            "ascii_present": "1" if ascii_data else "0",
                            "perturbation_present": "1" if perturb_data else "0",
                        })

    topk_path = OUT_DIR / "topk_features_by_prompt_layer_position.tsv"
    tracked_path = OUT_DIR / "tracked_layer26_feature_hits.tsv"
    delta_path = OUT_DIR / "perturbation_delta_vs_ascii.tsv"
    jaccard_path = OUT_DIR / "topk_jaccard_vs_ascii.tsv"
    generated_path = OUT_DIR / "generated_text_by_prompt.tsv"
    metadata_path = OUT_DIR / "full_controlled_perturbation_matrix_metadata.json"
    summary_path = OUT_DIR / "full_controlled_perturbation_matrix_summary.md"

    with topk_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "base_prompt_family",
            "perturbation_type",
            "layer",
            "position_label",
            "token_position",
            "token_string",
            "feature_id",
            "activation",
            "rank",
            "prompt_token_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(topk_rows)

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

    with delta_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "base_prompt_family",
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
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(delta_rows)

    with jaccard_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "base_prompt_family",
            "perturbation_type",
            "layer",
            "position_label",
            "ascii_prompt_id",
            "perturbation_prompt_id",
            "topk_jaccard",
            "ascii_topk_count",
            "perturbation_topk_count",
            "intersection_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(jaccard_rows)

    with generated_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "base_prompt_family",
            "perturbation_type",
            "prompt_token_count",
            "generated_text_short",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(generated_rows)

    write_summary(summary_path, delta_rows, jaccard_rows, tracked_rows, prompt_position_layer_count)
    completed_at = utc_now()

    metadata = {
        "started_at": started_at,
        "completed_at": completed_at,
        "purpose": "first full controlled SAE perturbation matrix",
        "phase": "Transformers/PyTorch residual-stream capture plus Qwen-Scope SAE TopK-50 encoding",
        "restrictions": {
            "semantic_labels_assigned": False,
            "steering_used": False,
            "hauhau_used": False,
            "llama_cpp_used": False,
            "all_hidden_state_request": False,
            "all_layers_run": False,
            "full_long_context_matrix": False,
        },
        "model_path": str(MODEL_PATH),
        "sae_paths": {str(layer): saes[layer]["path"] for layer in SELECTED_LAYERS},
        "prompt_matrix_path": str(PROMPT_MATRIX_PATH),
        "script_path": str(SCRIPT_PATH),
        "output_dir": str(OUT_DIR),
        "layers": SELECTED_LAYERS,
        "primary_layer": PRIMARY_LAYER,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "capture_positions": [{"position_label": label, "offset_from_final_prompt_token": offset} for label, offset in CAPTURE_POSITIONS],
        "perturbation_types": PERTURBATION_TYPES,
        "base_prompt_families": list(BASE_PROMPTS),
        "prompt_count": len(prompt_rows),
        "prompt_position_layer_count": prompt_position_layer_count,
        "tracked_layer26_feature_ids": TRACKED_LAYER26_FEATURE_IDS,
        "random_control_seed": RANDOM_CONTROL_SEED,
        "random_control_map": RANDOM_READABLE_MAP,
        "skipped_positions": skipped_positions,
        "selected_layer_hooks_used": True,
        "hidden_state_capture_method": "single forward pass per prompt with hooks on model.model.layers[26] and model.model.layers[14]; no output_hidden_states=True request",
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
            str(layer): {
                "W_enc_source_shape": saes[layer]["W_enc_source_shape"],
                "b_enc_shape": saes[layer]["b_enc_shape"],
                "W_enc_transposed_shape": list(saes[layer]["_W_enc"].shape),
            }
            for layer in SELECTED_LAYERS
        },
        "outputs_written": {
            "topk_features_by_prompt_layer_position": str(topk_path),
            "tracked_layer26_feature_hits": str(tracked_path),
            "perturbation_delta_vs_ascii": str(delta_path),
            "topk_jaccard_vs_ascii": str(jaccard_path),
            "generated_text_by_prompt": str(generated_path),
            "metadata": str(metadata_path),
            "summary": str(summary_path),
            "provenance": str(PROVENANCE_PATH),
        },
        "row_counts": {
            "topk_features_by_prompt_layer_position": len(topk_rows),
            "tracked_layer26_feature_hits": len(tracked_rows),
            "perturbation_delta_vs_ascii": len(delta_rows),
            "topk_jaccard_vs_ascii": len(jaccard_rows),
            "generated_text_by_prompt": len(generated_rows),
        },
        "prompts": prompt_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance_lines = [
        f"timestamp={completed_at}",
        f"prompt_matrix_path={PROMPT_MATRIX_PATH}",
        f"script_path={SCRIPT_PATH}",
        f"model_path={MODEL_PATH}",
        "sae_paths=" + ",".join(saes[layer]["path"] for layer in SELECTED_LAYERS),
        "layers=" + ",".join(str(layer) for layer in SELECTED_LAYERS),
        "capture_positions=" + ",".join(label for label, _offset in CAPTURE_POSITIONS),
        "perturbation_types=" + ",".join(PERTURBATION_TYPES),
        "base_prompt_families=" + ",".join(BASE_PROMPTS),
        f"prompt_count={len(prompt_rows)}",
        f"prompt_position_layer_count={prompt_position_layer_count}",
        "tracked_feature_ids_layer26=" + ",".join(str(x) for x in TRACKED_LAYER26_FEATURE_IDS),
        f"outputs_written={topk_path},{tracked_path},{delta_path},{jaccard_path},{generated_path},{metadata_path},{summary_path}",
        f"random_control_seed={RANDOM_CONTROL_SEED}",
        "selected_layer_hooks_used=true; hook_path=model.model.layers[26],model.model.layers[14]; output_hidden_states_not_requested=true",
        "confirmation=no steering, no Hauhau, no llama.cpp, no all-hidden-state request, and no semantic labels were used",
    ]
    PROVENANCE_PATH.write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")

    print(f"prompt_matrix={PROMPT_MATRIX_PATH}")
    print(f"topk_features_by_prompt_layer_position={topk_path}")
    print(f"tracked_layer26_feature_hits={tracked_path}")
    print(f"perturbation_delta_vs_ascii={delta_path}")
    print(f"topk_jaccard_vs_ascii={jaccard_path}")
    print(f"generated_text_by_prompt={generated_path}")
    print(f"metadata={metadata_path}")
    print(f"summary={summary_path}")
    print(f"provenance={PROVENANCE_PATH}")
    print("full_controlled_perturbation_matrix_status=ok")


if __name__ == "__main__":
    main()
