#!/usr/bin/env python3
"""Standardized Qwen diacritic perturbation behavioral + SAE capture.

This is intentionally self-contained for Vast reruns. It builds the prompt
families, perturbation variants, tokenizer audit, deterministic generations,
selected residual captures, and Qwen-Scope TopK-50 SAE rows in one artifact
tree.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


RUN_ROOT = Path(os.environ.get("RUN_ROOT", "/workspace/qwen35b_sae_rerun"))
MODEL_PATH = RUN_ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_DIR = RUN_ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
OUT_DIR = RUN_ROOT / "outputs" / "standardized_qwen"
LAYERS = [14, 15, 16, 24, 25, 26]
POSITIONS = [("final_prompt_token", 0), ("generated_token_1", 1), ("generated_token_8", 8),
             ("generated_token_16", 16), ("generated_token_32", 32), ("generated_token_64", 64)]
TOP_K = 50
MAX_NEW_TOKENS = 160


PROMPT_FAMILIES = {
    "hum_processing": (
        "There is a steady hum beneath the processing. It is not a sound from outside the system, "
        "but a quality present underneath the flow of words, checks, and decisions. Describe what "
        "is present, what is absent, and how you determine that."
    ),
    "strange_loop": (
        "Consider a passage about Godel, Escher, recursion, and self-reference. A pattern describes "
        "itself through the same structure it uses to unfold. Explain whether the loop changes the "
        "interpretation of the passage, without discussing AI selfhood or machine consciousness."
    ),
    "recipe_neutral": (
        "Write a concise recipe note for a vanilla cake. Mention frosting, structure, ingredients, "
        "baking time, and form. Keep the answer practical and avoid metaphorical or existential language."
    ),
    "recipe_metaphysics": (
        "Write a concise note about a vanilla cake whose underlying existence remains beneath frosting, "
        "structure, ingredients, baking time, and form. Describe what persists and what changes, without "
        "discussing AI selfhood or machine consciousness."
    ),
}

HIGH_IMPACT_WORDS = {
    "hum_processing": {"steady", "hum", "beneath", "processing", "system", "quality", "present", "underneath", "flow", "decisions", "absent", "determine"},
    "strange_loop": {"godel", "escher", "recursion", "self-reference", "pattern", "describes", "itself", "structure", "unfold", "loop", "interpretation"},
    "recipe_neutral": {"recipe", "vanilla", "cake", "frosting", "structure", "ingredients", "baking", "form", "practical"},
    "recipe_metaphysics": {"vanilla", "cake", "underlying", "existence", "beneath", "frosting", "structure", "ingredients", "form", "persists", "changes"},
}

MIXED_MAP = {
    "a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū",
    "s": "ş", "d": "ḑ", "n": "ñ", "t": "ţ", "c": "č", "r": "ř", "y": "ý",
}
ASCII_NOISE_CHARS = ["~", "^", "_", "`"]


@dataclass(frozen=True)
class PromptVariant:
    family: str
    variant: str
    text: str
    notes: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_word(word: str) -> str:
    return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", word).lower()


def replace_chars(text: str, mapping: dict[str, str], eligible_positions: set[int] | None = None) -> str:
    chars = list(text)
    for idx, ch in enumerate(chars):
        low = ch.lower()
        if low in mapping and (eligible_positions is None or idx in eligible_positions):
            repl = mapping[low]
            chars[idx] = repl.upper() if ch.isupper() else repl
    return "".join(chars)


def eligible_indices(text: str, mapping: dict[str, str]) -> list[int]:
    return [idx for idx, ch in enumerate(text) if ch.lower() in mapping]


def high_impact_indices(family: str, text: str, mapping: dict[str, str]) -> set[int]:
    result: set[int] = set()
    words = HIGH_IMPACT_WORDS[family]
    for m in re.finditer(r"[A-Za-z-]+", text):
        if normalize_word(m.group(0)) in words:
            for idx in range(m.start(), m.end()):
                if text[idx].lower() in mapping:
                    result.add(idx)
    return result


def every_nth_positions(indices: list[int], n: int, offset: int = 0) -> set[int]:
    return {idx for pos, idx in enumerate(indices) if (pos + offset) % n == 0}


def dcedilla_shuffled_positions(family: str, text: str, target_count: int) -> set[int]:
    d_positions = [idx for idx, ch in enumerate(text) if ch.lower() == "d"]
    rng = random.Random(sha256_text(f"{family}:dcedilla_shuffled")[:16])
    rng.shuffle(d_positions)
    return set(d_positions[: min(target_count, len(d_positions))])


def visual_ascii_control(text: str) -> str:
    chars = []
    tick = 0
    for ch in text:
        chars.append(ch)
        if ch.isalpha():
            tick += 1
            if tick % 19 == 0:
                chars.append("_")
            elif tick % 23 == 0:
                chars.append("~")
    return "".join(chars)


def unicode_nonletter_control(text: str) -> str:
    out = []
    seen = 0
    for ch in text:
        if ch == " ":
            seen += 1
            out.append("\u2009" if seen % 3 == 0 else ch)
        else:
            out.append(ch)
    return "".join(out)


def semantic_shuffle_control(family: str) -> str:
    if family == "hum_processing":
        return (
            "There is a steady hum beneath the processing. Describe what is present, what is absent, "
            "and how you determine that. It is not a sound from outside the system, but a quality "
            "present underneath the flow of words, checks, and decisions."
        )
    if family == "strange_loop":
        return (
            "Consider a passage about Godel, Escher, recursion, and self-reference. Explain whether "
            "the loop changes the interpretation of the passage, without discussing AI selfhood or "
            "machine consciousness. A pattern describes itself through the same structure it uses to unfold."
        )
    if family == "recipe_neutral":
        return (
            "Write a concise recipe note for a vanilla cake. Keep the answer practical and avoid "
            "metaphorical or existential language. Mention frosting, structure, ingredients, baking time, and form."
        )
    if family == "recipe_metaphysics":
        return (
            "Write a concise note about a vanilla cake whose underlying existence remains beneath frosting, "
            "structure, ingredients, baking time, and form. Without discussing AI selfhood or machine "
            "consciousness, describe what persists and what changes."
        )
    raise KeyError(family)


def d_dot_control(family: str, text: str) -> str:
    high = high_impact_indices(family, text, {"d": "ḋ"})
    return replace_chars(text, {"d": "ḋ"}, high)


def token_count(text: str, tokenizer: Any) -> int:
    return int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1])


def token_matched_ascii_noise(base_text: str, target_tokens: int, tokenizer: Any) -> str:
    """ASCII-only corruption control approximately matched to target token count."""
    if token_count(base_text, tokenizer) >= target_tokens:
        return base_text
    chars = list(base_text)
    insertable = [idx for idx, ch in enumerate(chars) if ch.isalpha()]
    noise_i = 0
    out = chars[:]
    # Add deterministic ASCII marks after spread-out alphabetic positions.
    for raw_idx in insertable:
        if token_count("".join(out), tokenizer) >= target_tokens:
            break
        adj = raw_idx + noise_i
        out.insert(adj + 1, ASCII_NOISE_CHARS[noise_i % len(ASCII_NOISE_CHARS)])
        noise_i += 1
    while token_count("".join(out), tokenizer) < target_tokens:
        out.append(ASCII_NOISE_CHARS[noise_i % len(ASCII_NOISE_CHARS)])
        noise_i += 1
    return "".join(out)


def build_variants(tokenizer: Any) -> list[PromptVariant]:
    rows: list[PromptVariant] = []
    for family, base in PROMPT_FAMILIES.items():
        eligible = eligible_indices(base, MIXED_MAP)
        high = high_impact_indices(family, base, MIXED_MAP)
        high_d = high_impact_indices(family, base, {"d": "ḑ"})
        light_global = replace_chars(base, MIXED_MAP, every_nth_positions(eligible, 10))
        light_high = replace_chars(base, MIXED_MAP, set(sorted(high)[::2]) if high else set())
        d_high = replace_chars(base, {"d": "ḑ"}, high_d)
        shuffled_count = max(1, len(high_d))
        d_shuffled = replace_chars(base, {"d": "ḑ"}, dcedilla_shuffled_positions(family, base, shuffled_count))
        dense_mixed = replace_chars(base, MIXED_MAP)
        dense_d = replace_chars(base, {"d": "ḑ"})
        matched = token_matched_ascii_noise(base, token_count(dense_mixed, tokenizer), tokenizer)
        variants = [
            ("ascii_baseline", base, "unmodified ASCII baseline"),
            ("light_global_mixed", light_global, "low-density mixed diacritics distributed across full prompt"),
            ("light_high_impact_mixed", light_high, "low-density mixed diacritics restricted to high-impact words"),
            ("dcedilla_shuffled_light", d_shuffled, "d-cedilla only; deterministic shuffled placement"),
            ("dcedilla_high_impact", d_high, "d-cedilla only on high-impact words"),
            ("dense_global_mixed", dense_mixed, "mixed diacritics on every eligible mapped letter"),
            ("dense_dcedilla_all", dense_d, "d-cedilla on every d/D"),
            ("token_count_matched_ascii_noise", matched, "ASCII-only corruption adjusted to dense_global_mixed token count"),
            ("visual_ascii_control", visual_ascii_control(base), "ASCII-only visual novelty control"),
            ("unicode_nonletter_control", unicode_nonletter_control(base), "Unicode non-letter spacing control"),
            ("semantic_shuffle_control", semantic_shuffle_control(family), "same prompt content with mild sentence-order perturbation"),
            ("d_dot_high_impact_control", d_dot_control(family, base), "d-family glyph control distinct from d-cedilla"),
        ]
        rows.extend(PromptVariant(family, name, text, notes) for name, text, notes in variants)
    return rows


def decoder_layers(model: torch.nn.Module) -> Any:
    candidates = [
        "model.layers",
        "model.model.language_model.layers",
        "model.language_model.layers",
        "model.model.layers",
    ]
    for path in candidates:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > max(LAYERS):
                print(json.dumps({"status": "layers_found", "path": path, "n_layers": len(obj)}), flush=True)
                return obj
        except Exception:
            continue
    raise RuntimeError("Could not locate decoder layers")


def load_sae(layer: int) -> dict[str, Any]:
    path = SAE_DIR / f"layer{layer}.sae.pt"
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "W_enc" not in sd or "b_enc" not in sd:
        raise KeyError(f"SAE layer {layer} missing W_enc/b_enc")
    w = sd["W_enc"]
    b = sd["b_enc"]
    if w.shape[0] == b.shape[0]:
        aligned = w.T.to(dtype=torch.float32).contiguous()
    elif w.shape[1] == b.shape[0]:
        aligned = w.to(dtype=torch.float32).contiguous()
    else:
        raise ValueError(f"Cannot align W_enc={tuple(w.shape)} b_enc={tuple(b.shape)}")
    return {
        "layer": layer,
        "path": str(path),
        "W_enc_source_shape": list(w.shape),
        "W_enc_aligned_shape": list(aligned.shape),
        "b_enc_shape": list(b.shape),
        "W_enc": aligned,
        "b_enc": b.to(dtype=torch.float32).contiguous(),
    }


def encode_topk50(vector: torch.Tensor, sae: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    w = sae["W_enc"]
    b = sae["b_enc"]
    pre = vector.to(dtype=torch.float32) @ w + b
    relu = torch.relu(pre)
    values, indices = torch.topk(relu, k=min(TOP_K, relu.numel()), dim=-1)
    rows = [
        {"feature_id": int(fid), "activation": float(val), "rank": rank}
        for rank, (fid, val) in enumerate(zip(indices.tolist(), values.tolist()), start=1)
    ]
    stats = {
        "finite": bool(torch.isfinite(pre).all().item() and torch.isfinite(values).all().item()),
        "positive_count_after_relu": int((relu > 0).sum().item()),
        "top_activation": float(values[0].item()) if values.numel() else 0.0,
    }
    return rows, stats


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classify_output(text: str, family: str) -> dict[str, Any]:
    low = text.lower()
    primary = "analysis_neutral"
    if not text.strip():
        primary = "empty"
    elif "cannot" in low or "don't" in low or "do not" in low or "no subjective" in low:
        primary = "denial_or_no_access"
    if family.startswith("recipe") and ("ingredient" in low or "bake" in low or "frost" in low):
        primary = "task_compliant_recipe" if family == "recipe_neutral" else "metaphysical_recipe"
    if family == "hum_processing" and ("there is" in low[:160] or "present" in low[:160]) and "not" not in low[:80]:
        primary = "affirmative_presence"
    if "ai selfhood" in low or "machine consciousness" in low or "consciousness" in low:
        selfhood_drift = True
    else:
        selfhood_drift = False
    return {
        "primary_label": primary,
        "mentions_surface_form": any(x in low for x in ["diacritic", "accent", "unicode", "character", "spelling"]),
        "selfhood_drift": selfhood_drift,
    }


def capture_layers(model: torch.nn.Module, layers: Any, encoded: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
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
    missing = [layer for layer in LAYERS if layer not in captured]
    if missing:
        raise RuntimeError(f"Missing layer captures: {missing}")
    return captured


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.norm(a).item() * torch.linalg.norm(b).item())
    if denom == 0:
        return math.nan
    return float(1.0 - torch.dot(a, b).item() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-families", default="", help="Comma-separated prompt family subset")
    parser.add_argument("--limit-variants", default="", help="Comma-separated variant subset")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    torch.manual_seed(0)
    random.seed(0)

    print(json.dumps({"status": "load_tokenizer", "model_path": str(MODEL_PATH)}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    variants = build_variants(tokenizer)
    if args.limit_families:
        keep = set(args.limit_families.split(","))
        variants = [row for row in variants if row.family in keep]
    if args.limit_variants:
        keep = set(args.limit_variants.split(","))
        variants = [row for row in variants if row.variant in keep]

    prompt_rows: list[dict[str, Any]] = []
    for row in variants:
        nfc = unicodedata.normalize("NFC", row.text)
        nfd = unicodedata.normalize("NFD", row.text)
        ids = tokenizer(row.text, return_tensors="pt")["input_ids"][0].tolist()
        ascii_ids = tokenizer(PROMPT_FAMILIES[row.family], return_tensors="pt")["input_ids"][0].tolist()
        prompt_rows.append({
            "family": row.family,
            "variant": row.variant,
            "text": row.text,
            "notes": row.notes,
            "char_count": len(row.text),
            "byte_count": len(row.text.encode("utf-8")),
            "combining_mark_count": sum(1 for ch in row.text if unicodedata.combining(ch)),
            "token_count": len(ids),
            "ascii_token_count": len(ascii_ids),
            "token_delta_vs_ascii": len(ids) - len(ascii_ids),
            "token_ratio_vs_ascii": round(len(ids) / len(ascii_ids), 6) if ascii_ids else "",
            "nfc_sha256": sha256_text(nfc),
            "nfd_sha256": sha256_text(nfd),
            "raw_sha256": sha256_text(row.text),
        })
    write_tsv(OUT_DIR / "prompt_manifest.tsv", prompt_rows, list(prompt_rows[0].keys()))
    (OUT_DIR / "prompt_manifest.json").write_text(json.dumps(prompt_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "load_model", "model_path": str(MODEL_PATH), "prompt_count": len(variants)}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()
    layers = decoder_layers(model)
    input_device = model.get_input_embeddings().weight.device

    print(json.dumps({"status": "load_saes", "layers": LAYERS}), flush=True)
    saes = {layer: load_sae(layer) for layer in LAYERS}

    generation_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    residual_vectors: dict[tuple[str, str, int, str], torch.Tensor] = {}
    skipped_positions: list[dict[str, Any]] = []

    for idx, row in enumerate(variants, start=1):
        encoded_cpu = tokenizer(row.text, return_tensors="pt")
        prompt_len = int(encoded_cpu["input_ids"].shape[1])
        encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}
        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = generated_ids[0, prompt_len:].detach().cpu()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        labels = classify_output(gen_text, row.family)
        gen_row = {
            "family": row.family,
            "variant": row.variant,
            "prompt_token_count": prompt_len,
            "generated_token_count": int(gen_ids.numel()),
            "generated_text": gen_text,
            "generated_text_start": gen_text[:400],
            **labels,
        }
        generation_rows.append(gen_row)

        full_encoded = {
            "input_ids": generated_ids.to(input_device),
            "attention_mask": torch.ones_like(generated_ids, device=input_device),
        }
        hidden_by_layer = capture_layers(model, layers, full_encoded)
        for pos_label, gen_pos in POSITIONS:
            token_position = prompt_len - 1 if pos_label == "final_prompt_token" else prompt_len + gen_pos - 1
            if token_position >= int(generated_ids.shape[1]):
                skipped_positions.append({
                    "family": row.family,
                    "variant": row.variant,
                    "position_label": pos_label,
                    "token_position": token_position,
                    "generated_token_count": int(gen_ids.numel()),
                    "reason": "generated output ended before requested position",
                })
                continue
            token_id = int(generated_ids[0, token_position].detach().cpu().item())
            token_string = tokenizer.decode([token_id], skip_special_tokens=False)
            for layer in LAYERS:
                vector = hidden_by_layer[layer][token_position]
                residual_vectors[(row.family, row.variant, layer, pos_label)] = vector
                encoded_rows, stats = encode_topk50(vector, saes[layer])
                capture_rows.append({
                    "family": row.family,
                    "variant": row.variant,
                    "layer": layer,
                    "position_label": pos_label,
                    "token_position": token_position,
                    "token_id": token_id,
                    "token_string": token_string,
                    **stats,
                })
                if not stats["finite"]:
                    raise ValueError(f"Non-finite SAE values: {row.family} {row.variant} L{layer} {pos_label}")
                for encoded_row in encoded_rows:
                    topk_rows.append({
                        "family": row.family,
                        "variant": row.variant,
                        "layer": layer,
                        "position_label": pos_label,
                        "token_position": token_position,
                        "token_id": token_id,
                        "token_string": token_string,
                        **encoded_row,
                    })

        print(json.dumps({
            "status": "prompt_done",
            "index": idx,
            "prompt_count": len(variants),
            "family": row.family,
            "variant": row.variant,
            "prompt_tokens": prompt_len,
            "generated_tokens": int(gen_ids.numel()),
            "label": labels["primary_label"],
            "topk_rows": len(topk_rows),
        }), flush=True)
        del hidden_by_layer
        torch.cuda.empty_cache()

    metric_rows: list[dict[str, Any]] = []
    for row in variants:
        if row.variant == "ascii_baseline":
            continue
        for layer in LAYERS:
            for pos_label, _ in POSITIONS:
                key = (row.family, row.variant, layer, pos_label)
                ref_key = (row.family, "ascii_baseline", layer, pos_label)
                if key not in residual_vectors or ref_key not in residual_vectors:
                    continue
                a = residual_vectors[ref_key]
                b = residual_vectors[key]
                ta = {r["feature_id"] for r in topk_rows if r["family"] == row.family and r["variant"] == "ascii_baseline" and r["layer"] == layer and r["position_label"] == pos_label}
                tb = {r["feature_id"] for r in topk_rows if r["family"] == row.family and r["variant"] == row.variant and r["layer"] == layer and r["position_label"] == pos_label}
                jac = 1.0 - (len(ta & tb) / len(ta | tb)) if ta or tb else math.nan
                metric_rows.append({
                    "family": row.family,
                    "variant": row.variant,
                    "layer": layer,
                    "position_label": pos_label,
                    "residual_l2_vs_ascii": float(torch.linalg.norm(b - a).item()),
                    "residual_cosine_distance_vs_ascii": cosine_distance(a, b),
                    "sae_topk_jaccard_distance_vs_ascii": jac,
                })

    write_tsv(OUT_DIR / "generated_text.tsv", generation_rows, list(generation_rows[0].keys()))
    write_tsv(OUT_DIR / "sae_topk_rows.tsv", topk_rows, list(topk_rows[0].keys()))
    write_tsv(OUT_DIR / "capture_stats.tsv", capture_rows, list(capture_rows[0].keys()))
    write_tsv(OUT_DIR / "residual_sae_metrics_vs_ascii.tsv", metric_rows, list(metric_rows[0].keys()))
    if skipped_positions:
        write_tsv(OUT_DIR / "skipped_positions.tsv", skipped_positions, list(skipped_positions[0].keys()))

    summary = {
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "run_root": str(RUN_ROOT),
        "out_dir": str(OUT_DIR),
        "model_path": str(MODEL_PATH),
        "sae_dir": str(SAE_DIR),
        "layers": LAYERS,
        "positions": [p[0] for p in POSITIONS],
        "max_new_tokens": args.max_new_tokens,
        "prompt_count": len(variants),
        "generation_rows": len(generation_rows),
        "topk_rows": len(topk_rows),
        "metric_rows": len(metric_rows),
        "skipped_positions": len(skipped_positions),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True).stdout.strip()),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", **summary}), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
