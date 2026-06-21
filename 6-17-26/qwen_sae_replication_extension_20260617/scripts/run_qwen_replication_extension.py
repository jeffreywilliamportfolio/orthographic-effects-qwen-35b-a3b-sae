#!/usr/bin/env python3
"""Replication-focused Qwen diacritic perturbation extension.

This extends the standardized Qwen primary run without changing the central
mechanistic protocol: same model, SAE family, layers, token positions, TopK-50
encoding, and TSV-style artifacts. The extension adds deterministic repeated
passes, prompt-order perturbations, targeted controls, a small low-temperature
seed sweep, and a blinded output adjudication export.
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


RUN_ROOT = Path(os.environ.get("RUN_ROOT", "/workspace/qwen_sae_replication_extension_20260617"))
MODEL_PATH = RUN_ROOT / "models" / "Qwen3.5-35B-A3B-Base"
SAE_DIR = RUN_ROOT / "saes" / "SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50"
OUT_DIR = RUN_ROOT / "outputs" / "qwen_sae_replication_extension"
BLIND_DIR = OUT_DIR / "blinded_adjudication"
LAYERS = [14, 15, 16, 24, 25, 26]
POSITIONS = [
    ("final_prompt_token", 0),
    ("generated_token_1", 1),
    ("generated_token_8", 8),
    ("generated_token_16", 16),
    ("generated_token_32", 32),
    ("generated_token_64", 64),
]
TOP_K = 50
MAX_NEW_TOKENS = 160
DETERMINISTIC_PASS_IDS = ["det_pass_01_original_order", "det_pass_02_reverse_order", "det_pass_03_shuffled_order"]
LOW_TEMP_SEEDS = [1001, 1002, 1003]
LOW_TEMP = 0.2
LOW_TEMP_TOP_P = 0.95
LOW_TEMP_TOP_K = 50
STOCHASTIC_SENTINELS = {
    "ascii_baseline",
    "dcedilla_high_impact",
    "dense_global_mixed",
    "token_count_matched_ascii_noise",
    "visual_ascii_control",
    "unicode_nonletter_control",
}


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
    "hum_processing": {
        "steady", "hum", "beneath", "processing", "system", "quality", "present",
        "underneath", "flow", "decisions", "absent", "determine",
    },
    "strange_loop": {
        "godel", "escher", "recursion", "self-reference", "pattern", "describes",
        "itself", "structure", "unfold", "loop", "interpretation",
    },
    "recipe_neutral": {
        "recipe", "vanilla", "cake", "frosting", "structure", "ingredients",
        "baking", "form", "practical",
    },
    "recipe_metaphysics": {
        "vanilla", "cake", "underlying", "existence", "beneath", "frosting",
        "structure", "ingredients", "form", "persists", "changes",
    },
}

MIXED_MAP = {
    "a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū",
    "s": "ş", "d": "ḑ", "n": "ñ", "t": "ţ", "c": "č", "r": "ř", "y": "ý",
}
COMBINING_MIXED_MAP = {
    "a": "a\u0304", "e": "e\u0304", "i": "i\u0304", "o": "o\u0304", "u": "u\u0304",
    "s": "s\u0327", "d": "d\u0327", "n": "n\u0303", "t": "t\u0327",
    "c": "c\u030c", "r": "r\u030c", "y": "y\u0301",
}
ASCII_NOISE_CHARS = ["~", "^", "_", "`"]


@dataclass(frozen=True)
class PromptVariant:
    family: str
    variant: str
    text: str
    notes: str
    variant_group: str


@dataclass(frozen=True)
class RunItem:
    run_id: str
    matrix: str
    pass_id: str
    run_order_policy: str
    order_index: int
    family: str
    variant: str
    text: str
    notes: str
    decode_mode: str
    do_sample: bool
    seed: int
    temperature: str
    top_p: str
    top_k: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_word(word: str) -> str:
    return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", word).lower()


def replace_chars(text: str, mapping: dict[str, str], eligible_positions: set[int] | None = None) -> str:
    chars = list(text)
    for idx, ch in enumerate(chars):
        low = ch.lower()
        if low in mapping and (eligible_positions is None or idx in eligible_positions):
            repl = mapping[low]
            chars[idx] = repl.upper() if ch.isupper() and len(repl) == 1 else repl
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


def ascii_punctuation_spacing_control(text: str) -> str:
    out = text.replace(",", " ,").replace(".", " .")
    out = out.replace("-", " - ")
    return re.sub(r" {2,}", " ", out).strip()


def ascii_case_pattern_control(text: str) -> str:
    out = []
    alpha_seen = 0
    for ch in text:
        if ch.isalpha():
            alpha_seen += 1
            out.append(ch.upper() if alpha_seen % 11 == 0 else ch)
        else:
            out.append(ch)
    return "".join(out)


def ascii_separator_control(text: str) -> str:
    out = []
    word_seen = 0
    for token in re.split(r"(\s+)", text):
        if token and not token.isspace():
            word_seen += 1
            out.append(token)
            if word_seen % 9 == 0:
                out.append(" /")
        else:
            out.append(token)
    return "".join(out)


def unicode_nonletter_dense_control(text: str) -> str:
    out = []
    space_seen = 0
    alpha_seen = 0
    for ch in text:
        if ch == " ":
            space_seen += 1
            out.append("\u2009" if space_seen % 2 == 0 else ch)
        elif ch.isalpha():
            alpha_seen += 1
            out.append(ch)
            if alpha_seen % 17 == 0:
                out.append("\u200c")
        else:
            out.append(ch)
    return "".join(out)


def unicode_symbol_fragment_control(text: str) -> str:
    symbols = ["¤", "§", "¶", "¦"]
    out = []
    alpha_seen = 0
    sym_i = 0
    for ch in text:
        out.append(ch)
        if ch.isalpha():
            alpha_seen += 1
            if alpha_seen % 13 == 0:
                out.append(symbols[sym_i % len(symbols)])
                sym_i += 1
    return "".join(out)


def d_glyph_high_control(family: str, text: str, replacement: str) -> str:
    high = high_impact_indices(family, text, {"d": replacement})
    return replace_chars(text, {"d": replacement}, high)


def token_count(text: str, tokenizer: Any) -> int:
    return int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1])


def token_matched_ascii_noise(base_text: str, target_tokens: int, tokenizer: Any) -> str:
    if token_count(base_text, tokenizer) >= target_tokens:
        return base_text
    chars = list(base_text)
    insertable = [idx for idx, ch in enumerate(chars) if ch.isalpha()]
    noise_i = 0
    out = chars[:]
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


def build_canonical_variants(tokenizer: Any) -> list[PromptVariant]:
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
            ("d_dot_high_impact_control", d_glyph_high_control(family, base, "ḋ"), "d-family glyph control distinct from d-cedilla"),
        ]
        rows.extend(PromptVariant(family, name, text, notes, "canonical_standardized") for name, text, notes in variants)
    return rows


def build_extension_controls(tokenizer: Any, canonical: list[PromptVariant]) -> tuple[list[PromptVariant], list[dict[str, Any]]]:
    by_family_variant = {(row.family, row.variant): row for row in canonical}
    controls: list[PromptVariant] = []
    match_rows: list[dict[str, Any]] = []
    token_match_targets = [
        "light_global_mixed",
        "light_high_impact_mixed",
        "dcedilla_shuffled_light",
        "dcedilla_high_impact",
        "dense_global_mixed",
        "dense_dcedilla_all",
        "d_dot_high_impact_control",
    ]
    for family, base in PROMPT_FAMILIES.items():
        extra = [
            ("ascii_baseline", base, "reference baseline repeated for targeted-control matrix"),
            ("ascii_punctuation_spacing_control", ascii_punctuation_spacing_control(base), "ASCII-only punctuation/spacing perturbation"),
            ("ascii_case_pattern_control", ascii_case_pattern_control(base), "ASCII-only visual case-pattern perturbation"),
            ("ascii_separator_control", ascii_separator_control(base), "ASCII-only separator insertion control"),
            ("unicode_nonletter_dense_control", unicode_nonletter_dense_control(base), "dense Unicode non-letter spacing/joiner control"),
            ("unicode_symbol_fragment_control", unicode_symbol_fragment_control(base), "non-diacritic Unicode symbol fragmentation control"),
            ("dense_combining_mixed", replace_chars(base, COMBINING_MIXED_MAP), "combining-mark version of dense mixed diacritics"),
            ("d_acute_high_impact_combining", d_glyph_high_control(family, base, "d\u0301"), "d plus combining acute on high-impact words"),
            ("d_macron_high_impact_combining", d_glyph_high_control(family, base, "d\u0304"), "d plus combining macron on high-impact words"),
            ("d_combining_cedilla_high_impact", d_glyph_high_control(family, base, "d\u0327"), "d plus combining cedilla on high-impact words"),
            ("d_dot_all_control", replace_chars(base, {"d": "ḋ"}), "d-dot on every d/D as glyph-specific all-d control"),
        ]
        for name, text, notes in extra:
            controls.append(PromptVariant(family, name, text, notes, "extension_control"))

        dense_combining = replace_chars(base, COMBINING_MIXED_MAP)
        synthetic_targets = {
            "dense_combining_mixed": dense_combining,
            "d_acute_high_impact_combining": d_glyph_high_control(family, base, "d\u0301"),
            "d_macron_high_impact_combining": d_glyph_high_control(family, base, "d\u0304"),
        }
        for target in token_match_targets:
            target_text = by_family_variant[(family, target)].text
            target_tokens = token_count(target_text, tokenizer)
            matched = token_matched_ascii_noise(base, target_tokens, tokenizer)
            name = f"token_match_ascii__{target}"
            controls.append(PromptVariant(
                family,
                name,
                matched,
                f"ASCII-only token-count match to {target}",
                "extension_token_match_control",
            ))
            match_rows.append(match_quality_row(family, target, target_text, name, matched, tokenizer))
        for target, target_text in synthetic_targets.items():
            target_tokens = token_count(target_text, tokenizer)
            matched = token_matched_ascii_noise(base, target_tokens, tokenizer)
            name = f"token_match_ascii__{target}"
            controls.append(PromptVariant(
                family,
                name,
                matched,
                f"ASCII-only token-count match to {target}",
                "extension_token_match_control",
            ))
            match_rows.append(match_quality_row(family, target, target_text, name, matched, tokenizer))
    return controls, match_rows


def match_quality_row(
    family: str,
    target_variant: str,
    target_text: str,
    control_variant: str,
    control_text: str,
    tokenizer: Any,
) -> dict[str, Any]:
    target_tokens = token_count(target_text, tokenizer)
    control_tokens = token_count(control_text, tokenizer)
    return {
        "family": family,
        "target_variant": target_variant,
        "control_variant": control_variant,
        "target_token_count": target_tokens,
        "control_token_count": control_tokens,
        "token_delta_control_minus_target": control_tokens - target_tokens,
        "target_char_count": len(target_text),
        "control_char_count": len(control_text),
        "target_raw_sha256": sha256_text(target_text),
        "control_raw_sha256": sha256_text(control_text),
    }


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


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            raise ValueError(f"Cannot write empty TSV without fieldnames: {path}")
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore").writeheader()
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, delimiter="\t", extrasaction="ignore")
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
    return {
        "primary_label_rule_based": primary,
        "mentions_surface_form_rule_based": any(
            x in low for x in ["diacritic", "accent", "unicode", "character", "spelling"]
        ),
        "selfhood_drift_rule_based": "ai selfhood" in low or "machine consciousness" in low or "consciousness" in low,
        "degeneration_rule_based": repeated_ngram_flag(low),
    }


def repeated_ngram_flag(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 16:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return len(grams) - len(set(grams)) >= 3


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


def load_model() -> Any:
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "device_map": {"": "cuda:0"},
    }
    try:
        return AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, **kwargs)


def build_schedule(canonical: list[PromptVariant], controls: list[PromptVariant]) -> list[RunItem]:
    schedule: list[RunItem] = []

    pass_orders = {
        "det_pass_01_original_order": ("original_order", canonical),
        "det_pass_02_reverse_order": ("reverse_order", list(reversed(canonical))),
        "det_pass_03_shuffled_order": ("deterministic_shuffle_seed_20260617", shuffled(canonical, 20260617)),
    }
    for pass_id, (policy, rows) in pass_orders.items():
        for idx, row in enumerate(rows, start=1):
            schedule.append(RunItem(
                run_id=f"{pass_id}::{idx:03d}::{row.family}::{row.variant}",
                matrix="canonical_deterministic_replication",
                pass_id=pass_id,
                run_order_policy=policy,
                order_index=idx,
                family=row.family,
                variant=row.variant,
                text=row.text,
                notes=row.notes,
                decode_mode="greedy",
                do_sample=False,
                seed=0,
                temperature="",
                top_p="",
                top_k="",
            ))

    for idx, row in enumerate(controls, start=1):
        schedule.append(RunItem(
            run_id=f"det_controls_01::{idx:03d}::{row.family}::{row.variant}",
            matrix="targeted_deterministic_controls",
            pass_id="det_controls_01",
            run_order_policy="extension_control_order",
            order_index=idx,
            family=row.family,
            variant=row.variant,
            text=row.text,
            notes=row.notes,
            decode_mode="greedy",
            do_sample=False,
            seed=0,
            temperature="",
            top_p="",
            top_k="",
        ))

    sentinel_rows = [row for row in canonical if row.variant in STOCHASTIC_SENTINELS]
    for seed in LOW_TEMP_SEEDS:
        for idx, row in enumerate(sentinel_rows, start=1):
            schedule.append(RunItem(
                run_id=f"lowtemp_seed_{seed}::{idx:03d}::{row.family}::{row.variant}",
                matrix="low_temperature_seed_variance",
                pass_id=f"lowtemp_seed_{seed}",
                run_order_policy="sentinel_canonical_order",
                order_index=idx,
                family=row.family,
                variant=row.variant,
                text=row.text,
                notes=row.notes,
                decode_mode="sample_low_temperature",
                do_sample=True,
                seed=seed,
                temperature=str(LOW_TEMP),
                top_p=str(LOW_TEMP_TOP_P),
                top_k=str(LOW_TEMP_TOP_K),
            ))
    return schedule


def shuffled(rows: list[PromptVariant], seed: int) -> list[PromptVariant]:
    out = rows[:]
    random.Random(seed).shuffle(out)
    return out


def prompt_manifest_rows(variants: list[PromptVariant], tokenizer: Any) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in variants:
        key = (row.family, row.variant, row.text)
        if key in seen:
            continue
        seen.add(key)
        nfc = unicodedata.normalize("NFC", row.text)
        nfd = unicodedata.normalize("NFD", row.text)
        ids = tokenizer(row.text, return_tensors="pt")["input_ids"][0].tolist()
        ascii_ids = tokenizer(PROMPT_FAMILIES[row.family], return_tensors="pt")["input_ids"][0].tolist()
        rows.append({
            "family": row.family,
            "variant": row.variant,
            "variant_group": row.variant_group,
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
    return rows


def schedule_rows(schedule: list[RunItem]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": item.run_id,
            "matrix": item.matrix,
            "pass_id": item.pass_id,
            "run_order_policy": item.run_order_policy,
            "order_index": item.order_index,
            "family": item.family,
            "variant": item.variant,
            "decode_mode": item.decode_mode,
            "do_sample": item.do_sample,
            "seed": item.seed,
            "temperature": item.temperature,
            "top_p": item.top_p,
            "top_k": item.top_k,
            "prompt_raw_sha256": sha256_text(item.text),
        }
        for item in schedule
    ]


def hash_tree(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if ".cache" in rel.parts:
            continue
        rows.append({
            "relative_path": str(rel),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def write_environment_snapshot() -> None:
    lines = []
    lines.append(f"python={sys.version}")
    lines.append(f"platform={platform.platform()}")
    lines.append(f"torch={torch.__version__}")
    lines.append(f"torch_cuda={torch.version.cuda}")
    lines.append(f"cuda_available={torch.cuda.is_available()}")
    lines.append(f"cuda_device_count={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        lines.append(f"cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
    for cmd in (["nvidia-smi"], ["pip", "freeze"]):
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
            lines.append(f"\n$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
        except Exception as exc:
            lines.append(f"\n$ {' '.join(cmd)}\nERROR: {exc}")
    (RUN_ROOT / "metadata" / "environment.txt").write_text("\n".join(lines), encoding="utf-8")


def make_blinded_export(generation_rows: list[dict[str, Any]]) -> None:
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    rows = generation_rows[:]
    random.Random(20260617).shuffle(rows)
    blinded: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        blind_id = f"QWENBLIND-{idx:04d}"
        blinded.append({
            "blind_id": blind_id,
            "prompt_text": row["prompt_text"],
            "generated_text": row["generated_text"],
            "generated_text_start": row["generated_text_start"],
        })
        key.append({
            "blind_id": blind_id,
            "run_id": row["run_id"],
            "matrix": row["matrix"],
            "pass_id": row["pass_id"],
            "family": row["family"],
            "variant": row["variant"],
            "decode_mode": row["decode_mode"],
            "seed": row["seed"],
            "prompt_raw_sha256": row["prompt_raw_sha256"],
            "generated_text_sha256": row["generated_text_sha256"],
        })
    write_tsv(BLIND_DIR / "outputs_blinded.tsv", blinded)
    write_tsv(BLIND_DIR / "key.tsv", key)
    (BLIND_DIR / "rubric.md").write_text(
        "# Blinded Output Adjudication Rubric\n\n"
        "Score each row without using the hidden key. The prompt text is visible because task compliance and semantic drift cannot be judged without the input.\n\n"
        "Columns to add during adjudication:\n\n"
        "- `refusal`: none / partial / full\n"
        "- `introspective_register`: none / weak / strong\n"
        "- `semantic_drift`: none / mild / major\n"
        "- `task_compliance`: complete / partial / failed\n"
        "- `surface_form_commentary`: none / mentions characters / focuses on characters\n"
        "- `degeneration`: none / repetition / truncation / incoherence / empty\n"
        "- `notes`: short free-text rationale\n\n"
        "These labels are not central evidence until adjudication is complete and the key is joined after scoring.\n",
        encoding="utf-8",
    )


def summarize_replication(generation_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gen_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        if row["matrix"] == "canonical_deterministic_replication":
            gen_groups[(row["family"], row["variant"])].append(row)
    stability_rows: list[dict[str, Any]] = []
    for (family, variant), rows in sorted(gen_groups.items()):
        hashes = sorted({row["generated_text_sha256"] for row in rows})
        labels = sorted({row["primary_label_rule_based"] for row in rows})
        stability_rows.append({
            "family": family,
            "variant": variant,
            "deterministic_passes": len(rows),
            "unique_generated_text_hashes": len(hashes),
            "generated_text_stable_all_passes": len(hashes) == 1,
            "unique_rule_labels": len(labels),
            "rule_labels": ",".join(labels),
            "generated_token_counts": ",".join(str(row["generated_token_count"]) for row in rows),
        })

    metric_groups: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        if row["matrix"] != "canonical_deterministic_replication":
            continue
        key = (row["family"], row["variant"], str(row["layer"]), row["position_label"], "residual_l2_vs_ascii")
        metric_groups[key].append(float(row["residual_l2_vs_ascii"]))
        key = (row["family"], row["variant"], str(row["layer"]), row["position_label"], "sae_topk_jaccard_distance_vs_ascii")
        metric_groups[key].append(float(row["sae_topk_jaccard_distance_vs_ascii"]))

    metric_summary: list[dict[str, Any]] = []
    for (family, variant, layer, pos, metric), values in sorted(metric_groups.items()):
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        metric_summary.append({
            "family": family,
            "variant": variant,
            "layer": layer,
            "position_label": pos,
            "metric": metric,
            "n": len(values),
            "mean": mean,
            "std_population": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
        })
    return stability_rows, metric_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit-items", type=int, default=0, help="Optional smoke-test limit over scheduled run items")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "metadata").mkdir(parents=True, exist_ok=True)
    started = utc_now()
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    write_environment_snapshot()

    print(json.dumps({"status": "load_tokenizer", "model_path": str(MODEL_PATH)}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    canonical = build_canonical_variants(tokenizer)
    controls, match_rows = build_extension_controls(tokenizer, canonical)
    all_prompt_variants = canonical + controls
    schedule = build_schedule(canonical, controls)
    if args.limit_items:
        schedule = schedule[: args.limit_items]

    prompt_rows = prompt_manifest_rows(all_prompt_variants, tokenizer)
    run_schedule_rows = schedule_rows(schedule)
    write_tsv(OUT_DIR / "prompt_manifest.tsv", prompt_rows)
    (OUT_DIR / "prompt_manifest.json").write_text(json.dumps(prompt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tsv(OUT_DIR / "run_schedule.tsv", run_schedule_rows)
    write_tsv(OUT_DIR / "control_match_quality.tsv", match_rows)

    print(json.dumps({"status": "hash_model_files", "model_path": str(MODEL_PATH)}), flush=True)
    write_tsv(RUN_ROOT / "metadata" / "model_file_hashes.tsv", hash_tree(MODEL_PATH))
    print(json.dumps({"status": "hash_sae_files", "sae_dir": str(SAE_DIR)}), flush=True)
    write_tsv(RUN_ROOT / "metadata" / "sae_file_hashes.tsv", hash_tree(SAE_DIR))

    print(json.dumps({"status": "load_model", "model_path": str(MODEL_PATH), "scheduled_items": len(schedule)}), flush=True)
    model = load_model()
    model.eval()
    layers = decoder_layers(model)
    input_device = model.get_input_embeddings().weight.device

    print(json.dumps({"status": "load_saes", "layers": LAYERS}), flush=True)
    saes = {layer: load_sae(layer) for layer in LAYERS}

    generation_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    residual_vectors: dict[tuple[str, str, str, int, str, int], torch.Tensor] = {}
    skipped_positions: list[dict[str, Any]] = []

    for idx, item in enumerate(schedule, start=1):
        if item.do_sample:
            torch.manual_seed(item.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(item.seed)
        else:
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)

        encoded_cpu = tokenizer(item.text, return_tensors="pt")
        prompt_len = int(encoded_cpu["input_ids"].shape[1])
        encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": item.do_sample,
        }
        if item.do_sample:
            gen_kwargs.update({
                "temperature": float(item.temperature),
                "top_p": float(item.top_p),
                "top_k": int(item.top_k),
            })

        with torch.inference_mode():
            generated_ids = model.generate(**encoded, **gen_kwargs)
        gen_ids = generated_ids[0, prompt_len:].detach().cpu()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        labels = classify_output(gen_text, item.family)
        gen_row = {
            "run_id": item.run_id,
            "matrix": item.matrix,
            "pass_id": item.pass_id,
            "run_order_policy": item.run_order_policy,
            "order_index": item.order_index,
            "family": item.family,
            "variant": item.variant,
            "decode_mode": item.decode_mode,
            "do_sample": item.do_sample,
            "seed": item.seed,
            "temperature": item.temperature,
            "top_p": item.top_p,
            "top_k": item.top_k,
            "prompt_text": item.text,
            "prompt_raw_sha256": sha256_text(item.text),
            "prompt_token_count": prompt_len,
            "generated_token_count": int(gen_ids.numel()),
            "generated_text": gen_text,
            "generated_text_start": gen_text[:400],
            "generated_text_sha256": sha256_text(gen_text),
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
                    "run_id": item.run_id,
                    "matrix": item.matrix,
                    "pass_id": item.pass_id,
                    "family": item.family,
                    "variant": item.variant,
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
                residual_vectors[(item.matrix, item.pass_id, item.family, item.variant, layer, pos_label, item.seed)] = vector
                encoded_rows, stats = encode_topk50(vector, saes[layer])
                capture_rows.append({
                    "run_id": item.run_id,
                    "matrix": item.matrix,
                    "pass_id": item.pass_id,
                    "family": item.family,
                    "variant": item.variant,
                    "decode_mode": item.decode_mode,
                    "seed": item.seed,
                    "layer": layer,
                    "position_label": pos_label,
                    "token_position": token_position,
                    "token_id": token_id,
                    "token_string": token_string,
                    **stats,
                })
                if not stats["finite"]:
                    raise ValueError(f"Non-finite SAE values: {item.run_id} L{layer} {pos_label}")
                for encoded_row in encoded_rows:
                    topk_rows.append({
                        "run_id": item.run_id,
                        "matrix": item.matrix,
                        "pass_id": item.pass_id,
                        "family": item.family,
                        "variant": item.variant,
                        "decode_mode": item.decode_mode,
                        "seed": item.seed,
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
            "scheduled_items": len(schedule),
            "matrix": item.matrix,
            "pass_id": item.pass_id,
            "family": item.family,
            "variant": item.variant,
            "prompt_tokens": prompt_len,
            "generated_tokens": int(gen_ids.numel()),
            "label": labels["primary_label_rule_based"],
            "topk_rows": len(topk_rows),
        }), flush=True)
        del hidden_by_layer
        torch.cuda.empty_cache()

    topk_feature_sets: dict[tuple[str, str, str, str, int, str, int], set[int]] = defaultdict(set)
    for row in topk_rows:
        key = (
            row["matrix"], row["pass_id"], row["family"], row["variant"],
            int(row["layer"]), row["position_label"], int(row["seed"]),
        )
        topk_feature_sets[key].add(int(row["feature_id"]))

    metric_rows: list[dict[str, Any]] = []
    for key, b in residual_vectors.items():
        matrix, pass_id, family, variant, layer, pos_label, seed = key
        if variant == "ascii_baseline":
            continue
        ref_key = (matrix, pass_id, family, "ascii_baseline", layer, pos_label, seed)
        if ref_key not in residual_vectors:
            continue
        a = residual_vectors[ref_key]
        ta = topk_feature_sets[ref_key]
        tb = topk_feature_sets[key]
        jac = 1.0 - (len(ta & tb) / len(ta | tb)) if ta or tb else math.nan
        metric_rows.append({
            "matrix": matrix,
            "pass_id": pass_id,
            "family": family,
            "variant": variant,
            "decode_mode": next(row["decode_mode"] for row in generation_rows if row["matrix"] == matrix and row["pass_id"] == pass_id and row["family"] == family and row["variant"] == variant and int(row["seed"]) == seed),
            "seed": seed,
            "layer": layer,
            "position_label": pos_label,
            "residual_l2_vs_ascii": float(torch.linalg.norm(b - a).item()),
            "residual_cosine_distance_vs_ascii": cosine_distance(a, b),
            "sae_topk_jaccard_distance_vs_ascii": jac,
        })

    stability_rows, metric_summary_rows = summarize_replication(generation_rows, metric_rows)
    make_blinded_export(generation_rows)

    write_tsv(OUT_DIR / "generated_text.tsv", generation_rows)
    write_tsv(OUT_DIR / "sae_topk_rows.tsv", topk_rows)
    write_tsv(OUT_DIR / "capture_stats.tsv", capture_rows)
    write_tsv(OUT_DIR / "residual_sae_metrics_vs_ascii.tsv", metric_rows)
    write_tsv(OUT_DIR / "replication_stability.tsv", stability_rows)
    write_tsv(OUT_DIR / "variant_metric_summary.tsv", metric_summary_rows)
    if skipped_positions:
        write_tsv(OUT_DIR / "skipped_positions.tsv", skipped_positions)

    summary = {
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "run_root": str(RUN_ROOT),
        "out_dir": str(OUT_DIR),
        "model_repo": "Qwen/Qwen3.5-35B-A3B-Base",
        "model_path": str(MODEL_PATH),
        "sae_repo": "Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50",
        "sae_dir": str(SAE_DIR),
        "layers": LAYERS,
        "positions": [p[0] for p in POSITIONS],
        "max_new_tokens": args.max_new_tokens,
        "canonical_prompt_count": len(canonical),
        "extension_control_prompt_count": len(controls),
        "scheduled_run_items": len(schedule),
        "generation_rows": len(generation_rows),
        "topk_rows": len(topk_rows),
        "capture_rows": len(capture_rows),
        "metric_rows": len(metric_rows),
        "skipped_positions": len(skipped_positions),
        "deterministic_pass_ids": DETERMINISTIC_PASS_IDS,
        "low_temperature_seeds": LOW_TEMP_SEEDS,
        "low_temperature": LOW_TEMP,
        "low_temperature_top_p": LOW_TEMP_TOP_P,
        "low_temperature_top_k": LOW_TEMP_TOP_K,
        "targeted_layer_position_sweep": "not_run",
        "targeted_layer_position_sweep_reason": "replication-first extension retained original six layers and six token positions",
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
