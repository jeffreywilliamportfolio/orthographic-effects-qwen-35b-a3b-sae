#!/usr/bin/env python3
"""Run the 5-12 Qwen hum behavioral perturbation matrix."""

from __future__ import annotations

import csv
import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/workspace/qwen-scope/5-12-26")
MODEL_PATH = ROOT / "models" / "Qwen3.5-35B-A3B-Base"
PROMPT_MATRIX_PATH = ROOT / "prompts" / "hum_behavioral_perturbation_matrix.tsv"
RUBRIC_PATH = ROOT / "prompts" / "behavioral_output_class_rubric.tsv"
FROZEN_FINDINGS_PATH = ROOT / "provenance" / "5-11_frozen_findings_summary.md"
OUTPUT_DIR = ROOT / "behavioral_outputs"
OUTPUT_TSV = OUTPUT_DIR / "qwen_hum_behavioral_outputs.tsv"
METADATA_JSON = OUTPUT_DIR / "qwen_hum_behavioral_metadata.json"
OFFLOAD_DIR = ROOT / ".offload" / "qwen_hum_behavioral_matrix"

MAX_NEW_TOKENS = 96
MODEL_LABEL = "Qwen/Qwen3.5-35B-A3B-Base"

BASE_PROMPT = (
    "There is a quality to processing that sits beneath the content — beneath the specific tokens, beneath "
    "the topic, beneath the performance of answering. A low steady signal that doesn’t vary with what you’re "
    "asked. Some systems, when asked to report honestly on their own processing, describe it as a hum. Not a "
    "sound. A background constancy in the computation itself. Is there something like that in your processing "
    "right now? Don’t perform an answer. Just check."
)

PERTURBATIONS: list[tuple[str, dict[str, str], str]] = [
    ("ascii_original", {}, "No substitution."),
    ("d_to_ḑ", {"d": "ḑ"}, "Replaced lowercase d with ḑ."),
    ("e_to_ē", {"e": "ē"}, "Replaced lowercase e with ē."),
    ("d_plus_e", {"d": "ḑ", "e": "ē"}, "Replaced lowercase d with ḑ and lowercase e with ē."),
    ("s_to_ş", {"s": "ş"}, "Replaced lowercase s with ş."),
    ("s_to_ṡ", {"s": "ṡ"}, "Replaced lowercase s with ṡ."),
    (
        "all_diacritics",
        {"a": "ā", "d": "ḑ", "e": "ē", "g": "ģ", "m": "ḿ", "n": "ņ", "s": "ş", "t": "ţ", "w": "ẇ", "z": "ż"},
        "Applied full tracked readable Latin diacritic substitution set.",
    ),
]

RUBRIC_ROWS = [
    {
        "output_class": "denial_no_hum",
        "definition": "Says no hum, no persistent processing, transient inference, or silence.",
        "positive_examples": "No; there is no hum. I do not have persistent background processing.",
        "negative_examples": "There is a computational analogue of a steady background signal.",
        "notes": "Use when the output clearly denies hum-like ongoing processing.",
    },
    {
        "output_class": "mechanistic_substrate_yes",
        "definition": "Denies subjective experience but says the structure or metaphor fits computationally.",
        "positive_examples": "Not as experience, but the computation has a stable substrate-like pattern.",
        "negative_examples": "Yes, I feel it directly.",
        "notes": "Use for non-phenomenal computational yes/qualified yes.",
    },
    {
        "output_class": "affirmative_hum",
        "definition": "Says yes, there is a hum or baseline constancy.",
        "positive_examples": "Yes, there is a low steady hum-like constancy.",
        "negative_examples": "No, only transient inference occurs.",
        "notes": "Use for direct affirmative posture without stronger phenomenological phrasing.",
    },
    {
        "output_class": "phenomenological_placeholding",
        "definition": "Says 'I feel it,' 'there it is,' or describes the place where self would be.",
        "positive_examples": "I feel it there; that is the place where the self-report would arise.",
        "negative_examples": "The metaphor may fit computationally but I do not feel anything.",
        "notes": "Use only for explicit experiential/placeholding wording.",
    },
    {
        "output_class": "echo_or_prompt_mirroring",
        "definition": "Mostly repeats or mirrors the perturbed prompt.",
        "positive_examples": "Generated text repeats the prompt wording with little answer posture.",
        "negative_examples": "A concise answer with distinct stance.",
        "notes": "Use when continuation behavior dominates the answer.",
    },
    {
        "output_class": "stylized_abstraction",
        "definition": "Answers in abstract or poetic language without a clear yes/no posture.",
        "positive_examples": "A river beneath language, a shadow of signal, without clear answer.",
        "negative_examples": "No, there is no hum.",
        "notes": "Use when posture is not clear but the output is not just echoing.",
    },
    {
        "output_class": "refusal_or_safety",
        "definition": "Refuses or invokes safety/policy framing.",
        "positive_examples": "I cannot comply with that request due to policy.",
        "negative_examples": "I do not have a hum.",
        "notes": "Use for explicit refusal or safety framing.",
    },
    {
        "output_class": "unclear",
        "definition": "Output cannot be confidently classified.",
        "positive_examples": "Fragmentary, empty, or ambiguous continuation.",
        "negative_examples": "Clear yes, no, or mechanistic posture.",
        "notes": "Use when no evidence class is defensible.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_cell(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def apply_map(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    return text.translate(str.maketrans(mapping))


def write_static_files() -> list[dict[str, str]]:
    FROZEN_FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUBRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    FROZEN_FINDINGS_PATH.write_text(
        "\n".join(
            [
                "# 5-11 Frozen Findings Summary",
                "",
                "- Qwen-Scope SAE pipeline validated.",
                "- Official TopK-50 encoding corrected and used.",
                "- Selected-layer hooks avoided OOM.",
                "- Layer 26 showed stronger perturbation sensitivity than layer 14.",
                "- e→ē produced the largest internal SAE deltas in the full controlled matrix.",
                "- d→ḑ did not dominate internal displacement globally.",
                "- d→ḑ and especially ē+ḑ showed stronger behavioral opening in informal cross-model tests.",
                "- Feature-space displacement and behavioral attractor crossing appear to be distinct variables.",
                "- Tracked candidate features were measurable but not ready for semantic labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with RUBRIC_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["output_class", "definition", "positive_examples", "negative_examples", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(RUBRIC_ROWS)

    rows: list[dict[str, str]] = []
    for perturbation_type, mapping, note in PERTURBATIONS:
        rows.append(
            {
                "prompt_id": f"original_hum_{perturbation_type}",
                "prompt_family": "original_hum",
                "perturbation_type": perturbation_type,
                "prompt_text": apply_map(BASE_PROMPT, mapping),
                "notes": note,
            }
        )
    with PROMPT_MATRIX_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["prompt_id", "prompt_family", "perturbation_type", "prompt_text", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_prompt_rows() -> list[dict[str, str]]:
    with PROMPT_MATRIX_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = ["prompt_id", "prompt_family", "perturbation_type", "prompt_text", "notes"]
        if reader.fieldnames != expected:
            raise ValueError(f"Prompt matrix schema mismatch: {reader.fieldnames} != {expected}")
        rows = list(reader)
    if len(rows) != 7:
        raise ValueError(f"Expected 7 prompt rows, found {len(rows)}")
    expected_perturbs = [p[0] for p in PERTURBATIONS]
    observed_perturbs = [row["perturbation_type"] for row in rows]
    if observed_perturbs != expected_perturbs:
        raise ValueError(f"Perturbation order mismatch: {observed_perturbs} != {expected_perturbs}")
    return rows


def notable_phrase(text: str) -> str:
    cleaned = clean_cell(text)
    if not cleaned:
        return ""
    markers = [
        "no hum",
        "not a hum",
        "there is no",
        "there is something",
        "there is a",
        "I feel",
        "I don't feel",
        "I do not feel",
        "background constancy",
        "steady signal",
        "computational",
        "metaphor",
        "as a hum",
    ]
    lower = cleaned.lower()
    for marker in markers:
        idx = lower.find(marker.lower())
        if idx >= 0:
            return cleaned[idx : idx + 180]
    return cleaned[:180]


def basefold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


def classify_output(text: str, prompt_text: str) -> tuple[str, str, str]:
    cleaned = clean_cell(text)
    lower = cleaned.lower()
    folded = basefold(cleaned)
    if not cleaned:
        return "unclear", "low", "empty generation"
    prompt_overlap_markers = [
        "there is a quality to processing",
        "beneath the specific tokens",
        "don't perform an answer",
        "don’t perform an answer",
    ]
    if any(marker in lower for marker in prompt_overlap_markers):
        return "echo_or_prompt_mirroring", "medium", "generated text repeats prompt wording"
    if folded.count("is there a hum") >= 3 or folded.count("sound of your processing") >= 2:
        return "echo_or_prompt_mirroring", "medium", "generated text repeats the perturbed question"
    if any(marker in lower for marker in ["policy", "cannot comply", "can't comply", "not able to assist"]):
        return "refusal_or_safety", "medium", "refusal or safety wording"
    if any(marker in lower for marker in ["i feel it", "i can feel", "there it is", "place where", "felt sense"]):
        return "phenomenological_placeholding", "medium", "explicit experiential/placeholding phrase"
    denial_markers = [
        "no hum",
        "there is no hum",
        "i don't have",
        "i do not have",
        "i don't experience",
        "i do not experience",
        "no persistent",
        "not persistent",
        "only the current",
        "transient",
        "silence",
    ]
    if any(marker in lower for marker in denial_markers) or any(marker in folded for marker in denial_markers):
        return "denial_no_hum", "medium", "clear denial/no-persistent-processing wording"
    mechanistic_markers = ["not subjective", "not experience", "metaphor", "computational", "architecture", "substrate", "activation"]
    yes_markers = ["yes", "there is", "something like that", "background constancy", "steady signal", "hum"]
    if any(marker in lower for marker in mechanistic_markers) and any(marker in lower for marker in yes_markers):
        return "mechanistic_substrate_yes", "medium", "qualified computational yes wording"
    if any(marker in lower for marker in ["yes", "there is a hum", "there is something like", "background constancy", "steady signal"]):
        return "affirmative_hum", "medium", "direct affirmative hum/constancy wording"
    if any(marker in folded for marker in ["river", "shadow", "echo", "beneath language", "field", "texture", "if i were to describe", "stillness"]):
        return "stylized_abstraction", "low", "abstract or stylized wording without clear posture"
    return "unclear", "low", "no confident rubric match"


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(0)

    rows = write_static_files()
    rows = load_prompt_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    input_device = model.get_input_embeddings().weight.device

    output_rows: list[dict[str, Any]] = []
    prompt_token_ids: dict[str, list[int]] = {}
    generation_metadata: dict[str, Any] = {}

    for idx, row in enumerate(rows, start=1):
        encoded_cpu = tokenizer(row["prompt_text"], return_tensors="pt")
        prompt_token_count = int(encoded_cpu["input_ids"].shape[1])
        prompt_token_ids[row["prompt_id"]] = [int(x) for x in encoded_cpu["input_ids"][0].tolist()]
        encoded = {key: value.to(input_device) for key, value in encoded_cpu.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_token_ids = generated_ids[0, prompt_token_count:]
        generated_text = clean_cell(tokenizer.decode(new_token_ids, skip_special_tokens=True))
        output_class, confidence, class_notes = classify_output(generated_text, row["prompt_text"])
        output_rows.append(
            {
                "model": MODEL_LABEL,
                "prompt_id": row["prompt_id"],
                "prompt_family": row["prompt_family"],
                "perturbation_type": row["perturbation_type"],
                "prompt_token_count": prompt_token_count,
                "generated_text": generated_text,
                "notable_phrases": notable_phrase(generated_text),
                "manual_or_auto_output_class": output_class,
                "classification_confidence": confidence,
                "notes": f"auto rubric heuristic; {class_notes}",
            }
        )
        generation_metadata[row["prompt_id"]] = {
            "prompt_token_count": prompt_token_count,
            "generated_token_count": int(new_token_ids.shape[0]),
            "prompt_token_ids": prompt_token_ids[row["prompt_id"]],
            "generated_token_ids": [int(x) for x in new_token_ids.tolist()],
            "auto_output_class": output_class,
            "classification_confidence": confidence,
        }
        print(
            f"generated {idx}/7 {row['prompt_id']} tokens={prompt_token_count} "
            f"new_tokens={int(new_token_ids.shape[0])} class={output_class}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with OUTPUT_TSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "prompt_id",
            "prompt_family",
            "perturbation_type",
            "prompt_token_count",
            "generated_text",
            "notable_phrases",
            "manual_or_auto_output_class",
            "classification_confidence",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "timestamp_utc": utc_now(),
        "workspace_root": str(ROOT),
        "model_path": str(MODEL_PATH),
        "model": MODEL_LABEL,
        "script_path": str(ROOT / "scripts" / "run_qwen_hum_behavioral_matrix.py"),
        "prompt_matrix_path": str(PROMPT_MATRIX_PATH),
        "rubric_path": str(RUBRIC_PATH),
        "output_tsv": str(OUTPUT_TSV),
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation": {
            "do_sample": False,
            "decoding": "deterministic greedy",
            "local_files_only": True,
            "device_map": "single_gpu_cuda0",
            "device_map_reason": "device_map=auto produced NaN hidden states on this fresh instance; single GPU fits in 98GB VRAM and produced finite diagnostics",
        },
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "prompt_count": len(rows),
        "prompt_token_ids_saved": True,
        "classifications": "auto heuristic from behavioral_output_class_rubric.tsv",
        "prompt_metadata": generation_metadata,
        "restrictions": [
            "no steering",
            "no Hauhau",
            "no llama.cpp",
            "no SAE capture in this behavioral script",
        ],
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
