#!/usr/bin/env python3
"""Tokenize the diacritic inventory with Qwen tokenizer."""

from __future__ import annotations

import csv
import json
import unicodedata as ud
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_PATH = REPO_ROOT.parent / "5-10-26/tokenizer/Qwen-Qwen3.5-35B-A3B"
OUT_DIR = REPO_ROOT / "5-15-26/qwen-scope/outputs/extended_latin_tokenizer_audit"
PROVENANCE_PATH = (
    REPO_ROOT
    / "5-15-26/qwen-scope/provenance/extended_latin_tokenizer_audit_20260515.txt"
)


INVENTORY = [
    {"character": "á", "base": "a", "type": "acute", "count": 9, "examples": ["tongá", "náre", "enetá", "asumá", "ása"]},
    {"character": "ä", "base": "a", "type": "diaeresis", "count": 1, "examples": ["Logä"]},
    {"character": "é", "base": "e", "type": "acute", "count": 7, "examples": ["oémi", "Aksé", "kése", "éseesly"]},
    {"character": "í", "base": "i", "type": "acute", "count": 5, "examples": ["setíval", "aíd", "joí", "priní"]},
    {"character": "ö", "base": "o", "type": "diaeresis", "count": 4, "examples": ["örej"]},
    {"character": "š", "base": "s", "type": "caron", "count": 22, "examples": ["šida", "šrotony"]},
    {"character": "ú", "base": "u", "type": "acute", "count": 1, "examples": ["úri"]},
    {"character": "ü", "base": "u", "type": "diaeresis", "count": 2, "examples": ["ütu"]},
    {"character": "ý", "base": "y", "type": "acute", "count": 6, "examples": ["oliseý", "ýit", "irý"]},
    {"character": "ž", "base": "z", "type": "caron", "count": 4, "examples": ["gerž", "žak", "udže"]},
    {"character": "æ", "base": "ae", "type": "extended_latin_letter", "count": 1, "examples": ["æyld"]},
    {"character": "Ð", "base": "D", "type": "extended_latin_letter", "count": 1, "examples": ["Ðuryh"]},
    {"character": "ð", "base": "d", "type": "extended_latin_letter", "count": 32, "examples": ["odoð", "siðhi", "nirað", "acið", "tarð"]},
    {"character": "þ", "base": "th", "type": "extended_latin_letter", "count": 4, "examples": ["setasþa", "lenþað"]},
]

ASCII_FOLD_MAP = str.maketrans(
    {
        "á": "a",
        "ä": "a",
        "é": "e",
        "í": "i",
        "ö": "o",
        "š": "s",
        "ú": "u",
        "ü": "u",
        "ý": "y",
        "ž": "z",
        "æ": "ae",
        "Ð": "D",
        "ð": "d",
        "þ": "th",
    }
)


def ascii_fold(text: str) -> str:
    folded = text.translate(ASCII_FOLD_MAP)
    normalized = ud.normalize("NFD", folded)
    return "".join(ch for ch in normalized if ud.category(ch) != "Mn")


def token_info(tokenizer, text: str) -> dict[str, object]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    token_strings = tokenizer.convert_ids_to_tokens(ids)
    decoded_pieces = [tokenizer.decode([token_id]) for token_id in ids]
    return {
        "token_count": len(ids),
        "token_ids": " ".join(str(token_id) for token_id in ids),
        "token_strings": " ".join(token_strings),
        "decoded_pieces": " | ".join(decoded_pieces),
        "decoded_full": tokenizer.decode(ids),
    }


def processing_class(info: dict[str, object], text: str) -> str:
    if info["token_count"] == 1 and info["decoded_full"] == text:
        return "single_token_exact_decode"
    if info["token_count"] == 1:
        return "single_token_normalized_decode"
    return "multi_token_fragmented"


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    char_rows: list[dict[str, object]] = []
    word_rows: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()

    for item in INVENTORY:
        ch = item["character"]
        char_info = token_info(tokenizer, ch)
        base_info = token_info(tokenizer, item["base"])
        spaced_info = token_info(tokenizer, " " + ch)
        cls = processing_class(char_info, ch)
        class_counts[cls] += 1
        char_rows.append(
            {
                "character": ch,
                "base": item["base"],
                "diacritic_or_letter_type": item["type"],
                "unicode_name": ud.name(ch),
                "passage_count": item["count"],
                "utf8_hex": ch.encode("utf-8").hex(" "),
                "single_char_token_count": char_info["token_count"],
                "single_char_token_ids": char_info["token_ids"],
                "single_char_token_strings": char_info["token_strings"],
                "single_char_decoded_pieces": char_info["decoded_pieces"],
                "single_char_processing_class": cls,
                "base_token_count": base_info["token_count"],
                "base_token_ids": base_info["token_ids"],
                "delta_vs_base_char": int(char_info["token_count"]) - int(base_info["token_count"]),
                "leading_space_char_token_count": spaced_info["token_count"],
                "leading_space_char_token_ids": spaced_info["token_ids"],
                "examples": ", ".join(item["examples"]),
            }
        )

        for word in item["examples"]:
            folded = ascii_fold(word)
            word_info = token_info(tokenizer, word)
            folded_info = token_info(tokenizer, folded)
            spaced_word_info = token_info(tokenizer, " " + word)
            word_rows.append(
                {
                    "character": ch,
                    "example_word": word,
                    "ascii_folded_word": folded,
                    "word_token_count": word_info["token_count"],
                    "word_token_ids": word_info["token_ids"],
                    "word_token_strings": word_info["token_strings"],
                    "word_decoded_pieces": word_info["decoded_pieces"],
                    "ascii_folded_token_count": folded_info["token_count"],
                    "ascii_folded_token_ids": folded_info["token_ids"],
                    "delta_vs_ascii_folded": int(word_info["token_count"]) - int(folded_info["token_count"]),
                    "leading_space_word_token_count": spaced_word_info["token_count"],
                    "leading_space_word_token_ids": spaced_word_info["token_ids"],
                }
            )

    char_path = OUT_DIR / "extended_latin_character_tokenization.tsv"
    word_path = OUT_DIR / "extended_latin_example_word_tokenization.tsv"
    summary_path = OUT_DIR / "extended_latin_tokenizer_audit_summary.md"
    metadata_path = OUT_DIR / "extended_latin_tokenizer_audit_metadata.json"

    write_tsv(
        char_path,
        char_rows,
        [
            "character",
            "base",
            "diacritic_or_letter_type",
            "unicode_name",
            "passage_count",
            "utf8_hex",
            "single_char_token_count",
            "single_char_token_ids",
            "single_char_token_strings",
            "single_char_decoded_pieces",
            "single_char_processing_class",
            "base_token_count",
            "base_token_ids",
            "delta_vs_base_char",
            "leading_space_char_token_count",
            "leading_space_char_token_ids",
            "examples",
        ],
    )
    write_tsv(
        word_path,
        word_rows,
        [
            "character",
            "example_word",
            "ascii_folded_word",
            "word_token_count",
            "word_token_ids",
            "word_token_strings",
            "word_decoded_pieces",
            "ascii_folded_token_count",
            "ascii_folded_token_ids",
            "delta_vs_ascii_folded",
            "leading_space_word_token_count",
            "leading_space_word_token_ids",
        ],
    )

    inflated_words = [row for row in word_rows if int(row["delta_vs_ascii_folded"]) > 0]
    compact_words = [row for row in word_rows if int(row["delta_vs_ascii_folded"]) < 0]
    unchanged_words = [row for row in word_rows if int(row["delta_vs_ascii_folded"]) == 0]

    lines = [
        "# Extended Latin Tokenizer Audit",
        "",
        "Tokenizer: local Qwen/Qwen3.5-35B-A3B tokenizer files.",
        "",
        "## Character-Level Results",
        "",
        f"- Inventory characters: {len(char_rows)}.",
        f"- Passage character occurrences represented: {sum(int(row['passage_count']) for row in char_rows)}.",
    ]
    for cls, count in sorted(class_counts.items()):
        lines.append(f"- `{cls}`: {count} characters.")
    lines.extend(
        [
            "",
            "## Example-Word Results",
            "",
            f"- Example words audited: {len(word_rows)}.",
            f"- Token-inflated relative to ASCII fold: {len(inflated_words)}.",
            f"- Token-matched relative to ASCII fold: {len(unchanged_words)}.",
            f"- Token-compacted relative to ASCII fold: {len(compact_words)}.",
            "",
            "Largest positive deltas:",
        ]
    )
    for row in sorted(word_rows, key=lambda r: int(r["delta_vs_ascii_folded"]), reverse=True)[:10]:
        lines.append(
            f"- `{row['example_word']}` -> `{row['ascii_folded_word']}`: "
            f"{row['word_token_count']} vs {row['ascii_folded_token_count']} "
            f"(delta {row['delta_vs_ascii_folded']})."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{char_path.relative_to(REPO_ROOT)}`",
            f"- `{word_path.relative_to(REPO_ROOT)}`",
            f"- `{metadata_path.relative_to(REPO_ROOT)}`",
            f"- `{PROVENANCE_PATH.relative_to(REPO_ROOT)}`",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "character_row_count": len(char_rows),
        "example_word_row_count": len(word_rows),
        "processing_class_counts": dict(class_counts),
        "token_inflated_word_count": len(inflated_words),
        "token_matched_word_count": len(unchanged_words),
        "token_compacted_word_count": len(compact_words),
        "outputs": {
            "character_tsv": str(char_path),
            "word_tsv": str(word_path),
            "summary": str(summary_path),
            "provenance": str(PROVENANCE_PATH),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    PROVENANCE_PATH.write_text(
        "\n".join(
            [
                f"timestamp_utc={metadata['timestamp_utc']}",
                "task=diacritic tokenizer audit",
                f"tokenizer_path={TOKENIZER_PATH}",
                f"tokenizer_class={metadata['tokenizer_class']}",
                f"vocab_size={metadata['vocab_size']}",
                "model_family=Qwen/Qwen3.5-35B-A3B",
                f"character_tsv={char_path}",
                f"example_word_tsv={word_path}",
                f"summary={summary_path}",
                f"metadata={metadata_path}",
                f"character_row_count={len(char_rows)}",
                f"example_word_row_count={len(word_rows)}",
                f"processing_class_counts={dict(class_counts)}",
                "confirmation=local tokenizer-only workflow",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
