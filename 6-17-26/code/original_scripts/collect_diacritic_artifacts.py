#!/usr/bin/env python3
"""Collect diacritic/orthographic-perturbation research artifacts.

The collector is intentionally conservative: it preserves source paths under
`collected_sources/`, excludes raw tensors, model files, caches, virtualenvs,
and compressed run archives, and emits checksums for every copied file.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


REPO = Path(__file__).resolve().parents[1]
DEST_ROOT = REPO / "collected_sources"
INVENTORY = REPO / "inventory"


EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".hf",
    ".huggingface",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-sae",
    ".venv-sae-inspect",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "offload",
    "venv",
}

EXCLUDED_SUFFIXES = {
    ".7z",
    ".arrow",
    ".aux",
    ".bin",
    ".ckpt",
    ".db",
    ".dylib",
    ".fdb_latexmk",
    ".fls",
    ".gguf",
    ".h5",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".out",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".so",
    ".sqlite",
    ".tar",
    ".tar.gz",
    ".tmp",
    ".xdv",
    ".zip",
}

EXCLUDED_BASENAMES = {
    ".DS_Store",
    ".check_for_update_done",
}


def path_has_excluded_dir(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def excluded_suffix(path: Path) -> str | None:
    name = path.name.lower()
    for suffix in sorted(EXCLUDED_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return None


def should_copy_file(path: Path, rel: Path) -> tuple[bool, str]:
    if path.name in EXCLUDED_BASENAMES:
        return False, "excluded_basename"
    if path_has_excluded_dir(rel):
        return False, "excluded_directory"
    suffix = excluded_suffix(path)
    if suffix:
        return False, f"excluded_suffix:{suffix}"
    return True, "included"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_prefix(path: Path) -> Path:
    parts = path.parts
    if parts[:3] == ("/", "Volumes", "ExternalSSD"):
        return Path("external_ssd").joinpath(*parts[3:])
    if parts[:3] == ("/", "Users", "jeffreyshorthill"):
        return Path("internal_drive").joinpath(*parts[3:])
    return Path("other").joinpath(*[p for p in parts if p != "/"])


@dataclass(frozen=True)
class CopySpec:
    label: str
    source: Path
    selector: Callable[[Path, Path], bool] | None = None


def select_all(path: Path, rel: Path) -> bool:
    return True


def select_name_contains(*needles: str) -> Callable[[Path, Path], bool]:
    lowered = tuple(n.lower() for n in needles)

    def inner(path: Path, rel: Path) -> bool:
        s = str(rel).lower()
        return any(n in s for n in lowered)

    return inner


def select_gemma(path: Path, rel: Path) -> bool:
    s = str(rel)
    return (
        s in {
            "probes/hum-clean.txt",
            "probes/hum-diacritics.txt",
            "captures/hum-clean-cap.json",
            "captures/hum-diacritics-cap.json",
            "captures/hum-clean-deep.json",
            "captures/hum-diacritics-deep.json",
            "examples/prior-runs/captures/hum-clean-s42.json",
            "examples/prior-runs/captures/hum-diacritics-s42.json",
            "scripts/e114_hum_attractor_gemma.py",
            "scripts/self_report_atlas.py",
            "docs/archive/results-journal-e114-hum-attractor-gemma.md",
            "docs/archive/results-journal-hum-self-report-gemma-scope.md",
        }
        or s.startswith("examples/prior-runs/results/e114_hum_attractor_gemma/")
    )


def select_attractor(path: Path, rel: Path) -> bool:
    s = str(rel)
    return (
        s in {
            "JOURNAL-ORTHOGRAPHIC-PERTURBATION.md",
            "JOURNAL-E114-CHARACTERIZATION.md",
            "JOURNAL-RESIDUAL-ANALYSIS.md",
            "PROMPTS.md",
            "SAE_FEATURE_MAP.README.md",
            "SAE_FEATURE_MAP.csv",
            "run-staging/all_diac_user.txt",
            "run-staging/diac_user.txt",
            "run-staging/fl_diac_user.txt",
            "run-staging/setup_diac.sh",
            "run-staging/scripts/gen_diac.py",
            "run-staging/saelens/diac_breakdown.py",
        }
        or s.startswith("run-staging/results/diac_sae/")
    )


def select_aave_data(path: Path, rel: Path) -> bool:
    s = str(rel).lower()
    return (
        "unicode_dstroke" in s
        or "dstroke" in s
        or path.name in {
            "INTROSPECTIVE_UNICODE_DSTROKE_TOKENIZER_SUMMARY.md",
            "primary_hum_prompt_dstroke.md",
            "processing_hum_check_dstroke.md",
        }
    )


def select_aave_5_10(path: Path, rel: Path) -> bool:
    s = str(rel)
    return s in {
        "fl_structured_opacity_conditions.tsv",
        "scripts/prepare_fl_controls.py",
        "prefixes/C4_diacritic_stripped_fl_prefix.txt",
        "prompts/dstroke_p2.txt",
    }


def select_paper(path: Path, rel: Path) -> bool:
    # Keep PDFs and source/support files; skip LaTeX intermediates through suffix filters.
    s = str(rel)
    if s.startswith("arxiv_submission/stage/anc/"):
        return False
    return True


def select_internal(path: Path, rel: Path) -> bool:
    return True


SPECS: list[CopySpec] = [
    CopySpec(
        "paper_short_paper_seed",
        Path("/Volumes/ExternalSSD/aave-registers/papers/diacritics-short-paper"),
        select_paper,
    ),
    CopySpec(
        "paper_short_paper_compiled_pdf",
        Path("/Volumes/ExternalSSD/aave-registers/papers/diacritics-short-paper/build/diacritics_short_paper.pdf"),
        select_all,
    ),
    CopySpec(
        "paper_short_paper_arxiv_stage_pdf",
        Path("/Volumes/ExternalSSD/aave-registers/papers/diacritics-short-paper/arxiv_submission/stage/build/diacritics_short_paper.pdf"),
        select_all,
    ),
    CopySpec(
        "qwen_orthographic_effects_publication_package",
        Path("/Volumes/ExternalSSD/aave-registers/orthographic-effects-qwen-35b-a3b-sae"),
        select_all,
    ),
    CopySpec(
        "aave_registers_dstroke_data",
        Path("/Volumes/ExternalSSD/aave-registers/data"),
        select_aave_data,
    ),
    CopySpec(
        "aave_registers_introspection_unicode_dstroke_remote",
        Path("/Volumes/ExternalSSD/aave-registers/analysis/introspection_unicode_dstroke_remote"),
        select_all,
    ),
    CopySpec(
        "aave_registers_fl_controls_2026_05_10",
        Path("/Volumes/ExternalSSD/aave-registers/5-10-26"),
        select_aave_5_10,
    ),
    CopySpec(
        "aave_registers_unicode_byte_fallback_control",
        Path("/Volumes/ExternalSSD/aave-registers/runs/unicode_byte_fallback_hum_control_20260509"),
        select_all,
    ),
    CopySpec(
        "aave_registers_cleaned_initial_register",
        Path("/Volumes/ExternalSSD/aave-registers-cleaned/experiments/2026-05-05_initial_50_pair_register_no_think"),
        select_all,
    ),
    CopySpec(
        "aave_registers_cleaned_extended_latin_tokenizer",
        Path("/Volumes/ExternalSSD/aave-registers-cleaned/experiments/2026-05-15_extended_latin_tokenizer_audit"),
        select_all,
    ),
    CopySpec(
        "aave_registers_cleaned_missing_diacritics_tokenizer",
        Path("/Volumes/ExternalSSD/aave-registers-cleaned/experiments/2026-05-22_passage_missing_diacritics_tokenizer"),
        select_all,
    ),
    CopySpec(
        "aave_registers_cleaned_requested_accent_tokenizer",
        Path("/Volumes/ExternalSSD/aave-registers-cleaned/experiments/2026-05-22_requested_accent_set_tokenizer"),
        select_all,
    ),
    CopySpec(
        "aave_registers_cleaned_requested_diacritic_tokenizer",
        Path("/Volumes/ExternalSSD/aave-registers-cleaned/experiments/2026-05-22_requested_diacritic_set_tokenizer"),
        select_all,
    ),
    CopySpec(
        "aave_registers_cleaned_requested_mixed_unicode_tokenizer",
        Path("/Volumes/ExternalSSD/aave-registers-cleaned/experiments/2026-05-22_requested_mixed_unicode_tokenizer"),
        select_all,
    ),
    CopySpec(
        "aave_register_bias_source_corpus",
        Path("/Volumes/ExternalSSD/aave-register-bias-study/data/source-corpus"),
        select_aave_data,
    ),
    CopySpec(
        "unicode_byte_fallback_hum_control",
        Path("/Volumes/ExternalSSD/sae-tests/runs/unicode_byte_fallback_hum_control_20260509"),
        select_all,
    ),
    CopySpec(
        "gemma_4b_local_diacritics",
        Path("/Volumes/ExternalSSD/gemma-4b-local"),
        select_gemma,
    ),
    CopySpec(
        "gemma_4b_sae_test_diacritics",
        Path("/Volumes/ExternalSSD/gemma-4b-sae-test/gemma-3-4b-it-sae-demo-0.1.1"),
        select_gemma,
    ),
    CopySpec(
        "qwen_attractor_shift_diacritics",
        Path("/Volumes/ExternalSSD/attractor-shift-qwen-35b"),
        select_attractor,
    ),
    CopySpec(
        "journal_qwen_orthographic",
        Path("/Volumes/ExternalSSD/journals/qwen/35b/JOURNAL-ORTHOGRAPHIC-PERTURBATION.md"),
        select_all,
    ),
    CopySpec(
        "journal_qwen_tokenizer",
        Path("/Volumes/ExternalSSD/journals/qwen/tokenizer/JOURNAL-UNICODE-TOKENIZER-AUDITS.md"),
        select_all,
    ),
    CopySpec(
        "journal_attractor_shift_orthographic",
        Path("/Volumes/ExternalSSD/journals/journals-to-be-made/attractor-shift-qwen-35b/JOURNAL-ORTHOGRAPHIC-PERTURBATION.md"),
        select_all,
    ),
    CopySpec(
        "usenet_gemma_diacritic_io_sweeps",
        Path("/Volumes/ExternalSSD/usenet-training-corpus/io-runs"),
        select_all,
    ),
    CopySpec(
        "usenet_gemma_diacritic_reports",
        Path("/Volumes/ExternalSSD/usenet-training-corpus"),
        select_name_contains("io-temp", "io.md", "all-text-temp-tests.md"),
    ),
    CopySpec(
        "internal_paper_draft_pdf",
        Path("/Users/jeffreyshorthill/Downloads/diacritics-paper-draft.pdf"),
        select_internal,
    ),
    CopySpec(
        "internal_e114_diacritic_note",
        Path("/Users/jeffreyshorthill/Downloads/JOURNAL-E114-CHARACTERIZATION.md"),
        select_internal,
    ),
    CopySpec(
        "internal_desktop_gemma_io",
        Path("/Users/jeffreyshorthill/Desktop/io.md"),
        select_internal,
    ),
]


def iter_files(root: Path) -> Iterable[tuple[Path, Path]]:
    if root.is_file():
        yield root, Path(root.name)
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            yield path, path.relative_to(root)


def copy_specs() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    copied: list[dict[str, str]] = []
    omitted: list[dict[str, str]] = []

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    INVENTORY.mkdir(parents=True, exist_ok=True)

    for spec in SPECS:
        if not spec.source.exists():
            omitted.append(
                {
                    "label": spec.label,
                    "source": str(spec.source),
                    "reason": "missing_source",
                    "dest": "",
                    "size": "",
                    "sha256": "",
                }
            )
            continue

        for path, rel in iter_files(spec.source):
            if spec.selector and not spec.selector(path, rel):
                continue

            ok, reason = should_copy_file(path, rel)
            dest = DEST_ROOT / source_prefix(path)
            if spec.source.is_dir():
                dest = DEST_ROOT / source_prefix(spec.source) / rel
            if not ok:
                omitted.append(
                    {
                        "label": spec.label,
                        "source": str(path),
                        "reason": reason,
                        "dest": str(dest.relative_to(REPO)),
                        "size": str(path.stat().st_size if path.exists() else ""),
                        "sha256": "",
                    }
                )
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            digest = sha256(dest)
            copied.append(
                {
                    "label": spec.label,
                    "source": str(path),
                    "dest": str(dest.relative_to(REPO)),
                    "size": str(dest.stat().st_size),
                    "mtime_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "sha256": digest,
                }
            )

    return copied, omitted


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(copied: list[dict[str, str]], omitted: list[dict[str, str]]) -> None:
    write_tsv(
        INVENTORY / "MANIFEST.tsv",
        copied,
        ["label", "source", "dest", "size", "mtime_utc", "sha256"],
    )
    write_tsv(
        INVENTORY / "OMITTED.tsv",
        omitted,
        ["label", "source", "reason", "dest", "size", "sha256"],
    )

    sha_lines = [f"{row['sha256']}  {row['dest']}\n" for row in copied]
    (INVENTORY / "SHA256SUMS").write_text("".join(sha_lines), encoding="utf-8")

    by_label: dict[str, int] = {}
    bytes_by_label: dict[str, int] = {}
    for row in copied:
        by_label[row["label"]] = by_label.get(row["label"], 0) + 1
        bytes_by_label[row["label"]] = bytes_by_label.get(row["label"], 0) + int(row["size"])

    omitted_by_reason: dict[str, int] = {}
    for row in omitted:
        omitted_by_reason[row["reason"]] = omitted_by_reason.get(row["reason"], 0) + 1

    lines = [
        "# Collection Report",
        "",
        f"Generated: {datetime.now(tz=timezone.utc).isoformat()}",
        "",
        "## Included Clusters",
        "",
    ]
    for label in sorted(by_label):
        mb = bytes_by_label[label] / (1024 * 1024)
        lines.append(f"- `{label}`: {by_label[label]} files, {mb:.2f} MiB")
    lines.extend(["", "## Omitted By Reason", ""])
    for reason in sorted(omitted_by_reason):
        lines.append(f"- `{reason}`: {omitted_by_reason[reason]} files")
    lines.extend(
        [
            "",
            "## Exclusion Policy",
            "",
            "Raw tensors/arrays, model weights, local databases, compressed run archives,",
            "virtualenvs, caches, dependency folders, and generated LaTeX intermediates are",
            "omitted. See `OMITTED.tsv` for source paths and exact reasons.",
        ]
    )
    (INVENTORY / "COLLECTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    copied, omitted = copy_specs()
    write_reports(copied, omitted)
    total_bytes = sum(int(row["size"]) for row in copied)
    print(f"copied_files={len(copied)}")
    print(f"omitted_files={len(omitted)}")
    print(f"copied_mib={total_bytes / (1024 * 1024):.2f}")
    print(f"manifest={INVENTORY / 'MANIFEST.tsv'}")


if __name__ == "__main__":
    main()
