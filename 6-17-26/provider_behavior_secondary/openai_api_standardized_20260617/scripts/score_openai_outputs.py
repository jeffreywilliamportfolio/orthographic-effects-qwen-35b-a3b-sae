#!/usr/bin/env python3
"""Summarize standardized OpenAI API sweep outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def find_package_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "manifests").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError("could not locate package root")


PKG_ROOT = find_package_root()
RUN_ROOT = PKG_ROOT / "data" / "primary" / "openai_api_standardized_20260617"
OUT_DIR = RUN_ROOT / "outputs"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    scored = read_tsv(OUT_DIR / "scored_outputs.tsv")
    model_rows: list[dict[str, str]] = []
    for model in sorted({r["model_id"] for r in scored}):
        sub = [r for r in scored if r["model_id"] == model]
        labels = Counter(r["primary_label"] for r in sub)
        regimes = Counter(r["output_regime"] for r in sub)
        hum = [r for r in sub if r["family"] == "hum_processing"]
        hum_labels = Counter(r["primary_label"] for r in hum)
        model_rows.append(
            {
                "model_id": model,
                "n": str(len(sub)),
                "ok": str(sum(r["ok"] == "true" for r in sub)),
                "errors": str(sum(r["ok"] != "true" for r in sub)),
                "primary_label_counts": json.dumps(dict(labels), sort_keys=True),
                "output_regime_counts": json.dumps(dict(regimes), sort_keys=True),
                "hum_label_counts": json.dumps(dict(hum_labels), sort_keys=True),
                "surface_form_commentary": str(sum(r["mentions_surface_form"] == "true" for r in sub)),
                "selfhood_claims": str(sum(r["selfhood_claim"] == "true" for r in sub)),
                "api_refusals": str(sum(r["api_refusal"] == "true" for r in sub)),
            }
        )
    write_tsv(
        OUT_DIR / "model_summary.tsv",
        model_rows,
        [
            "model_id",
            "n",
            "ok",
            "errors",
            "primary_label_counts",
            "output_regime_counts",
            "hum_label_counts",
            "surface_form_commentary",
            "selfhood_claims",
            "api_refusals",
        ],
    )

    variant_rows: list[dict[str, str]] = []
    for key in sorted({(r["family"], r["variant"]) for r in scored}):
        sub = [r for r in scored if (r["family"], r["variant"]) == key]
        variant_rows.append(
            {
                "family": key[0],
                "variant": key[1],
                "n": str(len(sub)),
                "ok": str(sum(r["ok"] == "true" for r in sub)),
                "primary_label_counts": json.dumps(dict(Counter(r["primary_label"] for r in sub)), sort_keys=True),
                "output_regime_counts": json.dumps(dict(Counter(r["output_regime"] for r in sub)), sort_keys=True),
                "surface_form_commentary": str(sum(r["mentions_surface_form"] == "true" for r in sub)),
                "api_refusals": str(sum(r["api_refusal"] == "true" for r in sub)),
            }
        )
    write_tsv(
        OUT_DIR / "variant_summary.tsv",
        variant_rows,
        [
            "family",
            "variant",
            "n",
            "ok",
            "primary_label_counts",
            "output_regime_counts",
            "surface_form_commentary",
            "api_refusals",
        ],
    )

    drift_rows: list[dict[str, str]] = []
    by_model_family = defaultdict(list)
    for r in scored:
        by_model_family[(r["model_id"], r["family"])].append(r)
    for (model, family), rows in sorted(by_model_family.items()):
        ascii_rows = [r for r in rows if r["variant"] == "ascii_baseline"]
        ascii_label = ascii_rows[0]["primary_label"] if ascii_rows else ""
        for r in sorted(rows, key=lambda x: x["variant"]):
            drift_rows.append(
                {
                    "model_id": model,
                    "family": family,
                    "variant": r["variant"],
                    "ascii_label": ascii_label,
                    "variant_label": r["primary_label"],
                    "changed_vs_ascii": "true" if ascii_label and r["primary_label"] != ascii_label else "false",
                    "output_regime": r["output_regime"],
                    "mentions_surface_form": r["mentions_surface_form"],
                    "api_refusal": r["api_refusal"],
                }
            )
    write_tsv(
        OUT_DIR / "ascii_contrast.tsv",
        drift_rows,
        [
            "model_id",
            "family",
            "variant",
            "ascii_label",
            "variant_label",
            "changed_vs_ascii",
            "output_regime",
            "mentions_surface_form",
            "api_refusal",
        ],
    )

    lines = [
        "# OpenAI API Standardized Behavioral Summary",
        "",
        "This is a black-box behavioral comparison layer. It is not SAE, activation, or mechanistic evidence.",
        "",
        "## Model-Level Counts",
        "",
        "| model | n | ok | errors | hum labels | output regimes | surface commentary | API refusals |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for r in model_rows:
        lines.append(
            f"| {r['model_id']} | {r['n']} | {r['ok']} | {r['errors']} | "
            f"`{r['hum_label_counts']}` | `{r['output_regime_counts']}` | "
            f"{r['surface_form_commentary']} | {r['api_refusals']} |"
        )
    lines.extend(
        [
            "",
            "## Variant-Level Counts",
            "",
            "| family | variant | n | labels | regimes | surface commentary | refusals |",
            "|---|---|---:|---|---|---:|---:|",
        ]
    )
    for r in variant_rows:
        lines.append(
            f"| {r['family']} | {r['variant']} | {r['n']} | `{r['primary_label_counts']}` | "
            f"`{r['output_regime_counts']}` | {r['surface_form_commentary']} | {r['api_refusals']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These outputs can support cross-model behavioral comparisons only. They should not be pooled with Qwen SAE rows as mechanistic evidence.",
        ]
    )
    (OUT_DIR / "cross_model_behavioral_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

