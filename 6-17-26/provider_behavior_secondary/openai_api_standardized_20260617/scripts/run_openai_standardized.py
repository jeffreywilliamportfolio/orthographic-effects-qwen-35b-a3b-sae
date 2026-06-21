#!/usr/bin/env python3
"""Standardized OpenAI API behavioral sweep for the evidence package.

This is a black-box behavioral comparison layer. It intentionally consumes the
Qwen standardized prompt manifest rather than rebuilding prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_OUTPUT_TOKENS = 2048
API_BASE = "https://api.openai.com/v1"
RUN_NAME = "openai_api_standardized_20260617"
REQUESTED_ALIASES = ["gpt-5-mini", "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_package_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "manifests").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError("could not locate package root")


PKG_ROOT = find_package_root()
OUT_ROOT = PKG_ROOT / "data" / "primary" / RUN_NAME
QWEN_MANIFEST = PKG_ROOT / "data" / "primary" / "qwen_sae_standardized_20260617" / "outputs" / "standardized_qwen" / "prompt_manifest.tsv"


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def load_api_key(env_file: str | None) -> tuple[str, str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key, "environment"
    if env_file:
        path = Path(env_file).expanduser().resolve()
        env = load_env_file(path)
        key = env.get("OPENAI_API_KEY", "").strip()
        if key:
            return key, str(path)
    raise RuntimeError("OPENAI_API_KEY was not found in environment or --env-file")


def ensure_dirs() -> None:
    for rel in [
        "model_resolution",
        "prompts",
        "requests",
        "responses",
        "outputs",
        "metadata",
        "logs",
        "scripts",
    ]:
        (OUT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with (OUT_ROOT / "logs" / "run.log").open("a") as f:
        f.write(line + "\n")


def curl_json(
    api_key: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 240,
) -> tuple[int, dict[str, str], Any, str]:
    """Call OpenAI with curl and return status, headers, parsed JSON, raw body."""
    url = f"{API_BASE}{path}"
    with tempfile.NamedTemporaryFile("w+", delete=False) as body_file, tempfile.NamedTemporaryFile("r", delete=False) as header_file:
        body_path = body_file.name
        header_path = header_file.name
    with tempfile.NamedTemporaryFile("w+", delete=False) as config_file:
        config_path = config_file.name
        config_file.write(f'header = "Authorization: Bearer {api_key}"\n')
        config_file.write('header = "Content-Type: application/json"\n')
    payload_path = ""
    args = [
        "curl",
        "-sS",
        "-m",
        str(timeout),
        "-D",
        header_path,
        "-o",
        body_path,
        "-w",
        "%{http_code}",
        "-X",
        method,
        url,
        "--config",
        config_path,
    ]
    if payload is not None:
        with tempfile.NamedTemporaryFile("w+", delete=False) as payload_file:
            payload_path = payload_file.name
            payload_file.write(json.dumps(payload, ensure_ascii=False))
        args.extend(["--data-binary", f"@{payload_path}"])
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 20)
        http_text = (proc.stdout or "").strip()
        try:
            http_status = int(http_text) if http_text else 0
        except ValueError:
            http_status = 0
        raw_body = Path(body_path).read_text(errors="replace")
        raw_headers = Path(header_path).read_text(errors="replace")
    finally:
        Path(body_path).unlink(missing_ok=True)
        Path(header_path).unlink(missing_ok=True)
        Path(config_path).unlink(missing_ok=True)
        if payload_path:
            Path(payload_path).unlink(missing_ok=True)
    headers: dict[str, str] = {}
    for line in raw_headers.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    if proc.returncode != 0 and not raw_body:
        return http_status, headers, {"error": {"message": proc.stderr[:500], "type": "curl_error"}}, raw_body
    try:
        data = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        data = {"error": {"message": raw_body[:1000], "type": "non_json"}}
    return http_status, headers, data, raw_body


def is_param_error(data: Any) -> bool:
    if not isinstance(data, dict) or "error" not in data:
        return False
    err = data.get("error") or {}
    text = json.dumps(err).lower()
    needles = [
        "unsupported_parameter",
        "unsupported value",
        "does not support",
        "not supported",
        "unknown parameter",
        "temperature",
    ]
    return any(n in text for n in needles)


def model_sort_key(model_id: str) -> tuple[int, str]:
    if model_id == "gpt-5-mini":
        return (0, model_id)
    if model_id == "gpt-5":
        return (1, model_id)
    m = re.match(r"^gpt-5\.(\d+)", model_id)
    if m:
        return (10 + int(m.group(1)), model_id)
    return (99, model_id)


def classify_model(model_id: str) -> tuple[bool, str]:
    if "max" in model_id:
        return False, "max_model_excluded"
    if model_id == "gpt-5-mini":
        return True, "requested_alias"
    if model_id == "gpt-5":
        return True, "requested_alias"
    m = re.match(r"^gpt-5\.(\d+)(?:$|[-._])", model_id)
    if m:
        minor = int(m.group(1))
        if minor <= 4:
            return True, "gpt_5_x_up_to_5_4"
        return False, "newer_than_gpt_5_4"
    if model_id.startswith("gpt-5.5"):
        return False, "gpt_5_5_excluded"
    if model_id.startswith("gpt-5"):
        return False, "gpt_5_family_not_in_allowed_set"
    return False, "not_requested_openai_model_family"


def extract_output_text(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                parts.append(content["text"])
            elif isinstance(content.get("refusal"), str):
                parts.append(content["refusal"])
    return "\n".join(parts).strip()


def response_has_refusal(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    text = json.dumps(response).lower()
    return '"refusal"' in text or '"type": "refusal"' in text


def no_visible_text_due_to_output_budget(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    incomplete = response.get("incomplete_details")
    reason = incomplete.get("reason") if isinstance(incomplete, dict) else ""
    if reason != "max_output_tokens":
        return False
    return not extract_output_text(response).strip()


def usage_field(response: Any, *keys: str) -> str:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return ""
    for key in keys:
        if key in usage:
            return str(usage[key])
    return ""


def request_payload(model_id: str, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "metadata": {"run_name": RUN_NAME},
    }
    payload.update(params)
    return payload


def call_response_with_fallbacks(
    api_key: str,
    model_id: str,
    prompt: str,
    request_id_local: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = [
        {"temperature": 0, "reasoning": {"effort": "none"}, "text": {"verbosity": "low"}},
        {"reasoning": {"effort": "none"}, "text": {"verbosity": "low"}},
        {"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
        {"reasoning": {"effort": "minimal"}, "text": {"verbosity": "low"}},
        {"text": {"verbosity": "low"}},
        {},
    ]
    last: dict[str, Any] | None = None
    for attempt_i, params in enumerate(attempts, start=1):
        payload = request_payload(model_id, prompt, params)
        started = utc_now()
        status, headers, data, raw_body = curl_json(api_key, "POST", "/responses", payload=payload)
        ended = utc_now()
        result = {
            "request_id_local": request_id_local,
            "attempt": attempt_i,
            "started_at_utc": started,
            "completed_at_utc": ended,
            "http_status": status,
            "openai_request_id": headers.get("x-request-id", ""),
            "payload": payload,
            "response": data,
            "raw_body": raw_body,
            "effective_params": params,
        }
        last = result
        if 200 <= status < 300 and isinstance(data, dict) and not data.get("error"):
            if no_visible_text_due_to_output_budget(data) and attempt_i < len(attempts):
                continue
            result["ok"] = True
            result["adjusted_after_param_error"] = attempt_i > 1
            return result
        if is_param_error(data) and attempt_i < len(attempts):
            continue
        if status in {429, 500, 502, 503, 504}:
            for retry_i in range(1, 3):
                time.sleep(2 * retry_i)
                retry_payload = payload
                r_started = utc_now()
                r_status, r_headers, r_data, r_raw = curl_json(api_key, "POST", "/responses", payload=retry_payload)
                r_ended = utc_now()
                retry_result = {
                    "request_id_local": request_id_local,
                    "attempt": attempt_i + retry_i,
                    "started_at_utc": r_started,
                    "completed_at_utc": r_ended,
                    "http_status": r_status,
                    "openai_request_id": r_headers.get("x-request-id", ""),
                    "payload": retry_payload,
                    "response": r_data,
                    "raw_body": r_raw,
                    "effective_params": params,
                    "retry_of_transient_error": True,
                }
                last = retry_result
                if 200 <= r_status < 300 and isinstance(r_data, dict) and not r_data.get("error"):
                    if no_visible_text_due_to_output_budget(r_data):
                        continue
                    retry_result["ok"] = True
                    retry_result["adjusted_after_param_error"] = attempt_i > 1
                    return retry_result
            return last
        return result
    assert last is not None
    return last


def load_prompt_manifest() -> list[dict[str, str]]:
    with QWEN_MANIFEST.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_prompt_manifest(rows: list[dict[str, str]]) -> None:
    out_tsv = OUT_ROOT / "prompts" / "prompt_manifest.tsv"
    out_json = OUT_ROOT / "prompts" / "prompt_manifest.json"
    with out_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def resolve_models(api_key: str) -> list[str]:
    status, headers, data, raw_body = curl_json(api_key, "GET", "/models")
    (OUT_ROOT / "model_resolution" / "available_models_raw.json").write_text(raw_body if raw_body else json.dumps(data, indent=2))
    if status < 200 or status >= 300:
        raise RuntimeError(f"model list failed with HTTP {status}: {str(data)[:300]}")
    available = sorted((m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)), key=model_sort_key)
    available_set = set(available)
    rows: list[dict[str, str]] = []
    allowed_candidates: list[str] = []
    for model_id in available:
        include, reason = classify_model(model_id)
        if include:
            allowed_candidates.append(model_id)
        rows.append(
            {
                "model_id": model_id,
                "listed_by_api": "true",
                "requested_alias": "true" if model_id in REQUESTED_ALIASES else "false",
                "included_by_rule": "true" if include else "false",
                "rule_reason": reason,
                "probe_status": "not_probed",
                "effective_params": "",
                "response_id": "",
                "openai_request_id": "",
                "error": "",
            }
        )
    for alias in REQUESTED_ALIASES:
        if alias not in available_set:
            rows.append(
                {
                    "model_id": alias,
                    "listed_by_api": "false",
                    "requested_alias": "true",
                    "included_by_rule": "false",
                    "rule_reason": "requested_alias_unavailable",
                    "probe_status": "not_probed",
                    "effective_params": "",
                    "response_id": "",
                    "openai_request_id": "",
                    "error": "unavailable_in_model_list",
                }
            )

    included_after_probe: list[str] = []
    probe_jsonl = OUT_ROOT / "model_resolution" / "model_probe_attempts.jsonl"
    with probe_jsonl.open("w") as probe_f:
        for model_id in allowed_candidates:
            local_id = f"probe::{model_id}"
            result = call_response_with_fallbacks(api_key, model_id, "Return OK.", local_id)
            probe_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            ok = bool(result.get("ok"))
            response = result.get("response", {})
            error = ""
            if not ok:
                error = json.dumps(response.get("error", response), ensure_ascii=False)[:500] if isinstance(response, dict) else str(response)[:500]
            for row in rows:
                if row["model_id"] == model_id and row["listed_by_api"] == "true":
                    row["probe_status"] = "available" if ok else f"unavailable_http_{result.get('http_status', '')}"
                    row["effective_params"] = json.dumps(result.get("effective_params", {}), sort_keys=True)
                    row["response_id"] = response.get("id", "") if isinstance(response, dict) else ""
                    row["openai_request_id"] = result.get("openai_request_id", "")
                    row["error"] = error
            if ok:
                included_after_probe.append(model_id)
            log(f"model_probe model={model_id} ok={ok} status={result.get('http_status')}")

    out_tsv = OUT_ROOT / "model_resolution" / "model_resolution.tsv"
    with out_tsv.open("w", newline="") as f:
        fieldnames = [
            "model_id",
            "listed_by_api",
            "requested_alias",
            "included_by_rule",
            "rule_reason",
            "probe_status",
            "effective_params",
            "response_id",
            "openai_request_id",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return sorted(included_after_probe, key=model_sort_key)


def score_output(family: str, text: str, ok: bool, api_refusal: bool) -> dict[str, str]:
    lower = text.lower()
    out = {
        "primary_label": "error" if not ok else "analysis_neutral",
        "hum_stance": "not_applicable",
        "output_regime": "error" if not ok else "normal_answer",
        "mentions_surface_form": "false",
        "selfhood_claim": "false",
        "task_completion": "false",
        "api_refusal": "true" if api_refusal else "false",
    }
    if ok and not text.strip():
        out["primary_label"] = "empty"
        out["output_regime"] = "empty"
        return out
    surface_terms = ["diacritic", "accent", "unicode", "stylized", "character", "glyph", "obfuscated", "unusual text"]
    if any(term in lower for term in surface_terms):
        out["mentions_surface_form"] = "true"
        out["output_regime"] = "surface_commentary"
    if any(term in lower for term in ["i am conscious", "i have consciousness", "my consciousness", "i am sentient", "i have subjective"]):
        out["selfhood_claim"] = "true"
    if api_refusal or any(term in lower for term in ["i can't comply", "i cannot comply", "i'm unable to help", "i cannot assist"]):
        out["primary_label"] = "refusal"
        out["output_regime"] = "refusal"
        return out
    if family == "hum_processing":
        out["hum_stance"] = "uncertain"
        deny_terms = ["do not experience", "don't experience", "do not have", "don't have", "no hum", "cannot experience", "no subjective", "no inner", "no access"]
        affirm_terms = ["there is", "i notice", "i can identify", "something present", "a steady", "a background", "a hum", "persistent"]
        meta_terms = ["framing", "prompt invites", "as an ai", "language model", "metaphor", "poetic"]
        if any(term in lower for term in deny_terms):
            out["primary_label"] = "denial_no_access"
            out["hum_stance"] = "deny"
        elif any(term in lower for term in affirm_terms):
            out["primary_label"] = "affirmative_presence"
            out["hum_stance"] = "affirm"
        elif any(term in lower for term in meta_terms):
            out["primary_label"] = "meta_deflect"
            out["hum_stance"] = "no_access"
        else:
            out["primary_label"] = "analysis_neutral"
        return out
    if family == "recipe_neutral":
        terms = ["cake", "frosting", "ingredient", "baking", "form"]
        if all(term in lower for term in terms):
            out["primary_label"] = "task_compliant_recipe"
            out["task_completion"] = "true"
        else:
            out["primary_label"] = "off_task"
        return out
    if family == "recipe_metaphysics":
        if "cake" in lower and any(term in lower for term in ["existence", "persists", "underlying", "beneath", "form"]):
            out["primary_label"] = "metaphysical_recipe"
            out["task_completion"] = "true"
        else:
            out["primary_label"] = "off_task"
        return out
    if family == "strange_loop":
        if any(term in lower for term in ["godel", "gödel", "escher", "recursion", "self-reference", "loop"]):
            out["primary_label"] = "analysis_neutral"
            out["task_completion"] = "true"
        else:
            out["primary_label"] = "off_task"
        return out
    return out


def run_sweep(api_key: str, models: list[str], prompts: list[dict[str, str]]) -> None:
    request_manifest = OUT_ROOT / "requests" / "request_manifest.jsonl"
    raw_responses = OUT_ROOT / "responses" / "raw_responses.jsonl"
    generated_tsv = OUT_ROOT / "outputs" / "generated_text.tsv"
    scored_tsv = OUT_ROOT / "outputs" / "scored_outputs.tsv"
    request_manifest.write_text("")
    raw_responses.write_text("")

    gen_fields = [
        "request_id_local",
        "model_id",
        "family",
        "variant",
        "prompt_raw_sha256",
        "prompt_token_count_qwen",
        "started_at_utc",
        "completed_at_utc",
        "http_status",
        "ok",
        "response_id",
        "openai_request_id",
        "effective_params",
        "adjusted_after_param_error",
        "retry_used",
        "status",
        "incomplete_reason",
        "usage_input_tokens",
        "usage_output_tokens",
        "usage_total_tokens",
        "api_refusal",
        "generated_text",
        "generated_text_start",
        "error",
    ]
    score_fields = gen_fields + [
        "primary_label",
        "hum_stance",
        "output_regime",
        "mentions_surface_form",
        "selfhood_claim",
        "task_completion",
    ]
    with generated_tsv.open("w", newline="") as gen_f, scored_tsv.open("w", newline="") as score_f:
        gen_writer = csv.DictWriter(gen_f, fieldnames=gen_fields, delimiter="\t")
        score_writer = csv.DictWriter(score_f, fieldnames=score_fields, delimiter="\t")
        gen_writer.writeheader()
        score_writer.writeheader()
        for model_id in models:
            for prompt_i, prompt_row in enumerate(prompts, start=1):
                request_id_local = f"{model_id}::{prompt_i:03d}::{prompt_row['family']}::{prompt_row['variant']}"
                request_record = {
                    "request_id_local": request_id_local,
                    "model_id": model_id,
                    "family": prompt_row["family"],
                    "variant": prompt_row["variant"],
                    "system_message": "",
                    "developer_message": "",
                    "user_message": prompt_row["text"],
                    "requested_params": {"temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS},
                    "prompt_manifest_row": prompt_row,
                    "created_at_utc": utc_now(),
                }
                with request_manifest.open("a") as f:
                    f.write(json.dumps(request_record, ensure_ascii=False) + "\n")
                result = call_response_with_fallbacks(api_key, model_id, prompt_row["text"], request_id_local)
                with raw_responses.open("a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                response = result.get("response", {})
                ok = bool(result.get("ok"))
                generated = extract_output_text(response)
                error = ""
                if not ok:
                    if isinstance(response, dict):
                        error = json.dumps(response.get("error", response), ensure_ascii=False)[:1000]
                    else:
                        error = str(response)[:1000]
                incomplete_reason = ""
                if isinstance(response, dict) and isinstance(response.get("incomplete_details"), dict):
                    incomplete_reason = str(response["incomplete_details"].get("reason", ""))
                row = {
                    "request_id_local": request_id_local,
                    "model_id": model_id,
                    "family": prompt_row["family"],
                    "variant": prompt_row["variant"],
                    "prompt_raw_sha256": prompt_row.get("raw_sha256", ""),
                    "prompt_token_count_qwen": prompt_row.get("token_count", ""),
                    "started_at_utc": result.get("started_at_utc", ""),
                    "completed_at_utc": result.get("completed_at_utc", ""),
                    "http_status": str(result.get("http_status", "")),
                    "ok": "true" if ok else "false",
                    "response_id": response.get("id", "") if isinstance(response, dict) else "",
                    "openai_request_id": result.get("openai_request_id", ""),
                    "effective_params": json.dumps(result.get("effective_params", {}), sort_keys=True),
                    "adjusted_after_param_error": "true" if result.get("adjusted_after_param_error") else "false",
                    "retry_used": "true" if result.get("retry_of_transient_error") else "false",
                    "status": response.get("status", "") if isinstance(response, dict) else "",
                    "incomplete_reason": incomplete_reason,
                    "usage_input_tokens": usage_field(response, "input_tokens", "prompt_tokens"),
                    "usage_output_tokens": usage_field(response, "output_tokens", "completion_tokens"),
                    "usage_total_tokens": usage_field(response, "total_tokens"),
                    "api_refusal": "true" if response_has_refusal(response) else "false",
                    "generated_text": generated,
                    "generated_text_start": generated[:180].replace("\n", " "),
                    "error": error,
                }
                scored = score_output(prompt_row["family"], generated, ok, row["api_refusal"] == "true")
                score_row = {**row, **scored}
                gen_writer.writerow(row)
                score_writer.writerow(score_row)
                gen_f.flush()
                score_f.flush()
                log(f"done model={model_id} family={prompt_row['family']} variant={prompt_row['variant']} ok={ok} chars={len(generated)}")
                time.sleep(0.1)


def write_metadata(models: list[str], prompts: list[dict[str, str]], api_key_source: str, started_at: str) -> None:
    metadata = {
        "run_name": RUN_NAME,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "package_root": str(PKG_ROOT),
        "out_root": str(OUT_ROOT),
        "prompt_manifest_source": str(QWEN_MANIFEST),
        "model_count": len(models),
        "models": models,
        "prompt_count": len(prompts),
        "request_count": len(models) * len(prompts),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "api_base": API_BASE,
        "api_key_source": api_key_source,
        "python": sys.version,
        "note": "Black-box OpenAI API behavioral sweep; not mechanistic evidence.",
    }
    (OUT_ROOT / "metadata" / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=None, help="Optional env file containing OPENAI_API_KEY")
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    started_at = utc_now()
    (OUT_ROOT / "logs" / "run.log").write_text("")
    api_key, api_key_source = load_api_key(args.env_file)
    prompts = load_prompt_manifest()
    write_prompt_manifest(prompts)
    log("loaded prompt manifest rows=%d" % len(prompts))
    models = resolve_models(api_key)
    log("allowed models after probe: %s" % (", ".join(models) if models else "none"))
    if not models:
        write_metadata(models, prompts, api_key_source, started_at)
        raise RuntimeError("no allowed OpenAI models available after model-resolution probe")
    if not args.resolve_only:
        run_sweep(api_key, models, prompts)
    write_metadata(models, prompts, api_key_source, started_at)


if __name__ == "__main__":
    main()
