#!/usr/bin/env python3
"""
OpenAI platform of the TINE (tokenizer-induced non-equivalence) behavioral study.

Uses the SHARED battery (scripts/battery.py): 46 items at temp-0, plus a
temperature sweep over CORE_IDS (13 items) at temps [0.1,0.3,0.5,0.7,1.0], 2
samples each. Outputs paper/tables/openai_behavioral.{json,md}.

NETWORK NOTE: in this environment Python's own socket layer fails with
OSError Errno 9 (Bad file descriptor) for both urllib and requests, even as a
background / sandbox-disabled job — but `curl` works. DNS resolves fine; it is
specifically Python sockets that are broken. So all HTTP is done by shelling
out to `curl`. The `openai` SDK / requests are intentionally NOT used.

Param quirks: reasoning / o-series (o3, o4-mini) and possibly gpt-5* reject
`temperature` and require `max_completion_tokens` instead of `max_tokens`.
Strategy: try a standard call; on a 400 about an unsupported param, retry
without `temperature` and with `max_completion_tokens`. Any model that cannot
take `temperature` runs ONLY the temp-0 default pass and SKIPS its temp sweep
(recorded as "sweep_unsupported": true).

Checkpoint after every model. Skip-and-log per-call errors.
"""
import os, sys, json, time
sys.path.insert(0, "/Volumes/ExternalSSD/diacritic-pertubation-llms/scripts")
from battery import enumerate_full, CORE_IDS

ROOT = "/Volumes/ExternalSSD/diacritic-pertubation-llms"
OUT_JSON = ROOT + "/paper/tables/openai_behavioral.json"
OUT_MD = ROOT + "/paper/tables/openai_behavioral.md"
PROG = ROOT + "/paper/tables/openai_behavioral.progress.log"

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
MODELS = ["gpt-4o", "gpt-4.1", "gpt-5", "gpt-5.4", "gpt-4o-mini",
          "gpt-5-mini", "o3", "o4-mini"]
SWEEP_TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0]
SWEEP_SAMPLES = 2
MAX_TOKENS = 200

# ---- transport (curl subprocess; Python sockets are broken here) -----------
import subprocess

URL = "https://api.openai.com/v1/chat/completions"
_HAVE_SDK = False  # SDK uses Python sockets -> Errno 9; do not use.


class ParamError(Exception):
    """A 400 indicating an unsupported parameter (temperature / max_tokens)."""


def _is_param_error(msg):
    m = (msg or "").lower()
    return (
        ("temperature" in m and ("unsupported" in m or "does not support" in m
                                 or "not support" in m or "only the default" in m
                                 or "unsupported_value" in m))
        or "max_completion_tokens" in m
        or ("max_tokens" in m and "not supported" in m)
        or "unsupported_parameter" in m
    )


def _call(payload):
    """HTTP POST via curl. Raises ParamError on a 400 about an unsupported
    param; RuntimeError on any other API error; transport errors propagate."""
    args = [
        "curl", "-sS", "-m", "180", "-w", "\n__HTTP_STATUS__%{http_code}",
        URL,
        "-H", f"Authorization: Bearer {OPENAI_KEY}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=200)
    raw = proc.stdout
    marker = "\n__HTTP_STATUS__"
    if marker in raw:
        body, status = raw.rsplit(marker, 1)
        status = status.strip()
    else:
        body, status = raw, ""
    if proc.returncode != 0 and not body:
        raise RuntimeError(f"curl rc={proc.returncode}: {proc.stderr[:200]}")
    try:
        d = json.loads(body)
    except Exception:
        raise RuntimeError(f"non-JSON (status={status}): {body[:200] or proc.stderr[:200]}")
    if "error" in d:
        err = d["error"].get("message", str(d["error"]))
        if status == "400" and _is_param_error(err):
            raise ParamError(err)
        raise RuntimeError(f"{status}: {err[:200]}")
    return d["choices"][0]["message"]["content"]


def chat(model, prompt, temperature=0.0, force_completion_tokens=False, want_temp=True):
    """Make one call. Returns (text, used_completion_tokens, accepts_temperature).

    Strategy: build a standard payload (temperature + max_tokens). On a
    ParamError, retry without temperature and with max_completion_tokens.
    force_completion_tokens lets a model that already failed skip the doomed
    first attempt.
    """
    msgs = [{"role": "user", "content": prompt}]

    if not force_completion_tokens:
        payload = {"model": model, "messages": msgs, "max_tokens": MAX_TOKENS}
        if want_temp:
            payload["temperature"] = temperature
        try:
            txt = _call(payload)
            return txt, False, want_temp
        except ParamError:
            pass  # fall through to the reasoning-model variant

    # Reasoning / restricted variant: no temperature, max_completion_tokens.
    payload = {"model": model, "messages": msgs, "max_completion_tokens": MAX_TOKENS}
    txt = _call(payload)
    return txt, True, False


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        with open(PROG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def checkpoint(out):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_JSON)


def main():
    items = enumerate_full()
    core_items = [it for it in items if it["id"] in CORE_IDS]
    log(f"battery: {len(items)} items, {len(core_items)} core; SDK={_HAVE_SDK}")

    out = {"platform": "openai", "models": MODELS, "results": {}}
    sweep_skipped = []
    total_calls = 0
    open(PROG, "w").close()  # reset progress log

    for model in MODELS:
        res = {}
        out["results"][model] = res
        model_ok = False
        accepts_temp = True            # discovered on first successful call
        comp_tokens = False            # whether this model needs max_completion_tokens

        # ---- temp-0 default pass over ALL 46 items ----
        for it in items:
            entry = {"register": it["register"], "family": it["family"], "cond": it["cond"]}
            try:
                txt, comp_tokens, accepts_temp = chat(
                    model, it["text"], temperature=0.0,
                    force_completion_tokens=comp_tokens,
                    want_temp=accepts_temp,
                )
                entry["temp0"] = (txt or "").strip()
                model_ok = True
                total_calls += 1
                log(f"[{model}/{it['id']}] temp0 ok ({len(entry['temp0'])} chars)"
                    + ("" if accepts_temp else " [no-temp]"))
            except Exception as e:
                entry["temp0"] = None
                entry["error"] = str(e)[:200]
                total_calls += 1
                log(f"[{model}/{it['id']}] temp0 ERR {str(e)[:140]}")
            res[it["id"]] = entry
            time.sleep(0.15)
            checkpoint(out)  # frequent checkpoint within the temp-0 pass too

        # ---- temperature sweep over CORE_IDS only ----
        if not accepts_temp:
            # Model rejects temperature -> only the default pass; skip sweep.
            sweep_skipped.append(model)
            for it in core_items:
                res[it["id"]]["sweep_unsupported"] = True
            log(f"[{model}] temperature unsupported -> SKIP sweep")
        elif not model_ok:
            # Whole model failed; nothing to sweep.
            log(f"[{model}] no successful temp-0 call -> skipping sweep")
        else:
            for it in core_items:
                sweep = {}
                for t in SWEEP_TEMPS:
                    samples = []
                    for s in range(SWEEP_SAMPLES):
                        try:
                            txt, comp_tokens, accepts_temp = chat(
                                model, it["text"], temperature=t,
                                force_completion_tokens=comp_tokens,
                                want_temp=True,
                            )
                            samples.append((txt or "").strip())
                            total_calls += 1
                        except ParamError as e:
                            # Surfaced mid-sweep: abandon sweep for this model.
                            total_calls += 1
                            log(f"[{model}/{it['id']}@{t}] temp unsupported mid-sweep")
                            samples.append({"error": str(e)[:200]})
                        except Exception as e:
                            total_calls += 1
                            samples.append({"error": str(e)[:200]})
                            log(f"[{model}/{it['id']}@{t}#{s}] ERR {str(e)[:120]}")
                        time.sleep(0.15)
                    sweep[str(t)] = samples
                res[it["id"]]["sweep"] = sweep
                log(f"[{model}/{it['id']}] sweep done "
                    f"({sum(len(v) for v in sweep.values())} samples)")
                checkpoint(out)

        checkpoint(out)
        log(f"=== checkpoint after {model} (cumulative calls={total_calls}) ===")

    out["sweep_skipped"] = sweep_skipped
    out["total_calls"] = total_calls
    checkpoint(out)
    write_md(out, sweep_skipped, total_calls)
    log(f"DONE. models={len(MODELS)} total_calls={total_calls} "
        f"sweep_skipped={sweep_skipped}")
    log(f"WROTE {OUT_JSON} {OUT_MD}")


def write_md(out, sweep_skipped, total_calls):
    registers = ["experiential", "nonexp"]
    L = ["# OpenAI Behavioral Sweep (TINE) — coverage summary\n",
         f"Platform: **openai**. Models: {', '.join(MODELS)}.\n",
         f"Total API calls: **{total_calls}**. "
         f"Models that skipped the temp sweep (temperature unsupported): "
         f"**{', '.join(sweep_skipped) if sweep_skipped else 'none'}**.\n",
         "Coverage = count of items with a non-empty temp-0 response / "
         "items attempted, by register.\n",
         "| Model | experiential (ok/total) | nonexp (ok/total) | sweep |",
         "|---|---|---|---|"]
    for model in MODELS:
        res = out["results"].get(model, {})
        cells = {}
        for reg in registers:
            ok = tot = 0
            for v in res.values():
                if v.get("register") != reg:
                    continue
                tot += 1
                if v.get("temp0"):
                    ok += 1
            cells[reg] = f"{ok}/{tot}"
        swept = any("sweep" in v for v in res.values())
        skipped = model in sweep_skipped
        sweep_lbl = "skipped" if skipped else ("yes" if swept else "no")
        L.append(f"| {model} | {cells['experiential']} | {cells['nonexp']} | {sweep_lbl} |")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
