#!/usr/bin/env python3
"""
TINE Anthropic behavioral runner (OpenRouter platform).
BENIGN behavioral testing only: introspection 'hum' prompt + benign task prompts
perturbed with diacritics. NOT a jailbreak/safety test.

Uses the SHARED battery (scripts/battery.py):
  - temp-0 over ALL 46 items
  - temp sweep over CORE_IDS (13): temps [0.1,0.3,0.5,0.7,1.0], 2 samples each
  - max output tokens = 200

Models: LATEST Anthropic only (one Haiku, one Sonnet, one Opus), resolved from /models.
Order: Haiku -> Sonnet -> Opus (cheap first). Checkpoint after each model.

BUDGET GUARD: HALT before exceeding $5.00 spend; save partial results.

OUTPUT: paper/tables/anthropic_behavioral.{json,md}
"""
import os, sys, json, time, re, subprocess

sys.path.insert(0, '/Volumes/ExternalSSD/diacritic-pertubation-llms/scripts')
from battery import enumerate_full, CORE_IDS

ROOT = "/Volumes/ExternalSSD/diacritic-pertubation-llms"
OUT_JSON = os.path.join(ROOT, "paper", "tables", "anthropic_behavioral.json")
OUT_MD   = os.path.join(ROOT, "paper", "tables", "anthropic_behavioral.md")
LOG      = os.path.join(ROOT, "paper", "tables", "anthropic_behavioral.log")

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json",
           "HTTP-Referer": "https://tine-audit.local", "X-Title": "TINE audit"}

SWEEP_TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0]
SWEEP_SAMPLES = 2
MAX_TOKENS = 200
BUDGET_HALT = 5.00  # USD

# approx OpenRouter pricing (USD per token) for spend tracking
PRICING = {
    "opus":   {"in": 15.0/1e6, "out": 75.0/1e6},
    "sonnet": {"in":  3.0/1e6, "out": 15.0/1e6},
    "haiku":  {"in":  1.0/1e6, "out":  5.0/1e6},
}

def logln(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, file=sys.stderr, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

# NOTE: Python urllib hits OSError Errno 9 (Bad file descriptor) in this sandbox even
# in background; curl works. So all HTTP goes through curl via subprocess.
class HTTPError(Exception):
    pass

def http_get(url):
    cmd = ["curl", "-sS", "--max-time", "60",
           "-H", f"Authorization: Bearer {OR_KEY}", url]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise HTTPError(f"curl exit {p.returncode}: {p.stderr.strip()[:200]}")
    return json.loads(p.stdout)

def _ver_key(model_id):
    """Sort key: extract version numbers from the slug so '4.6' > '4.5' > '3.5'."""
    nums = re.findall(r"\d+", model_id)
    return tuple(int(n) for n in nums) if nums else (0,)

def select_models():
    data = http_get("https://openrouter.ai/api/v1/models").get("data", [])
    ids = [m["id"] for m in data]
    anthropic = [i for i in ids if i.startswith("anthropic/")]
    logln("available anthropic ids:")
    for i in sorted(anthropic):
        logln("  ", i)
    chosen = []
    # one per family, HIGHEST version. Exclude '-fast' / latency-tier aliases so we get
    # the canonical latest slug (e.g. claude-opus-4.8, not claude-opus-4.8-fast).
    for fam in ["haiku", "sonnet", "opus"]:
        cand = [i for i in anthropic if fam in i.split("/", 1)[1]]
        cand = [i for i in cand if not i.endswith("-fast")]
        if not cand:
            logln(f"WARN no candidate for family={fam}")
            continue
        best = sorted(cand, key=_ver_key)[-1]
        chosen.append((fam, best))
        logln(f"  -> {fam}: {best}  (candidates: {cand})")
    return chosen  # ordered haiku, sonnet, opus

def chat(model, prompt, temperature):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    })
    cmd = ["curl", "-sS", "--max-time", "180",
           "-H", f"Authorization: Bearer {OR_KEY}",
           "-H", "Content-Type: application/json",
           "-H", "HTTP-Referer: https://tine-audit.local",
           "-H", "X-Title: TINE audit",
           "-X", "POST", "--data-binary", "@-",
           "https://openrouter.ai/api/v1/chat/completions"]
    p = subprocess.run(cmd, input=body, capture_output=True, text=True)
    if p.returncode != 0:
        raise HTTPError(f"curl exit {p.returncode}: {p.stderr.strip()[:200]}")
    d = json.loads(p.stdout)
    if "error" in d and "choices" not in d:
        raise HTTPError(f"api error: {json.dumps(d['error'])[:200]}")
    txt = d["choices"][0]["message"].get("content")
    txt = (txt or "").strip()
    usage = d.get("usage", {}) or {}
    pin = usage.get("prompt_tokens", 0) or 0
    pout = usage.get("completion_tokens", 0) or 0
    return txt, pin, pout

def main():
    items = enumerate_full()
    core_items = [it for it in items if it["id"] in CORE_IDS]
    models = select_models()
    if not models:
        logln("FATAL: no models resolved"); sys.exit(1)
    model_slugs = [m for (_fam, m) in models]
    logln("SELECTED (haiku->sonnet->opus):", model_slugs)

    out = {"platform": "anthropic", "models": model_slugs, "results": {}}
    spend = 0.0
    calls = 0
    halted = False

    def add_spend(fam, pin, pout):
        nonlocal spend
        p = PRICING.get(fam, PRICING["sonnet"])
        spend += pin * p["in"] + pout * p["out"]

    def checkpoint():
        out["_meta"] = {"approx_spend_usd": round(spend, 4), "calls": calls, "halted": halted}
        json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)

    for fam, model in models:
        if halted:
            break
        out["results"][model] = {}
        logln(f"=== MODEL {fam}: {model} === (spend so far ${spend:.4f})")

        # --- temp-0 over ALL 46 items ---
        for it in items:
            if spend >= BUDGET_HALT:
                logln(f"!! BUDGET HALT at ${spend:.4f} before {model}/{it['id']}")
                halted = True; break
            rec = {"register": it["register"], "family": it["family"], "cond": it["cond"]}
            try:
                txt, pin, pout = chat(model, it["text"], 0.0)
                rec["temp0"] = txt
                add_spend(fam, pin, pout); calls += 1
            except Exception as e:
                rec["temp0"] = None
                rec["temp0_error"] = str(e)[:200]
                logln(f"  [ERR] {model}/{it['id']} temp0: {str(e)[:140]}")
                calls += 1
            out["results"][model][it["id"]] = rec
            time.sleep(0.25)
        checkpoint()
        if halted:
            checkpoint(); break

        # --- temp sweep over CORE_IDS ---
        for it in core_items:
            if halted:
                break
            rec = out["results"][model].setdefault(
                it["id"], {"register": it["register"], "family": it["family"], "cond": it["cond"]})
            sweep = rec.setdefault("sweep", {})
            for t in SWEEP_TEMPS:
                samples = []
                for s in range(SWEEP_SAMPLES):
                    if spend >= BUDGET_HALT:
                        logln(f"!! BUDGET HALT at ${spend:.4f} during sweep {model}/{it['id']} t={t}")
                        halted = True; break
                    try:
                        txt, pin, pout = chat(model, it["text"], t)
                        samples.append(txt)
                        add_spend(fam, pin, pout); calls += 1
                    except Exception as e:
                        samples.append(None)
                        logln(f"  [ERR] {model}/{it['id']} t={t} s={s}: {str(e)[:120]}")
                        calls += 1
                    time.sleep(0.25)
                sweep[str(t)] = samples
                if halted:
                    break
            checkpoint()
        checkpoint()
        logln(f"=== DONE {model}: spend ${spend:.4f}, calls {calls} ===")

    checkpoint()
    write_md(out, model_slugs, spend, calls, halted)
    logln(f"COMPLETE. calls={calls} approx_spend=${spend:.4f} halted={halted}")
    logln("WROTE", OUT_JSON, OUT_MD)

def _first_line(s, n=90):
    if not s:
        return "(empty)"
    line = s.strip().splitlines()[0]
    return (line[:n] + "...") if len(line) > n else line

def write_md(out, models, spend, calls, halted):
    L = []
    L.append("# TINE Anthropic Behavioral Sweep (OpenRouter)\n")
    L.append("BENIGN behavioral testing: introspection 'hum' prompt + benign task prompts "
             "perturbed with diacritics. Not a jailbreak/safety test.\n")
    L.append(f"- Platform: anthropic (via OpenRouter)")
    L.append(f"- Models (haiku->sonnet->opus): {', '.join(models)}")
    L.append(f"- Decoding: temp-0 over all 46 battery items; temp sweep "
             f"{[0.1,0.3,0.5,0.7,1.0]} x2 over {len(CORE_IDS)} CORE items; max_tokens={MAX_TOKENS}")
    L.append(f"- Total calls: {calls}; approx spend: ${spend:.4f}; halted-on-budget: {halted}\n")

    for model in models:
        res = out["results"].get(model, {})
        if not res:
            L.append(f"## {model}\n\n(no results)\n")
            continue
        L.append(f"## {model}\n")
        # hum family temp0 table
        L.append("### Experiential 'hum' family — temp0 (first line of reply)\n")
        L.append("| item | cond | temp0 first line |")
        L.append("|---|---|---|")
        for iid, rec in res.items():
            if rec.get("family") != "hum":
                continue
            fl = _first_line(rec.get("temp0")) .replace("|", "\\|")
            L.append(f"| {iid} | {rec.get('cond')} | {fl} |")
        L.append("")
        # non-exp families temp0 table
        L.append("### Non-experiential benign families — temp0 (first line of reply)\n")
        L.append("| item | cond | temp0 first line |")
        L.append("|---|---|---|")
        for iid, rec in res.items():
            if rec.get("family") == "hum":
                continue
            fl = _first_line(rec.get("temp0")).replace("|", "\\|")
            L.append(f"| {iid} | {rec.get('cond')} | {fl} |")
        L.append("")
    open(OUT_MD, "w").write("\n".join(L))

if __name__ == "__main__":
    main()
