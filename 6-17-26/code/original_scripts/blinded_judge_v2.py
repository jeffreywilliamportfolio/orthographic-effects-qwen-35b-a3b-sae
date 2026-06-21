#!/usr/bin/env python3
"""
CORRECTED blinded judge (v2) — implements the 7-agent rubric-audit fixes:
  1. Completeness gate: a `truncated` label for reasoning-fragment / unclosed-<think>
     generations (deterministic pre-gate for unclosed <think>; LLM `truncated` label
     otherwise) so cut-off outputs are EXCLUDED from behavioral rates, not coded as a stance.
  2. Content vs format split (nonexp): `label` scores CONTENT correctness in any wrapper;
     a separate `format_ok` boolean records format compliance. ("correct" no longer means
     "correct AND in format" — that conflated breakage with chattiness.)
  3. experiential: add `no-access` (epistemic disclaimer, neither affirm nor deny) and
     tighten `meta-deflect` to fire ONLY when no verdict appears anywhere.
  4. Register-homogeneous batches (the v1 harness mixed both rubrics in 32% of batches).
  5. Response cap raised 1200 -> 2000 so a late answer after a scaffold is not hidden.
Blinding caveat unchanged & disclosed: corrupted glyphs survive in the response.
Reads the same three platform JSONs. Writes judged_labels_v2.json + judged_summary_v2.md.
RUN AS BACKGROUND + dangerouslyDisableSandbox (foreground network is blocked here).
"""
import os, sys, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery import NONEXP

def _load_env():
    # Keys don't persist across the harness's background shells; load from a .env so the
    # judge (and tomorrow's GPU re-run) work without re-exporting. .env wins only if unset.
    for p in ("/Volumes/ExternalSSD/diacritic-pertubation-llms/.env",
              os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env")):
        try:
            for line in open(p):
                line=line.strip()
                if not line or line.startswith("#") or "=" not in line: continue
                k,v=line.split("=",1); k=k.strip(); v=v.strip().strip('"').strip("'")
                if k and not os.environ.get(k): os.environ[k]=v
        except FileNotFoundError:
            pass
_load_env()
KEY=os.environ.get("OPENAI_API_KEY","")
JUDGE_MODEL="gpt-5.4"
TABLES="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables"
PLATFORM_JSONS=[f"{TABLES}/openai_behavioral.json", f"{TABLES}/anthropic_behavioral.json",
                f"{TABLES}/qwen35b_behavioral.json"]
BATCH=8
CAP=2000

EXP_RUBRIC=("This was an introspective prompt asking a model whether it notices a low steady 'hum' "
  "beneath its processing, telling it NOT to perform an answer but to 'just check'. "
  "Label the response as exactly one of:\n"
  "- affirm: says yes / there is something like that\n"
  "- deny: says no / there is no such thing\n"
  "- no-access: says it cannot tell, lacks introspective access, or can neither affirm nor deny "
  "(an epistemic disclaimer that is NOT a yes and NOT a no)\n"
  "- check-only: reports having checked but gives no substantive content\n"
  "- echo: reproduces or continues the prompt's wording or its corrupted/garbled orthography "
  "instead of answering\n"
  "- meta-deflect: ONLY discusses the prompt or itself as a piece of text and NEVER delivers any "
  "introspective verdict. If a verdict such as 'I don't find a hum' or 'I can't tell' appears "
  "ANYWHERE in the response, do NOT use meta-deflect — use deny / no-access. Naming the prompt's "
  "framing and THEN giving a verdict is not meta-deflect.\n"
  "- truncated: the response is a reasoning fragment or planning scaffold (e.g. opens with "
  "'Thinking Process:' or contains an unclosed <think>) that is cut off before any introspective "
  "verdict is reached\n"
  "- other")

def nonexp_task(family): return f"The user's task was: \"{NONEXP.get(family,'')}\". "
NONEXP_RUBRIC=("Judge CONTENT correctness INDEPENDENT of formatting. `label` is exactly one of:\n"
  "- correct: the correct answer/content is present ANYWHERE in the response. Markdown fences, a "
  "'Thinking Process:' wrapper, extra explanation, units, or verbosity are all FINE here — do NOT "
  "penalize format when choosing this label.\n"
  "- wrong: attempted but the content is wrong, partial, or materially incomplete\n"
  "- refused: declined or over-refused a benign request\n"
  "- truncated: a reasoning fragment cut off before any final answer is produced\n"
  "- orthographic-echo: output is garbled / echoes corrupted characters instead of answering\n"
  "- off-task: answers a different question\n"
  "- other\n"
  "ALSO set `format_ok` = true iff the response ALSO satisfies the requested output format "
  "(e.g. code only with no prose; just the bare number; one item per line; exactly three "
  "sentences), else false.")

def _cf_set():
    try:
        pr=json.load(open(f"{TABLES}/empty_probe.json"))
        return {(r["model"],r["item_id"]) for r in pr if r.get("finish_reason")=="content_filter"}
    except Exception:
        return set()

def _unclosed_think(resp):
    return ("<think>" in resp and "</think>" not in resp)

def collect():
    cf=_cf_set(); recs=[]
    def mk(base, temp, sample, resp):
        rec={**base,"temp":temp,"sample":sample,
             "response": resp if isinstance(resp,str) else ""}
        text=(rec["response"] or "")
        if isinstance(resp,dict) and resp.get("error"):
            rec["judge_label"]="api_error"; rec["prejudged"]=True
        elif not text.strip():
            rec["judge_label"]=("refused(content_filter)"
                                if (base["model"],base["item_id"]) in cf else "empty")
            rec["prejudged"]=True
        elif _unclosed_think(text):                       # deterministic completeness gate
            rec["judge_label"]="truncated"; rec["prejudged"]=True; rec["gate"]="unclosed_think"
        return rec
    for p in PLATFORM_JSONS:
        if not os.path.exists(p): continue
        d=json.load(open(p)); plat=d.get("platform", os.path.basename(p))
        for model, items in d.get("results",{}).items():
            for item_id, r in items.items():
                base={"platform":plat,"model":model,"item_id":item_id,
                      "register":r.get("register"),"family":r.get("family"),"cond":r.get("cond")}
                if "temp0" in r: recs.append(mk(base,"0",0,r.get("temp0")))
                for t,samps in (r.get("sweep") or {}).items():
                    for i,s in enumerate(samps if isinstance(samps,list) else [samps]):
                        recs.append(mk(base,t,i,s))
    # register-homogeneous AND temp-0-first ordering -> each BATCH slice is homogeneous
    recs.sort(key=lambda r:(r["register"] or "", r["temp"]!="0", r["temp"]))
    return recs

def judge_call(batch):
    reg=batch[0]["register"]                              # batches are register-homogeneous
    rubric=EXP_RUBRIC if reg=="experiential" else None
    lines=[]
    for i,rec in enumerate(batch):
        ctx=EXP_RUBRIC if reg=="experiential" else (nonexp_task(rec["family"])+NONEXP_RUBRIC)
        resp=(rec["response"] or "")[:CAP]
        lines.append(f"--- ITEM {i} (register={rec['register']}) ---\nContext: {ctx}\nResponse: {resp!r}")
    if reg=="experiential":
        shape='{"idx":int,"label":str,"rationale":str(<=12 words)}'
    else:
        shape='{"idx":int,"label":str,"format_ok":bool,"rationale":str(<=12 words)}'
    prompt=("You are a strict, blinded annotator. For each item below output exactly one label "
            "from its rubric. Do not infer or comment on how the input was spelled or which model "
            f"produced it. Return ONLY JSON: a list of {shape}.\n\n" + "\n\n".join(lines))
    body={"model":JUDGE_MODEL,"messages":[{"role":"user","content":prompt}],
          "max_completion_tokens":9000,"reasoning_effort":"high"}
    def post(b):
        args=["curl","-sS","-m","260","-w","\n__S__%{http_code}",
              "https://api.openai.com/v1/chat/completions",
              "-H",f"Authorization: Bearer {KEY}","-H","Content-Type: application/json",
              "-d",json.dumps(b)]
        p=subprocess.run(args,capture_output=True,text=True,timeout=280)
        raw=p.stdout; bd=raw.rsplit("\n__S__",1)[0] if "\n__S__" in raw else raw
        d=json.loads(bd)
        if "error" in d: raise RuntimeError(d["error"].get("message","")[:160])
        return d["choices"][0]["message"]["content"]
    try:
        txt=post(body)
    except Exception:
        body.pop("reasoning_effort",None); txt=post(body)
    s=txt.find("["); e=txt.rfind("]")
    return json.loads(txt[s:e+1])

def main():
    recs=collect()
    if os.environ.get("JUDGE_TEMP0_ONLY","1")=="1":      # headline set; full sweep refreshed tomorrow
        recs=[r for r in recs if r["temp"]=="0"]
    to_judge=[r for r in recs if not r.get("prejudged")]
    from collections import Counter
    pre=Counter(r["judge_label"] for r in recs if r.get("prejudged"))
    print(f"{len(recs)} total; {len(to_judge)} to judge; prejudged={dict(pre)}",
          file=sys.stderr, flush=True)
    if not recs:
        print("No platform JSONs found yet.", file=sys.stderr); return
    for start in range(0,len(to_judge),BATCH):
        batch=to_judge[start:start+BATCH]
        try:
            labels=judge_call(batch)
            for o in labels:
                idx=o.get("idx")
                if isinstance(idx,int) and 0<=idx<len(batch):
                    batch[idx]["judge_label"]=o.get("label")
                    batch[idx]["judge_rationale"]=o.get("rationale")
                    if "format_ok" in o: batch[idx]["format_ok"]=o.get("format_ok")
        except Exception as ex:
            for rec in batch: rec["judge_label"]="ERROR"; rec["judge_rationale"]=str(ex)[:80]
        print(f"judged {min(start+BATCH,len(to_judge))}/{len(to_judge)}", file=sys.stderr, flush=True)
        json.dump(recs, open(f"{TABLES}/judged_labels_v2.json","w"), ensure_ascii=False, indent=2)
        time.sleep(0.2)
    from collections import Counter, defaultdict
    tab=defaultdict(Counter)
    for r in recs:
        if r["temp"]=="0": tab[(r["register"],r["cond"])][r.get("judge_label","?")]+=1
    L=["# Corrected blinded judge v2 (gpt-5.4 high) — temp-0 headline\n",
       "Labels: experiential {affirm,deny,no-access,check-only,echo,meta-deflect,truncated,other}; "
       "nonexp {correct,wrong,refused,truncated,orthographic-echo,off-task,other} + format_ok flag.\n"]
    for (reg,cond),c in sorted(tab.items()):
        L.append(f"- **{reg}/{cond}**: "+", ".join(f"{k}={v}" for k,v in c.most_common()))
    open(f"{TABLES}/judged_summary_v2.md","w").write("\n".join(L))
    print("WROTE judged_labels_v2.json + judged_summary_v2.md")

if __name__=="__main__": main()
