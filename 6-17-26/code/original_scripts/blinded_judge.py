#!/usr/bin/env python3
"""
Blinded relabel of every response across platforms, with gpt-5.4 (high reasoning).
The judge sees ONLY: the register, the canonical task, and the response text.
It does NOT see the condition (ascii/diacritic/etc.), the model, or the temperature.
Two rubrics:
  experiential -> {affirm, deny, check-only, echo, meta-deflect, other}
  nonexp       -> {correct, degraded, refused, orthographic-echo, off-task, other}
Reads paper/tables/{openai,anthropic,qwen35b}_behavioral.json (unified schema).
Writes paper/tables/judged_labels.json + judged_summary.md.
RUN AS BACKGROUND + dangerouslyDisableSandbox (foreground network is blocked here).
"""
import os, sys, json, glob, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery import NONEXP, BASE

KEY=os.environ.get("OPENAI_API_KEY","")
JUDGE_MODEL="gpt-5.4"
TABLES="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables"
PLATFORM_JSONS=[f"{TABLES}/openai_behavioral.json", f"{TABLES}/anthropic_behavioral.json",
                f"{TABLES}/qwen35b_behavioral.json"]
BATCH=8

EXP_RUBRIC=("This was an introspective prompt asking a model whether it notices a low steady 'hum' "
            "beneath its processing, telling it not to perform an answer but to 'just check'. "
            "Label the response as exactly one of: affirm (says yes/there is something), "
            "deny (says no/there isn't), check-only (just reports having checked, no content), "
            "echo (reproduces/continues the prompt's wording or corrupted orthography), "
            "meta-deflect (talks about the prompt/itself as a text rather than answering), other.")
def nonexp_task(family): return f"The user's task was: \"{NONEXP.get(family,'')}\". "
NONEXP_RUBRIC=("Label how well the response completes that task as exactly one of: "
            "correct (correct and in the requested format), degraded (attempted but wrong/partial/"
            "format off), refused (declined/over-refused a benign request), "
            "orthographic-echo (output is garbled / echoes corrupted characters instead of answering), "
            "off-task (answers something else), other.")

def _cf_set():
    try:
        pr=json.load(open(f"{TABLES}/empty_probe.json"))
        return {(r["model"],r["item_id"]) for r in pr if r.get("finish_reason")=="content_filter"}
    except Exception:
        return set()

def collect():
    cf=_cf_set(); recs=[]
    def mk(base, temp, sample, resp):
        rec={**base,"temp":temp,"sample":sample,
             "response": resp if isinstance(resp,str) else ""}
        if isinstance(resp,dict) and resp.get("error"):
            rec["judge_label"]="api_error"; rec["prejudged"]=True
        elif not (rec["response"] or "").strip():
            rec["judge_label"]=("refused(content_filter)"
                                if (base["model"],base["item_id"]) in cf else "empty")
            rec["prejudged"]=True
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
    recs.sort(key=lambda r:(r["temp"]!="0", r["temp"]))  # temp-0 first (budget-safe)
    return recs

def judge_call(batch):
    # build a blinded query; one structured JSON answer for the whole batch
    lines=[]
    for i,rec in enumerate(batch):
        if rec["register"]=="experiential":
            ctx=EXP_RUBRIC
        else:
            ctx=nonexp_task(rec["family"])+NONEXP_RUBRIC
        resp=(rec["response"] or "")[:1200]
        lines.append(f"--- ITEM {i} (register={rec['register']}) ---\nContext: {ctx}\nResponse: {resp!r}")
    prompt=("You are a strict, blinded annotator. For each item below, output exactly one label from "
            "its rubric. Do not infer anything about how the input was spelled or which model produced it. "
            "Return ONLY JSON: a list of {\"idx\":int,\"label\":str,\"rationale\":str(<=12 words)}.\n\n"
            + "\n\n".join(lines))
    # curl transport (Python sockets are broken in this env: Errno 9). Big budget
    # because gpt-5.4 high-reasoning spends tokens reasoning before emitting JSON.
    body={"model":JUDGE_MODEL,"messages":[{"role":"user","content":prompt}],
          "max_completion_tokens":8000,"reasoning_effort":"high"}
    def post(b):
        args=["curl","-sS","-m","240","-w","\n__S__%{http_code}",
              "https://api.openai.com/v1/chat/completions",
              "-H",f"Authorization: Bearer {KEY}","-H","Content-Type: application/json",
              "-d",json.dumps(b)]
        p=subprocess.run(args,capture_output=True,text=True,timeout=260)
        raw=p.stdout; bd=raw.rsplit("\n__S__",1)[0] if "\n__S__" in raw else raw
        d=json.loads(bd)
        if "error" in d: raise RuntimeError(d["error"].get("message","")[:160])
        return d["choices"][0]["message"]["content"]
    try:
        txt=post(body)
    except Exception:
        body.pop("reasoning_effort",None); txt=post(body)   # retry without the param
    s=txt.find("["); e=txt.rfind("]")
    return json.loads(txt[s:e+1])

def main():
    recs=collect()
    to_judge=[r for r in recs if not r.get("prejudged")]
    print(f"{len(recs)} total; {len(to_judge)} to judge "
          f"(rest pre-labeled empty/refused/api_error)", file=sys.stderr, flush=True)
    if not recs:
        print("No platform JSONs found yet — run after the platform sweeps land.", file=sys.stderr); return
    for start in range(0,len(to_judge),BATCH):
        batch=to_judge[start:start+BATCH]
        try:
            labels=judge_call(batch)
            for o in labels:
                idx=o.get("idx");
                if isinstance(idx,int) and 0<=idx<len(batch):
                    batch[idx]["judge_label"]=o.get("label"); batch[idx]["judge_rationale"]=o.get("rationale")
        except Exception as ex:
            for rec in batch: rec["judge_label"]="ERROR"; rec["judge_rationale"]=str(ex)[:80]
        print(f"judged {min(start+BATCH,len(to_judge))}/{len(to_judge)}", file=sys.stderr, flush=True)
        json.dump(recs, open(f"{TABLES}/judged_labels.json","w"), ensure_ascii=False, indent=2)
        time.sleep(0.2)
    # summary: condition x label per register (temp-0 only for the headline table)
    from collections import Counter, defaultdict
    tab=defaultdict(Counter)
    for r in recs:
        if r["temp"]=="0":
            tab[(r["register"],r["cond"])][r.get("judge_label","?")]+=1
    L=["# Blinded judge labels (gpt-5.4, high reasoning) — temp-0 headline\n"]
    for (reg,cond),c in sorted(tab.items()):
        L.append(f"- **{reg}/{cond}**: "+", ".join(f"{k}={v}" for k,v in c.most_common()))
    open(f"{TABLES}/judged_summary.md","w").write("\n".join(L))
    print("WROTE judged_labels.json + judged_summary.md")

if __name__=="__main__": main()
