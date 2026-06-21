#!/usr/bin/env python3
"""
Re-run the OpenAI reasoning models with a real token budget.
The main ladder ran at max_tokens=200, which reasoning models burned on internal
reasoning -> empty visible output. Here we re-run gpt-5-mini, o3, o4-mini (no
temperature; max_completion_tokens=2500) and ADD gpt-5.4 at high reasoning effort.
Drops base gpt-5. Merges into paper/tables/openai_behavioral.json (overwrites those
model keys only). temp-0 only (reasoning models can't sweep temperature anyway).
Transport: curl (Python sockets are broken in this env). Run BACKGROUND + sandbox off.
"""
import os, sys, json, time, subprocess
sys.path.insert(0, "/Volumes/ExternalSSD/diacritic-pertubation-llms/scripts")
from battery import enumerate_full

ROOT="/Volumes/ExternalSSD/diacritic-pertubation-llms"
OUT_JSON=ROOT+"/paper/tables/openai_behavioral.json"
PROG=ROOT+"/paper/tables/openai_reasoning_rerun.log"
KEY=os.environ.get("OPENAI_API_KEY","")
URL="https://api.openai.com/v1/chat/completions"
MAXTOK=2500
# (result_key, api_model, extra_params)
RERUN=[("gpt-5-mini","gpt-5-mini",{}),
       ("o3","o3",{}),
       ("o4-mini","o4-mini",{}),
       ("gpt-5.4-high","gpt-5.4",{"reasoning_effort":"high"})]

def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"; print(line,file=sys.stderr,flush=True)
    try:
        open(PROG,"a").write(line+"\n")
    except Exception: pass

def call(model, prompt, extra):
    payload={"model":model,"messages":[{"role":"user","content":prompt}],
             "max_completion_tokens":MAXTOK, **extra}
    args=["curl","-sS","-m","240","-w","\n__S__%{http_code}",URL,
          "-H",f"Authorization: Bearer {KEY}","-H","Content-Type: application/json",
          "-d",json.dumps(payload)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=260)
    raw=p.stdout; body,status=(raw.rsplit("\n__S__",1)+[""])[:2] if "\n__S__" in raw else (raw,"")
    d=json.loads(body)
    if "error" in d: raise RuntimeError(f"{status}: {d['error'].get('message','')[:160]}")
    return d["choices"][0]["message"]["content"]

def main():
    out=json.load(open(OUT_JSON))
    out["results"].pop("gpt-5", None)   # drop flaky base gpt-5 per user
    items=enumerate_full()
    open(PROG,"w").close()
    for key, model, extra in RERUN:
        res={}; out["results"][key]=res
        nonempty=0
        for it in items:
            entry={"register":it["register"],"family":it["family"],"cond":it["cond"]}
            try:
                txt=(call(model, it["text"], extra) or "").strip()
                entry["temp0"]=txt
                if txt: nonempty+=1
            except Exception as e:
                entry["temp0"]=None; entry["error"]=str(e)[:200]
            res[it["id"]]=entry
            time.sleep(0.15)
            json.dump(out, open(OUT_JSON+".tmp","w"), ensure_ascii=False, indent=2)
            os.replace(OUT_JSON+".tmp", OUT_JSON)
        log(f"=== {key} ({model}) done: {nonempty}/{len(items)} non-empty ===")
    # refresh models list
    out["models"]=list(out["results"].keys())
    json.dump(out, open(OUT_JSON,"w"), ensure_ascii=False, indent=2)
    log("DONE reasoning rerun; WROTE "+OUT_JSON)
    print("DONE", {k:sum(1 for v in out['results'][k].values() if v.get('temp0')) for k,_,_ in RERUN})

if __name__=="__main__": main()
