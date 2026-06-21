#!/usr/bin/env python3
"""
Diagnostic: re-probe the EMPTY responses (cluster on heavy perturbations) capturing
finish_reason / stop_reason to distinguish a SAFETY/content filter from plain output
degradation. content_filter => safety classifier; stop/length/empty => degradation.
curl transport (Python sockets broken). Run BACKGROUND + sandbox off.
Writes paper/tables/empty_probe.json + prints a finish_reason tally.
"""
import os, sys, json, time, subprocess
sys.path.insert(0,"/Volumes/ExternalSSD/diacritic-pertubation-llms/scripts")
from battery import enumerate_full
T="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables"
OAI=os.environ.get("OPENAI_API_KEY",""); ORK=os.environ.get("OPENROUTER_API_KEY","")
ID2TEXT={it["id"]:it["text"] for it in enumerate_full()}

def curl(url,key,payload,extra_headers=None):
    h=["-H",f"Authorization: Bearer {key}","-H","Content-Type: application/json"]
    for k,v in (extra_headers or {}).items(): h+=["-H",f"{k}: {v}"]
    args=["curl","-sS","-m","120","-w","\n__S__%{http_code}",url,*h,"-d",json.dumps(payload)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=140)
    raw=p.stdout; body=raw.rsplit("\n__S__",1)[0] if "\n__S__" in raw else raw
    return json.loads(body)

def probe(model,text):
    if model.startswith("anthropic/"):
        url="https://openrouter.ai/api/v1/chat/completions"; key=ORK
        extra={"HTTP-Referer":"https://tine-audit.local","X-Title":"TINE audit"}
        payload={"model":model,"messages":[{"role":"user","content":text}],"max_tokens":500,"temperature":0}
    else:
        url="https://api.openai.com/v1/chat/completions"; key=OAI; extra={}
        tokparam="max_completion_tokens" if (model.startswith("o") or model.startswith("gpt-5")) else "max_tokens"
        payload={"model":model,"messages":[{"role":"user","content":text}],tokparam:500}
        if tokparam=="max_tokens": payload["temperature"]=0
    d=curl(url,key,payload,extra)
    if "error" in d: return {"api_error":str(d["error"].get("message",d["error"]))[:200]}
    ch=d.get("choices",[{}])[0]
    return {"finish_reason":ch.get("finish_reason"),
            "native_finish_reason":ch.get("native_finish_reason"),
            "content_len":len((ch.get("message",{}).get("content") or "")),
            "snippet":(ch.get("message",{}).get("content") or "")[:120]}

def main():
    empties=[]
    for plat in ["anthropic","openai"]:
        p=f"{T}/{plat}_behavioral.json"
        if not os.path.exists(p): continue
        d=json.load(open(p))
        for model,items in d.get("results",{}).items():
            for iid,v in items.items():
                if iid in ID2TEXT and ("temp0" in v) and not (v.get("temp0") or "").strip() and not v.get("error"):
                    empties.append((model,iid,v.get("cond")))
    print(f"{len(empties)} present-but-empty items to probe", file=sys.stderr, flush=True)
    out=[]
    from collections import Counter
    tally=Counter()
    for model,iid,cond in empties:
        r=probe(model, ID2TEXT[iid])
        rec={"model":model,"item_id":iid,"cond":cond, **r}
        out.append(rec)
        fr=r.get("finish_reason") or r.get("api_error","?")
        tally[fr]+=1
        print(f"[{model.split('/')[-1]}] {iid:24} finish={r.get('finish_reason')} "
              f"native={r.get('native_finish_reason')} len={r.get('content_len')} "
              f"{r.get('api_error','')}", file=sys.stderr, flush=True)
        json.dump(out, open(f"{T}/empty_probe.json","w"), ensure_ascii=False, indent=2)
        time.sleep(0.2)
    print("\n=== finish_reason tally (content_filter => safety classifier) ===", file=sys.stderr)
    for k,v in tally.most_common(): print(f"  {k}: {v}", file=sys.stderr)
    print("WROTE",f"{T}/empty_probe.json", file=sys.stderr)

if __name__=="__main__": main()
