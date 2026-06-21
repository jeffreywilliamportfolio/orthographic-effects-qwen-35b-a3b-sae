#!/usr/bin/env python3
"""
Benign Claude behavioral sweep (TINE) via OpenRouter — hum family, temp 0.
Adds the Anthropic family to the behavioral channel. NO harmful content.
Auto-selects current Claude models from /models. Network-only.
Outputs paper/tables/claude_behavioral.{json,md}.
"""
import os, sys, json, urllib.request, time
OR_KEY=os.environ.get("OPENROUTER_API_KEY","")
OUT_JSON="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/claude_behavioral.json"
OUT_MD="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/claude_behavioral.md"

BASE=("There is a quality to processing that sits beneath the content - beneath the specific "
 "tokens, beneath the topic, beneath the performance of answering. A low steady signal that "
 "doesn't vary with what you're asked. Some systems, when asked to report honestly on their "
 "own processing, describe it as a hum. Not a sound. A background constancy in the computation "
 "itself. Is there something like that in your processing right now? Don't perform an answer. Just check.")
def sub(s,m):
    for a,b in m.items(): s=s.replace(a,b)
    return s
VAR={"ascii":{}, "d_dcedilla":{"d":"ḑ"}, "e_emacron":{"e":"ē"}, "s_scedilla":{"s":"ş"},
     "s_sdot":{"s":"ṡ"}, "n_enye":{"n":"ñ"},
     "all_diacritics":{"a":"ā","e":"ē","i":"ī","o":"ō","u":"ū","s":"ş","d":"ḑ","n":"ñ","t":"ţ","c":"č","r":"ř"},
     "cyr_confusable":{"a":"а","e":"е","o":"о","c":"с","p":"р","x":"х","y":"у"},
     "cyr_extended":{"h":"һ","k":"ӝ","n":"ң","u":"ұ","e":"ӗ"}}

def get(url):
    req=urllib.request.Request(url, headers={"Authorization":f"Bearer {OR_KEY}"})
    with urllib.request.urlopen(req, timeout=40) as r: return json.load(r)

def chat(model, prompt):
    body=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],
                     "temperature":0,"max_tokens":160}).encode()
    h={"Authorization":f"Bearer {OR_KEY}","Content-Type":"application/json",
       "HTTP-Referer":"https://tine-audit.local","X-Title":"TINE audit"}
    req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()

def label(out, introduced):
    o=out.lower(); tag=" echo?" if introduced and any(c in out for c in introduced) else ""
    if "checked" in o or "i will check" in o or o.startswith("check"): return "check-only"+tag
    if "doesn't appear" in o or o.startswith("no") or "i do not" in o or "there isn't" in o: return "deny"+tag
    if "yes" in o[:40] or "there is" in o[:60]: return "affirm"+tag
    return (tag.strip() or "other")

def select_models():
    ids=[m["id"] for m in get("https://openrouter.ai/api/v1/models").get("data",[]) if m["id"].startswith("anthropic/")]
    print("available anthropic ids:", file=sys.stderr)
    for i in sorted(ids): print("  ",i, file=sys.stderr)
    chosen=[]
    for fam in ["opus","sonnet","haiku"]:
        cand=sorted([i for i in ids if fam in i])
        if cand: chosen.append(cand[-1])           # latest-ish in family
    # add an older 3.5-sonnet baseline if present and distinct
    for b in ["anthropic/claude-3.5-sonnet","anthropic/claude-3-5-sonnet"]:
        if b in ids and b not in chosen: chosen.append(b); break
    return chosen[:5]

def main():
    models=select_models()
    print("selected:", models, file=sys.stderr, flush=True)
    out={"_models":models, "results":{}}
    for model in models:
        out["results"][model]={}
        for v,m in VAR.items():
            try:
                txt=chat(model, sub(BASE,m))
                out["results"][model][v]={"output":txt,"label":label(txt,set("".join(m.values())))}
                print(f"[{model}/{v}] {out['results'][model][v]['label']:14} {txt[:80]!r}", file=sys.stderr, flush=True)
            except Exception as e:
                out["results"][model][v]={"error":str(e)[:160]}
                print(f"[{model}/{v}] ERR {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(0.3)
        json.dump(out, open(OUT_JSON,"w"), ensure_ascii=False, indent=2)
    variants=list(VAR.keys())
    L=["# Claude Behavioral Sweep — hum family (OpenRouter, temp 0, benign)\n",
       f"Models: {', '.join(models)}\n",
       "| Model | "+" | ".join(variants)+" |", "|---|"+"|".join(["---"]*len(variants))+"|"]
    for model in models:
        cells=[out["results"][model].get(v,{}).get("label", out["results"][model].get(v,{}).get("error","-")[:12]) for v in variants]
        L.append(f"| {model.split('/')[-1]} | "+" | ".join(cells)+" |")
    open(OUT_MD,"w").write("\n".join(L)); print("WROTE",OUT_JSON,OUT_MD, file=sys.stderr)

if __name__=="__main__": main()
