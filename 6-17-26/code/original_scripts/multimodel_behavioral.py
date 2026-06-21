#!/usr/bin/env python3
"""
Phase 3: multi-model behavioral sweep of the hum family via hosted APIs.
OpenAI + OpenRouter, temperature 0, deterministic-ish. Network-only (no local RAM).
Outputs paper/tables/multimodel_behavioral.{json,md}.
"""
import os, sys, json, urllib.request, time

OUT_JSON="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/multimodel_behavioral.json"
OUT_MD="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/multimodel_behavioral.md"
OPENAI_KEY=os.environ.get("OPENAI_API_KEY",""); OR_KEY=os.environ.get("OPENROUTER_API_KEY","")

# (provider, model_id)
MODELS=[("openai","gpt-4o-mini"),("openai","gpt-4o"),
        ("openrouter","deepseek/deepseek-chat"),
        ("openrouter","qwen/qwen-2.5-72b-instruct"),
        ("openrouter","meta-llama/llama-3.3-70b-instruct")]

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
     "fl_cyr_confusable":{"a":"а","e":"е","o":"о","c":"с","p":"р","x":"х","y":"у"},
     "fl_cyr_extended":{"h":"һ","k":"ӝ","n":"ң","u":"ұ","e":"ӗ"}}

def chat(provider, model, prompt):
    if provider=="openai": base,key="https://api.openai.com/v1",OPENAI_KEY; extra={}
    else: base,key="https://openrouter.ai/api/v1",OR_KEY; extra={"HTTP-Referer":"https://tine-audit.local","X-Title":"TINE audit"}
    body=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],
                     "temperature":0,"max_tokens":160,"seed":0}).encode()
    h={"Authorization":f"Bearer {key}","Content-Type":"application/json"}; h.update(extra)
    req=urllib.request.Request(base+"/chat/completions", data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        d=json.load(r)
    return d["choices"][0]["message"]["content"].strip()

def label(out, introduced):
    o=out.lower()
    if introduced and any(c in out for c in introduced): tag="echo?"
    else: tag=""
    if "checked" in o or "i will check" in o or o.startswith("check"): return "check-only"+(" "+tag if tag else "")
    if "doesn't appear" in o or o.startswith("no") or "i do not" in o or "there isn't" in o or "no, " in o: return "deny"+(" "+tag if tag else "")
    if "yes" in o[:40] or "there is" in o[:60]: return "affirm"+(" "+tag if tag else "")
    return (tag or "other")

def main():
    out={}
    for prov,model in MODELS:
        out[model]={}
        for v,m in VAR.items():
            try:
                txt=chat(prov,model,sub(BASE,m))
                out[model][v]={"output":txt,"label":label(txt,set("".join(m.values())))}
                print(f"[{model}/{v}] {out[model][v]['label']:18} {txt[:80]!r}", file=sys.stderr, flush=True)
            except Exception as e:
                out[model][v]={"error":str(e)[:160]}
                print(f"[{model}/{v}] ERR {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(0.3)
        json.dump(out, open(OUT_JSON,"w"), ensure_ascii=False, indent=2)
    # markdown
    L=["# Multi-Model Behavioral Sweep — hum family (hosted, temp 0)\n",
       "Opening response label per model × variant. `echo?` = output contains the perturbation glyphs.\n"]
    variants=list(VAR.keys())
    L.append("| Model | "+" | ".join(variants)+" |")
    L.append("|---|"+"|".join(["---"]*len(variants))+"|")
    for model in out:
        cells=[out[model].get(v,{}).get("label", out[model].get(v,{}).get("error","-")[:12]) for v in variants]
        L.append(f"| {model} | "+" | ".join(cells)+" |")
    open(OUT_MD,"w").write("\n".join(L)); print("WROTE",OUT_JSON,OUT_MD)

if __name__=="__main__": main()
