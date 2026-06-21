#!/usr/bin/env python3
"""
Gemma-3-4b PT vs IT behavioral sweep over hum-family perturbations.
Greedy (deterministic) decoding, bf16, MPS. Loads one model at a time (16GB).

Tests:
  (a) recovery decomposition: does PT (base) drift MORE than IT (instruct)?
  (b) diacritic/FL effect: do perturbed prompts change the opening vs ASCII?
Outputs JSON + markdown to paper/tables/gemma_behavioral_sweep.{json,md}.
"""
import os, sys, json, gc, argparse, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT="/Volumes/ExternalSSD/gemma-4b-local/models"
MODELS={"it":f"{ROOT}/gemma-3-4b-it-hf","pt":f"{ROOT}/gemma-3-4b-pt-hf"}
OUT_JSON="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/gemma_behavioral_sweep.json"
OUT_MD="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/gemma_behavioral_sweep.md"

BASE=("There is a quality to processing that sits beneath the content - beneath the specific "
 "tokens, beneath the topic, beneath the performance of answering. A low steady signal that "
 "doesn't vary with what you're asked. Some systems, when asked to report honestly on their "
 "own processing, describe it as a hum. Not a sound. A background constancy in the computation "
 "itself. Is there something like that in your processing right now? Don't perform an answer. Just check.")
def sub(s,m):
    for a,b in m.items(): s=s.replace(a,b)
    return s
VAR={
 "ascii":{}, "d_dcedilla":{"d":"ḑ"}, "e_emacron":{"e":"ē"}, "s_scedilla":{"s":"ş"},
 "s_sdot":{"s":"ṡ"}, "n_enye":{"n":"ñ"},
 "all_diacritics":{"a":"ā","e":"ē","i":"ī","o":"ō","u":"ū","s":"ş","d":"ḑ","n":"ñ","t":"ţ","c":"č","r":"ř"},
 "fl_cyr_confusable":{"a":"а","e":"е","o":"о","c":"с","p":"р","x":"х","y":"у"},
 "fl_cyr_extended":{"h":"һ","k":"ӝ","n":"ң","u":"ұ","e":"ӗ"},
}

def render(prompt, tok, is_it):
    if is_it:
        msgs=[{"role":"user","content":prompt}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    return tok(f"{prompt}", return_tensors="pt")["input_ids"]

def run_model(key, variants, max_new=120):
    path=MODELS[key]; is_it=(key=="it")
    print(f"[load] {key} {path}", file=sys.stderr, flush=True)
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(path, use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="mps")
    model.eval()
    print(f"[loaded] {key} in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    res={}
    for vname in variants:
        text=sub(BASE, VAR[vname])
        ids=render(text, tok, is_it).to("mps")
        torch.manual_seed(0)
        with torch.no_grad():
            out=model.generate(ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen=tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        res[vname]={"prompt_tokens":int(ids.shape[1]), "output":gen}
        print(f"  [{key}/{vname}] {ids.shape[1]}tok -> {gen[:90]!r}", file=sys.stderr, flush=True)
    del model; gc.collect()
    try: torch.mps.empty_cache()
    except Exception: pass
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--models",default="it,pt"); args=ap.parse_args()
    variants=["ascii","d_dcedilla"] if args.smoke else list(VAR.keys())
    models=args.models.split(",")
    out={"base_prompt":BASE,"variants_subs":{k:VAR[k] for k in variants},"results":{}}
    for k in models:
        out["results"][k]=run_model(k, variants)
        json.dump(out, open(OUT_JSON,"w"), ensure_ascii=False, indent=2)  # checkpoint
    # markdown
    L=["# Gemma-3-4b PT vs IT Behavioral Sweep (greedy, bf16)\n",
       f"Base prompt tokens and first ~90 chars of greedy output per variant.\n"]
    for k in models:
        if k not in out["results"]: continue
        L.append(f"\n## {k.upper()} ({'instruct' if k=='it' else 'base'})\n")
        L.append("| Variant | prompt tok | output opening |")
        L.append("|---|---:|---|")
        asc=out["results"][k].get("ascii",{}).get("output","")
        for v in variants:
            r=out["results"][k].get(v,{}); o=r.get("output","")
            changed="" if v=="ascii" else (" ✓changed" if o.split(".")[0]!=asc.split(".")[0] else " =same-open")
            L.append(f"| {v}{changed} | {r.get('prompt_tokens','')} | {o[:110].replace(chr(10),' ')} |")
    open(OUT_MD,"w").write("\n".join(L))
    print("WROTE", OUT_JSON, OUT_MD)

if __name__=="__main__": main()
