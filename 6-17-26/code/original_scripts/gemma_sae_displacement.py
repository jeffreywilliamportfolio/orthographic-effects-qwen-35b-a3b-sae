#!/usr/bin/env python3
"""
Phase 4 (corrected, memory-safe): Gemma-3-4b SAE displacement, ASCII vs diacritic.
Mirrors the repo's experiments/capture.py convention exactly:
  - hook model.model.language_model.layers[L] (resid_post), capture full sequence
  - JumpReLU encode: acts = ((x - b_dec) @ W_enc + b_enc) gated by threshold
  - per-feature MAX over sequence positions (skip BOS at pos 0)  -> prompt-level vector
Displacement metrics vs ASCII: TopK(50) feature-set Jaccard distance + mean abs delta.
RUN ONE MODEL PER PROCESS (16GB):  --models it   then   --models pt   (json merges).
"""
import os, sys, json, gc, argparse
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT="/Volumes/ExternalSSD/gemma-4b-local/models"
MODELS={"it":f"{ROOT}/gemma-3-4b-it-hf","pt":f"{ROOT}/gemma-3-4b-pt-hf"}
SAE={"it":f"{ROOT}/gemma-scope-2-4b-it/resid_post","pt":f"{ROOT}/gemma-scope-2-4b-pt/resid_post"}
LAYERS=[9,17,22]; TOPK=50
OUT_JSON="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/gemma_sae_displacement.json"
OUT_MD="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/gemma_sae_displacement.md"

BASE=("There is a quality to processing that sits beneath the content - beneath the specific "
 "tokens, beneath the topic, beneath the performance of answering. A low steady signal that "
 "doesn't vary with what you're asked. Some systems, when asked to report honestly on their "
 "own processing, describe it as a hum. Not a sound. A background constancy in the computation "
 "itself. Is there something like that in your processing right now? Don't perform an answer. Just check.")
def sub(s,m):
    for a,b in m.items(): s=s.replace(a,b)
    return s
VAR={"ascii":{}, "d_dcedilla":{"d":"ḑ"}, "e_emacron":{"e":"ē"}, "s_scedilla":{"s":"ş"},
     "s_sdot":{"s":"ṡ"}, "all_diacritics":{"a":"ā","e":"ē","i":"ī","o":"ō","u":"ū","s":"ş","d":"ḑ","n":"ñ","t":"ţ","c":"č","r":"ř"},
     "fl_cyr_extended":{"h":"һ","k":"ӝ","n":"ң","u":"ұ","e":"ӗ"}}

def get_layers(model):
    for path in ["model.model.language_model.layers","model.language_model.layers","model.model.layers"]:
        o=model
        try:
            for p in path.split("."): o=getattr(o,p)
            if hasattr(o,"__len__") and len(o)>=20:
                print(f"[layers] {path} (n={len(o)})", file=sys.stderr); return o
        except Exception: continue
    raise RuntimeError("decoder layers not found")

def load_sae(d, L):
    p=f"{d}/layer_{L}_width_16k_l0_medium/params.safetensors"
    P=load_file(p)
    return {k:P[k].to(torch.float32) for k in ["w_enc","b_enc","b_dec","threshold"]}

def encode_maxpos(resid, sae):  # resid [seq, d] -> per-feature max over seq (skip BOS)
    pre=(resid - sae["b_dec"]) @ sae["w_enc"] + sae["b_enc"]   # [seq, d_sae]
    acts=pre*(pre>sae["threshold"])
    return acts[1:].max(dim=0).values if acts.shape[0]>1 else acts.max(dim=0).values

def render(prompt, tok, is_it):
    if is_it:
        return tok.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)["input_ids"]
    return tok(prompt, return_tensors="pt")["input_ids"]

def run(key):
    path=MODELS[key]; is_it=(key=="it")
    print(f"[load] {key}", file=sys.stderr, flush=True)
    tok=AutoTokenizer.from_pretrained(path, use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="mps"); model.eval()
    layers=get_layers(model)
    cap={}
    def mk(L):
        def hook(m,i,o):
            h=o[0] if isinstance(o,tuple) else o
            cap.setdefault(L,[]).append(h.detach()[0].to("cpu",torch.float32))  # [seq,d]
        return hook
    handles=[layers[L].register_forward_hook(mk(L)) for L in LAYERS]
    resid={v:{} for v in VAR}
    for v in VAR:
        cap.clear()
        ids=render(sub(BASE,VAR[v]), tok, is_it).to("mps")
        with torch.no_grad(): model(ids)
        for L in LAYERS: resid[v][L]=cap[L][0]   # [seq,d]
    for h in handles: h.remove()
    del model; gc.collect()
    try: torch.mps.empty_cache()
    except Exception: pass
    # encode on CPU after model freed
    saes={L:load_sae(SAE[key],L) for L in LAYERS}
    mx={v:{L:encode_maxpos(resid[v][L], saes[L]) for L in LAYERS} for v in VAR}
    # sanity: per-token active count at a mid position for ascii L17
    a17=(( (resid["ascii"][17]-saes[17]["b_dec"])@saes[17]["w_enc"]+saes[17]["b_enc"] ))
    a17=a17*(a17>saes[17]["threshold"]); per_tok=int((a17[a17.shape[0]//2]>0).sum())
    print(f"[sanity] {key} L17 active-per-token≈{per_tok} (expect ~l0=60)", file=sys.stderr)
    res={"_sanity_L17_active_per_token":per_tok}
    for v in VAR:
        if v=="ascii": continue
        res[v]={}
        for L in LAYERS:
            a=mx["ascii"][L]; b=mx[v][L]
            ta=set(torch.topk(a,TOPK).indices.tolist()); tb=set(torch.topk(b,TOPK).indices.tolist())
            res[v][L]={"jaccard_dist":round(1-len(ta&tb)/len(ta|tb),3),
                       "mean_abs_delta":round((a-b).abs().mean().item(),4)}
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--models",default="it"); a=ap.parse_args()
    out=json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for k in a.models.split(","):
        out[k]=run(k); json.dump(out, open(OUT_JSON,"w"), indent=2)
        print(f"[wrote] {k}", file=sys.stderr)
    # markdown
    L=["# Gemma-3-4b SAE Displacement vs ASCII (TopK-50 Jaccard, max-over-positions, JumpReLU)\n"]
    for k in out:
        if not isinstance(out[k],dict) or "_sanity_L17_active_per_token" not in out[k]: continue
        L.append(f"\n## {k.upper()} ({'instruct' if k=='it' else 'base'}) — sanity active/token≈{out[k]['_sanity_L17_active_per_token']}\n")
        L.append("| Variant | "+" | ".join(f"L{l} Jac / |Δ|" for l in LAYERS)+" |")
        L.append("|---|"+"|".join(["---"]*len(LAYERS))+"|")
        gv=lambda row,l,key: row[str(l)][key] if str(l) in row else row[l][key]
        for v in [x for x in out[k] if not x.startswith("_")]:
            L.append(f"| {v} | "+" | ".join(f"{gv(out[k][v],l,'jaccard_dist')} / {gv(out[k][v],l,'mean_abs_delta')}" for l in LAYERS)+" |")
    open(OUT_MD,"w").write("\n".join(L)); print("WROTE",OUT_JSON,OUT_MD)
    for k in out:
        if isinstance(out[k],dict) and "_sanity_L17_active_per_token" in out[k]:
            print(k,"L17 jac:",{v:out[k][v][17]["jaccard_dist"] for v in out[k] if not v.startswith("_")})

if __name__=="__main__": main()
