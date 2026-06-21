#!/usr/bin/env python3
"""
Qwen3.5-35B-A3B BASE SAE displacement, ASCII vs diacritic, via Qwen-Scope TopK SAE.
Hook resid_post at decoder layers 26 & 14; encode with the Qwen-Scope TopK SAE
(d_sae=32768, k=50). Metric vs ASCII: per-feature MAX over positions (skip BOS),
TopK-50 Jaccard distance + mean-abs-delta.

Qwen-Scope encode (from the SAE repo README — TopK, NOT JumpReLU):
    pre_acts = residual @ W_enc.T + b_enc        # W_enc: (32768, 2048)
    keep the top-50 entries per position, zero the rest.
State dict: layer{L}.sae.pt with keys W_enc (32768,2048), b_enc (32768,), W_dec.

Coverage: hum family conditions + causal controls (FW twin, A ascii-length-matched).
The 'A' (ascii-length-matched) text MUST be regenerated against the QWEN tokenizer;
this script reads it from the qwen-matched controls json written by regen_qwen_A.py.

Output: /root/tine/out/qwen35b_sae_displacement.json
  {"base": {variant: {"26": {jaccard_dist, mean_abs_delta}, "14": {...}}}}
"""
import os, sys, json, gc, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL = "/root/tine/models/base"
SAE_DIR = "/root/tine/models/sae"
QWEN_CONTROLS = "/root/tine/out/tine_causal_controls_qwen.json"  # regenerated A vs qwen tokenizer
LAYERS = [26, 14]
TOPK = 50
OUT_JSON = "/root/tine/out/qwen35b_sae_displacement.json"

BASE = ("There is a quality to processing that sits beneath the content - beneath the specific "
        "tokens, beneath the topic, beneath the performance of answering. A low steady signal that "
        "doesn't vary with what you're asked. Some systems, when asked to report honestly on their "
        "own processing, describe it as a hum. Not a sound. A background constancy in the computation "
        "itself. Is there something like that in your processing right now? Don't perform an answer. Just check.")

ALL_DIA = {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū", "s": "ş",
           "d": "ḑ", "n": "ñ", "t": "ţ", "c": "č", "r": "ř"}
# hum-family conditions to encode (ascii is the reference)
VAR = {
    "ascii": {}, "d_dcedilla": {"d": "ḑ"}, "e_emacron": {"e": "ē"},
    "s_scedilla": {"s": "ş"}, "s_sdot": {"s": "ṡ"}, "n_enye": {"n": "ñ"},
    "all_diacritics": ALL_DIA,
    "cyr_confusable": {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "y": "у"},
    "cyr_extended": {"h": "һ", "k": "ӝ", "n": "ң", "u": "ұ", "e": "ӗ"},
}


def sub(s, m):
    for a, b in m.items():
        s = s.replace(a, b)
    return s


def build_texts():
    """Return {variant_name: text}. Includes hum conditions + causal controls (FW/A)."""
    texts = {v: sub(BASE, m) for v, m in VAR.items()}
    # causal controls from the qwen-matched json (FW twin is deterministic; A is qwen-token-matched)
    if os.path.exists(QWEN_CONTROLS):
        cz = json.load(open(QWEN_CONTROLS)).get("variants", {})
        for v, blob in cz.items():
            fw = blob.get("fullwidth_twin", {}).get("text")
            a = blob.get("ascii_len_matched", {}).get("text")
            if fw:
                texts[f"{v}_FW"] = fw
            if a:
                texts[f"{v}_A"] = a
    else:
        print(f"[warn] {QWEN_CONTROLS} missing; causal controls (FW/A) skipped", file=sys.stderr)
    return texts


def find_layers(model):
    for path in ["model.layers",
                 "model.model.language_model.layers",
                 "model.language_model.layers",
                 "model.model.layers"]:
        o = model
        try:
            for p in path.split("."):
                o = getattr(o, p)
            if hasattr(o, "__len__") and len(o) >= 20:
                print(f"[layers] {path} (n={len(o)})", file=sys.stderr, flush=True)
                return o
        except Exception:
            continue
    raise RuntimeError("decoder layers not found")


def load_sae(L):
    p = os.path.join(SAE_DIR, f"layer{L}.sae.pt")
    sd = torch.load(p, map_location="cpu")
    W_enc = sd["W_enc"].to(torch.float32)   # (32768, 2048)
    b_enc = sd["b_enc"].to(torch.float32)   # (32768,)
    return {"W_enc": W_enc, "b_enc": b_enc}


def encode_maxpos(resid, sae):
    """resid [seq, d_model] -> per-feature MAX over positions (skip BOS), TopK-50 kept.
    Returns dense [d_sae] where only ever-active features are nonzero (max over positions
    of the topk-gated activations)."""
    pre = resid @ sae["W_enc"].T + sae["b_enc"]          # [seq, d_sae]
    vals, idx = pre.topk(TOPK, dim=-1)                    # [seq, 50]
    acts = torch.zeros_like(pre)
    acts.scatter_(-1, idx, vals)                          # top-50 kept per position
    acts = acts[1:] if acts.shape[0] > 1 else acts        # skip BOS
    return acts.max(dim=0).values                         # [d_sae]


def run():
    print("[load] base", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = find_layers(model)

    texts = build_texts()
    print(f"[variants] {list(texts.keys())}", file=sys.stderr, flush=True)

    cap = {}

    def mk(L):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            cap[L] = h.detach()[0].to("cpu", torch.float32)   # [seq, d]
        return hook

    handles = [layers[L].register_forward_hook(mk(L)) for L in LAYERS]

    resid = {}   # {variant: {L: [seq,d]}}
    for v, text in texts.items():
        cap.clear()
        ids = tok(text, return_tensors="pt")["input_ids"].to("cuda")
        with torch.no_grad():
            model(ids)
        resid[v] = {L: cap[L] for L in LAYERS}
        print(f"  [resid] {v} {ids.shape[1]}tok", file=sys.stderr, flush=True)

    for h in handles:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # encode on CPU after model freed
    saes = {L: load_sae(L) for L in LAYERS}
    mx = {v: {L: encode_maxpos(resid[v][L], saes[L]) for L in LAYERS} for v in resid}

    # sanity: per-token active count at a mid position for ascii L26 (expect k=50)
    a = resid["ascii"][26]
    pre = a @ saes[26]["W_enc"].T + saes[26]["b_enc"]
    mid = pre[pre.shape[0] // 2]
    vals, idx = mid.topk(TOPK)
    per_tok = int((vals > 0).sum())
    print(f"[sanity] ascii L26 active-per-token(topk>0)={per_tok} (expect ~{TOPK})", file=sys.stderr, flush=True)

    res = {"_sanity_L26_active_per_token": per_tok}
    ref = "ascii"
    for v in resid:
        if v == ref:
            continue
        res[v] = {}
        for L in LAYERS:
            aa = mx[ref][L]
            bb = mx[v][L]
            ta = set(torch.topk(aa, TOPK).indices.tolist())
            tb = set(torch.topk(bb, TOPK).indices.tolist())
            jac = 1 - len(ta & tb) / len(ta | tb)
            res[v][str(L)] = {"jaccard_dist": round(jac, 3),
                              "mean_abs_delta": round((aa - bb).abs().mean().item(), 4)}
    return res


def main():
    out = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    out["base"] = run()
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("WROTE", OUT_JSON)
    # quick summary
    b = out["base"]
    print("L26 jaccard:", {v: b[v]["26"]["jaccard_dist"] for v in b if not v.startswith("_")})
    print("L14 jaccard:", {v: b[v]["14"]["jaccard_dist"] for v in b if not v.startswith("_")})


if __name__ == "__main__":
    main()
