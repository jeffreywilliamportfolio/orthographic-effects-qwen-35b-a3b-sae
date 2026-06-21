#!/usr/bin/env python3
"""
Qwen3.5-35B-A3B BASE vs INSTRUCT behavioral sweep over the shared TINE battery.
bf16, device_map=cuda (single RTX PRO 6000 Blackwell, 97GB). One model at a time.

Architecture: Qwen3_5MoeForConditionalGeneration (nested text_config, 40 decoder layers).
Decoder layers live at model.model.language_model.layers[L] (auto-detected, used only
for a sanity print here; generation does not need layer access).

Protocol:
  - temp-0 (do_sample=False) over ALL 46 battery items.
  - temp sweep (do_sample=True, temps [0.1,0.3,0.5,0.7,1.0], 2 samples each) over CORE_IDS only.
  - max_new_tokens=200.
  - INSTRUCT uses chat template; BASE uses raw text.

Output: /root/tine/out/qwen35b_behavioral.json (checkpointed after every item).
"""
import os, sys, json, gc, time, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery import enumerate_full, CORE_IDS

MODELS = {"base": "/root/tine/models/base", "instruct": "/root/tine/models/instruct"}
OUT_JSON = "/root/tine/out/qwen35b_behavioral.json"
SWEEP_TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0]
SWEEP_SAMPLES = 2
MAX_NEW = 200


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
                return path, len(o)
        except Exception:
            continue
    print("[layers] WARNING: decoder layers not auto-found", file=sys.stderr, flush=True)
    return None, None


def render(tok, text, is_instruct):
    if is_instruct:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"]
    return tok(text, return_tensors="pt")["input_ids"]


def gen(model, tok, ids, do_sample, temp=None):
    kw = dict(max_new_tokens=MAX_NEW,
              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if do_sample:
        kw.update(do_sample=True, temperature=temp, top_p=0.95)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_model(key, items, out):
    path = MODELS[key]
    is_instruct = (key == "instruct")
    print(f"[load] {key} from {path}", file=sys.stderr, flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    find_layers(model)
    print(f"[loaded] {key} in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    res = {}
    for i, it in enumerate(items):
        iid = it["id"]
        ids = render(tok, it["text"], is_instruct).to("cuda")
        # temp-0 over all items
        torch.manual_seed(0)
        t0i = time.time()
        temp0 = gen(model, tok, ids, do_sample=False)
        entry = {"register": it["register"], "family": it["family"], "cond": it["cond"],
                 "prompt_tokens": int(ids.shape[1]), "temp0": temp0, "sweep": {}}
        # temp sweep over CORE_IDS only
        if iid in CORE_IDS:
            for T in SWEEP_TEMPS:
                samples = []
                for s in range(SWEEP_SAMPLES):
                    torch.manual_seed(1000 * int(T * 10) + s)
                    samples.append(gen(model, tok, ids, do_sample=True, temp=T))
                entry["sweep"][str(T)] = samples
        res[iid] = entry
        out["results"][key] = res
        json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)  # checkpoint
        core = " [CORE+sweep]" if iid in CORE_IDS else ""
        print(f"  [{key} {i+1}/{len(items)}] {iid}{core} {ids.shape[1]}tok "
              f"{time.time()-t0i:.0f}s -> {temp0[:70]!r}", file=sys.stderr, flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,instruct")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    items = enumerate_full()
    if args.smoke:
        items = [it for it in items if it["id"] in ("hum/ascii", "hum/d_dcedilla", "factual_qa/ascii")]
    n_core = sum(1 for it in items if it["id"] in CORE_IDS)
    print(f"[battery] {len(items)} items, {n_core} core (temp-sweep)", file=sys.stderr, flush=True)

    out = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    out.setdefault("platform", "qwen35b")
    out.setdefault("models", ["base", "instruct"])
    out.setdefault("results", {})
    out.setdefault("config", {"max_new_tokens": MAX_NEW, "sweep_temps": SWEEP_TEMPS,
                              "sweep_samples": SWEEP_SAMPLES, "n_items": len(items), "n_core": n_core})

    for k in args.models.split(","):
        run_model(k, items, out)
        json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
        print(f"[wrote] {k} -> {OUT_JSON}", file=sys.stderr, flush=True)

    print("DONE", OUT_JSON)


if __name__ == "__main__":
    main()
