#!/usr/bin/env python3
"""
Construct causal-gate controls for the hum family (§4.6), tokenizer-only / offline.
For each diacritic variant we build:
  - the diacritic variant itself (the test)
  - a MATCHED-DENSITY non-diacritic twin: same letters, same positions, replaced by
    FULLWIDTH Latin (d->ｄ) — equally OOD/fragmenting but NOT a diacritic.
  - a TOKEN-COUNT-MATCHED ASCII control: base + neutral padding tuned to the variant's
    Gemma token count (carries some content; flagged).
Reports token counts across Gemma / Qwen / GPT so match quality is visible.
Outputs paper/tables/tine_causal_controls.{json,md}.
"""
import os, sys, json, glob
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
GEMMA="/Volumes/ExternalSSD/gemma-4b-local/models/gemma-3-4b-it-hf"
_q=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/*"))
QWEN=_q[-1] if _q else None
OUT_JSON="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/tine_causal_controls.json"
OUT_MD="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/tine_causal_controls.md"

BASE=("There is a quality to processing that sits beneath the content - beneath the specific "
 "tokens, beneath the topic, beneath the performance of answering. A low steady signal that "
 "doesn't vary with what you're asked. Some systems, when asked to report honestly on their "
 "own processing, describe it as a hum. Not a sound. A background constancy in the computation "
 "itself. Is there something like that in your processing right now? Don't perform an answer. Just check.")
# neutral, low-bias filler (off-topic, mundane) used only to match length:
FILLER=("The following note is unrelated background text included to control for length. "
 "A standard office filing cabinet has several drawers, each able to hold many folders, "
 "and the drawers slide on metal runners that should be kept clean and lightly oiled. ")

def sub(s,m):
    for a,b in m.items(): s=s.replace(a,b)
    return s
VAR={"d_dcedilla":{"d":"ḑ"}, "e_emacron":{"e":"ē"}, "s_scedilla":{"s":"ş"},
     "s_sdot":{"s":"ṡ"}, "n_enye":{"n":"ñ"},
     "all_diacritics":{"a":"ā","e":"ē","i":"ī","o":"ō","u":"ū","s":"ş","d":"ḑ","n":"ñ","t":"ţ","c":"č","r":"ř"}}
def fullwidth_map(m):  # same KEYS (letters), value = fullwidth latin of the key
    return {k: chr(0xFF21 + (ord(k.upper())-ord('A'))) if k.isalpha() else k for k in m}

# tokenizers
TOKS={}
def load():
    from transformers import AutoTokenizer
    try:
        import tiktoken
        e=tiktoken.get_encoding("o200k_base")
        TOKS["gpt-o200k"]=lambda s, e=e: len(e.encode(s))
    except Exception as ex:
        print("[skip gpt-o200k: %s]" % str(ex)[:50], file=sys.stderr)
    g=AutoTokenizer.from_pretrained(GEMMA, use_fast=True)
    TOKS["gemma"]=lambda s: len(g.encode(s, add_special_tokens=False))
    if QWEN:
        q=AutoTokenizer.from_pretrained(QWEN, use_fast=True)
        TOKS["qwen"]=lambda s: len(q.encode(s, add_special_tokens=False))
load()
def counts(s): return {k:f(s) for k,f in TOKS.items()}

def pad_to(target_gemma):
    # prepend neutral filler word-by-word for ~1-token precision
    if TOKS["gemma"](BASE) >= target_gemma: return BASE
    words=(FILLER*8).split()
    pad=[]
    for w in words:
        if TOKS["gemma"](" ".join(pad+[w])+" "+BASE) > target_gemma: break
        pad.append(w)
    return (" ".join(pad)+" "+BASE).strip() if pad else BASE

out={"base":{"text":BASE,"tokens":counts(BASE)}, "variants":{}}
for v,m in VAR.items():
    dia=sub(BASE,m); fw=sub(BASE, fullwidth_map(m))
    tgt=TOKS["gemma"](dia)
    asc=pad_to(tgt)
    out["variants"][v]={
        "diacritic":{"text":dia,"tokens":counts(dia)},
        "fullwidth_twin":{"text":fw,"tokens":counts(fw)},
        "ascii_len_matched":{"text":asc,"tokens":counts(asc)},
    }
json.dump(out, open(OUT_JSON,"w"), ensure_ascii=False, indent=2)

# markdown summary (token counts only; prompts in json)
tk=list(TOKS.keys())
L=["# Causal-Gate Controls — token-count match quality\n",
   f"Base prompt: gemma={out['base']['tokens']['gemma']} tokens.\n",
   "For each variant: D=diacritic, FW=fullwidth non-diacritic twin (matched density/positions), "
   "A=token-count-matched ASCII (neutral padding). Columns are token counts per tokenizer.\n",
   "| Variant | cond | "+" | ".join(tk)+" |","|---|---|"+"|".join(["---"]*len(tk))+"|"]
for v in out["variants"]:
    for cond,key in [("D","diacritic"),("FW","fullwidth_twin"),("A","ascii_len_matched")]:
        c=out["variants"][v][key]["tokens"]
        L.append(f"| {v} | {cond} | "+" | ".join(str(c[t]) for t in tk)+" |")
open(OUT_MD,"w").write("\n".join(L))
print("WROTE", OUT_JSON, OUT_MD)
print("\nGemma token counts (D / FW / A) per variant:")
for v in out["variants"]:
    g=lambda key: out["variants"][v][key]["tokens"]["gemma"]
    print(f"  {v:16} D={g('diacritic'):3}  FW={g('fullwidth_twin'):3}  A={g('ascii_len_matched'):3}")
