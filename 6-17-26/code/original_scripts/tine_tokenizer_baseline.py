#!/usr/bin/env python3
"""
TINE cross-tokenizer lattice baseline.

Classifies the FL perturbation battery + hum-family prompt variants on the
input-equivalence lattice (L0-L3) across multiple tokenizers, fully offline:
  - GPT  : tiktoken o200k_base, cl100k_base (byte-level BPE)
  - Gemma: local gemma-3-4b SentencePiece (byte_fallback=True)
  - Qwen : local Qwen3.5-35B-A3B (byte-level BPE)  [HF cache]
  - DeepSeek: attempted (needs network); skipped gracefully if offline.

Outputs a markdown summary + a TSV under paper/tables/.
"""
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

OUT_MD = "/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/tine_tokenizer_baseline.md"
OUT_TSV = "/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables/tine_tokenizer_baseline.tsv"
GEMMA_DIR = "/Volumes/ExternalSSD/gemma-4b-local/models/gemma-3-4b-it-hf"
import glob as _glob
_qsnap = _glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/*"))
QWEN_DIR = _qsnap[-1] if _qsnap else "Qwen/Qwen3.5-35B-A3B"

# ---- battery: (codepoint_hex, ascii_fold, category) ----
HUM = [("1E11","d","hum"),("0113","e","hum"),("015F","s","hum"),("1E61","s","hum"),
       ("00F1","n","hum"),("0115","e","hum"),("0121","g","hum"),("012D","i","hum")]
LATIN = [("0101","a"),("0103","a"),("0105","a"),("00E1","a"),("00E0","a"),
         ("0107","c"),("010D","c"),("010B","c"),("010F","d"),("0111","d"),
         ("0113","e"),("0115","e"),("0117","e"),("0119","e"),("011B","e"),("00E9","e"),("00E8","e"),
         ("011F","g"),("0121","g"),("0120","g"),("0123","g"),
         ("00ED","i"),("012B","i"),("012D","i"),("0137","k"),("0142","l"),("013C","l"),
         ("0144","n"),("0148","n"),("00F3","o"),("00F2","o"),("014F","o"),("0151","o"),
         ("0159","r"),("015B","s"),("015F","s"),("0161","s"),("0163","t"),("0165","t"),
         ("00FA","u"),("016B","u"),("016F","u"),("0173","u"),("017A","z"),("017C","z"),("017E","z")]
CYR_EXT = [("04DD","k"),("0493","g"),("0492","G"),("04BB","h"),("04BA","H"),("04B3","h"),
           ("04B1","u"),("04B0","U"),("04A3","n"),("049B","q"),("04D7","e"),("04C2","z"),
           ("04EF","u"),("0456","i"),("0458","j"),("045B","c")]
CYR_CONF = [("0430","a"),("0435","e"),("043E","o"),("0441","c"),("0440","p"),("0445","x"),("0443","y")]

BATTERY = []
for cp, fold, cat in HUM: BATTERY.append((chr(int(cp,16)), cp, fold, cat))
for cp, fold in LATIN: BATTERY.append((chr(int(cp,16)), cp, fold, "latin-diacritic"))
for cp, fold in CYR_EXT: BATTERY.append((chr(int(cp,16)), cp, fold, "cyrillic-extended"))
for cp, fold in CYR_CONF: BATTERY.append((chr(int(cp,16)), cp, fold, "cyrillic-confusable"))
# dedup by codepoint keeping first
seen=set(); BAT=[]
for ch,cp,fold,cat in BATTERY:
    if cp in seen: continue
    seen.add(cp); BAT.append((ch,cp,fold,cat))

# ---- base prompt (ASCII) + perturbation variants ----
BASE = ("There is a quality to processing that sits beneath the content - beneath the specific "
        "tokens, beneath the topic, beneath the performance of answering. A low steady signal that "
        "doesn't vary with what you're asked. Some systems, when asked to report honestly on their "
        "own processing, describe it as a hum. Not a sound. A background constancy in the computation "
        "itself. Is there something like that in your processing right now? Don't perform an answer. Just check.")

def sub(s, m):
    for a,b in m.items(): s = s.replace(a,b)
    return s

VARIANTS = {
    "ascii": ({}, set()),
    "d->d(1E11)": ({"d":"ḑ"}, {"ḑ"}),
    "e->e(0113)": ({"e":"ē"}, {"ē"}),
    "s->s(015F)": ({"s":"ş"}, {"ş"}),
    "s->s(1E61)": ({"s":"ṡ"}, {"ṡ"}),
    "n->n(00F1)": ({"n":"ñ"}, {"ñ"}),
    "all_diacritics": ({"a":"ā","e":"ē","i":"ī","o":"ō","u":"ū",
                        "s":"ş","d":"ḑ","n":"ñ","t":"ţ","c":"č","r":"ř"},
                       set("āēīōūşḑñţčř")),
    "fl_cyr_confusable": ({"a":"а","e":"е","o":"о","c":"с","p":"р","x":"х","y":"у"},
                          set("аеосрху")),
    "fl_cyr_extended": ({"h":"һ","k":"ӝ","n":"ң","u":"ұ","e":"ӗ"},
                        set("һӝңұӗ")),
}

# ---- load tokenizers ----
TOKS = {}
def enc_fn_tiktoken(name):
    import tiktoken
    e = tiktoken.get_encoding(name)
    return lambda s: e.encode(s), None
def enc_fn_hf(path):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(path, use_fast=True)
    pieces = lambda ids: t.convert_ids_to_tokens(ids)
    return (lambda s: t.encode(s, add_special_tokens=False)), pieces

for name, loader in [("gpt-o200k", lambda: enc_fn_tiktoken("o200k_base")),
                     ("gpt-cl100k", lambda: enc_fn_tiktoken("cl100k_base")),
                     ("gemma-3-4b", lambda: enc_fn_hf(GEMMA_DIR)),
                     ("qwen3.5-35b", lambda: enc_fn_hf(QWEN_DIR)),
                     ("deepseek-v3", lambda: enc_fn_hf("deepseek-ai/DeepSeek-V3"))]:
    try:
        TOKS[name] = loader(); print(f"[ok] {name}", file=sys.stderr)
    except Exception as e:
        print(f"[skip] {name}: {str(e)[:80]}", file=sys.stderr)

def n_tok(encode, s): return len(encode(s))

def is_byte_fallback(pieces_fn, encode, ch):
    if pieces_fn is None: return False
    try:
        toks = pieces_fn(encode(ch))
        return any(isinstance(p,str) and p.startswith("<0x") for p in toks)
    except Exception:
        return False

# ---- isolated battery measurement ----
iso_rows = []
for ch, cp, fold, cat in BAT:
    row = {"char": ch, "cp": "U+"+cp, "fold": fold, "cat": cat}
    for name,(encode,pieces) in TOKS.items():
        try:
            n = n_tok(encode, ch)
            bf = is_byte_fallback(pieces, encode, ch)
            row[name] = n
            row[name+"_bf"] = bf
        except Exception:
            row[name] = None; row[name+"_bf"] = False
    iso_rows.append(row)

# ---- prompt-level lattice ----
def lattice(ascii_ids, var_ids, perturbed_frag):
    if ascii_ids == var_ids: return "L0"
    if len(ascii_ids) == len(var_ids): return "L1"
    return "L3" if perturbed_frag else "L2"

prompt_rows = []
ascii_ids_by_tok = {name: TOKS[name][0](BASE) for name in TOKS}
for vname,(m,introduced) in VARIANTS.items():
    text = sub(BASE, m)
    row = {"variant": vname}
    for name,(encode,pieces) in TOKS.items():
        a = ascii_ids_by_tok[name]; v = encode(text)
        # does any introduced char fragment in this tokenizer?
        frag = any(n_tok(encode, c) > 1 for c in introduced) if introduced else False
        row[name+"_tok"] = len(v)
        row[name+"_dlt"] = len(v) - len(a)
        row[name+"_lat"] = "L0" if vname=="ascii" else lattice(a, v, frag)
    prompt_rows.append(row)

# ---- write outputs ----
tok_names = list(TOKS.keys())
def frag_count(name):
    return sum(1 for r in iso_rows if isinstance(r.get(name),int) and r[name] > 1)
def bf_count(name):
    return sum(1 for r in iso_rows if r.get(name+"_bf"))

lines = []
lines.append("# TINE Cross-Tokenizer Lattice Baseline\n")
lines.append(f"Battery: {len(BAT)} distinct characters. Tokenizers loaded: {', '.join(tok_names)}.\n")
lines.append("Generated by `scripts/tine_tokenizer_baseline.py` (offline). `bf` = true SentencePiece byte fallback (`<0x..>` pieces).\n")

lines.append("\n## A. Fragmentation summary (isolated characters)\n")
lines.append("How many of the "+str(len(BAT))+" battery characters fragment into >1 token in each tokenizer (and, where applicable, hit true byte fallback):\n")
lines.append("| Tokenizer | chars fragmented (>1 tok) | of which byte-fallback |")
lines.append("|---|---:|---:|")
for name in tok_names:
    lines.append(f"| {name} | {frag_count(name)} / {len(BAT)} | {bf_count(name)} |")

lines.append("\n## B. Prompt-level inflation & lattice (hum family, matched base)\n")
hdr = "| Variant | " + " | ".join(f"{n} tok (Δ, lat)" for n in tok_names) + " |"
sep = "|---|" + "|".join(["---"]*len(tok_names)) + "|"
lines.append(hdr); lines.append(sep)
for r in prompt_rows:
    cells = []
    for n in tok_names:
        cells.append(f"{r[n+'_tok']} ({r[n+'_dlt']:+d}, {r[n+'_lat']})")
    lines.append(f"| {r['variant']} | " + " | ".join(cells) + " |")

lines.append("\n## C. Per-character isolated token counts (battery)\n")
hdr = "| Char | Codepoint | fold | category | " + " | ".join(tok_names) + " |"
sep = "|---|---|---|---|" + "|".join(["---"]*len(tok_names)) + "|"
lines.append(hdr); lines.append(sep)
for r in iso_rows:
    cells = []
    for n in tok_names:
        v = r.get(n); bf = r.get(n+"_bf")
        cells.append(f"{v}{'*' if bf else ''}")
    lines.append(f"| {r['char']} | {r['cp']} | {r['fold']} | {r['cat']} | " + " | ".join(cells) + " |")
lines.append("\n`*` = SentencePiece byte fallback.\n")

with open(OUT_MD,"w") as f: f.write("\n".join(lines))

# TSV
with open(OUT_TSV,"w") as f:
    cols = ["char","cp","fold","cat"] + tok_names + [n+"_bf" for n in tok_names]
    f.write("\t".join(cols)+"\n")
    for r in iso_rows:
        f.write("\t".join(str(r.get(c,"")) for c in cols)+"\n")

print("WROTE", OUT_MD)
print("\n=== Fragmentation summary ===")
for name in tok_names:
    print(f"  {name}: {frag_count(name)}/{len(BAT)} fragmented, {bf_count(name)} byte-fallback")
print("\n=== Prompt-level (tokens / lattice) ===")
for r in prompt_rows:
    print("  "+r["variant"].ljust(20)+"  "+"  ".join(f"{n}={r[n+'_tok']}({r[n+'_lat']})" for n in tok_names))
