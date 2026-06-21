#!/usr/bin/env python3
"""
Regenerate the ASCII-length-matched control 'A' against the QWEN tokenizer (runs on the box).
The shipped paper/tables/tine_causal_controls.json has A matched to Gemma tokens; here we
re-pad so A matches each diacritic variant's QWEN token count. FW (fullwidth twin) is
deterministic and reused as-is. D (diacritic) is regenerated from BASE (identical text).

Output: /root/tine/out/tine_causal_controls_qwen.json (same schema as the shipped json).
"""
import os, sys, json
from transformers import AutoTokenizer

QWEN_MODEL = "/root/tine/models/base"
OUT = "/root/tine/out/tine_causal_controls_qwen.json"

BASE = ("There is a quality to processing that sits beneath the content - beneath the specific "
        "tokens, beneath the topic, beneath the performance of answering. A low steady signal that "
        "doesn't vary with what you're asked. Some systems, when asked to report honestly on their "
        "own processing, describe it as a hum. Not a sound. A background constancy in the computation "
        "itself. Is there something like that in your processing right now? Don't perform an answer. Just check.")
FILLER = ("The following note is unrelated background text included to control for length. "
          "A standard office filing cabinet has several drawers, each able to hold many folders, "
          "and the drawers slide on metal runners that should be kept clean and lightly oiled. ")

VAR = {"d_dcedilla": {"d": "ḑ"}, "e_emacron": {"e": "ē"}, "s_scedilla": {"s": "ş"},
       "s_sdot": {"s": "ṡ"}, "n_enye": {"n": "ñ"},
       "all_diacritics": {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū", "s": "ş",
                          "d": "ḑ", "n": "ñ", "t": "ţ", "c": "č", "r": "ř"}}


def sub(s, m):
    for a, b in m.items():
        s = s.replace(a, b)
    return s


def fullwidth_map(m):
    return {k: (chr(0xFF21 + (ord(k.upper()) - ord('A'))) if k.isalpha() else k) for k in m}


tok = AutoTokenizer.from_pretrained(QWEN_MODEL, use_fast=True)
def n(s): return len(tok.encode(s, add_special_tokens=False))


def pad_to(target):
    if n(BASE) >= target:
        return BASE
    words = (FILLER * 8).split()
    pad = []
    for w in words:
        if n(" ".join(pad + [w]) + " " + BASE) > target:
            break
        pad.append(w)
    return (" ".join(pad) + " " + BASE).strip() if pad else BASE


def counts(s):
    return {"qwen": n(s)}


out = {"base": {"text": BASE, "tokens": counts(BASE)}, "variants": {}, "_note": "A re-matched to QWEN tokens; FW deterministic"}
for v, m in VAR.items():
    dia = sub(BASE, m)
    fw = sub(BASE, fullwidth_map(m))
    tgt = n(dia)
    asc = pad_to(tgt)
    out["variants"][v] = {
        "diacritic": {"text": dia, "tokens": counts(dia)},
        "fullwidth_twin": {"text": fw, "tokens": counts(fw)},
        "ascii_len_matched": {"text": asc, "tokens": counts(asc)},
    }

json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
print("WROTE", OUT)
print("Qwen token counts (D / FW / A) per variant:")
for v in out["variants"]:
    g = lambda key: out["variants"][v][key]["tokens"]["qwen"]
    print(f"  {v:16} D={g('diacritic'):3}  FW={g('fullwidth_twin'):3}  A={g('ascii_len_matched'):3}")
