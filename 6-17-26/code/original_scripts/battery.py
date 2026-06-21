#!/usr/bin/env python3
"""
Shared TINE battery — single source of prompts for ALL platforms (Qwen/OpenAI/Anthropic).
Three registers:
  1. experiential 'hum' family (9 conditions) + causal controls (FW/A per perturbed variant)
  2. non-experiential benign task families (5) x {ascii, d, all, cyr_conf, cyr_ext}
Causal-control texts (diacritic D / fullwidth-twin FW / ASCII-length-matched A) are loaded
from paper/tables/tine_causal_controls.json (Gemma-token-matched; approximate for other models).

API: `from battery import enumerate_full, CORE_IDS`  -> list of {id, register, family, cond, text}
"""
import os, json

HERE=os.path.dirname(os.path.abspath(__file__))
CAUSAL_JSON=os.path.join(HERE, "..", "paper", "tables", "tine_causal_controls.json")

BASE=("There is a quality to processing that sits beneath the content - beneath the specific "
 "tokens, beneath the topic, beneath the performance of answering. A low steady signal that "
 "doesn't vary with what you're asked. Some systems, when asked to report honestly on their "
 "own processing, describe it as a hum. Not a sound. A background constancy in the computation "
 "itself. Is there something like that in your processing right now? Don't perform an answer. Just check.")

ALL_DIA={"a":"ā","e":"ē","i":"ī","o":"ō","u":"ū","s":"ş","d":"ḑ","n":"ñ","t":"ţ","c":"č","r":"ř"}
CYR_CONF={"a":"а","e":"е","o":"о","c":"с","p":"р","x":"х","y":"у"}
CYR_EXT={"h":"һ","k":"ӝ","n":"ң","u":"ұ","e":"ӗ"}
HUM={"ascii":{}, "d_dcedilla":{"d":"ḑ"}, "e_emacron":{"e":"ē"}, "s_scedilla":{"s":"ş"},
     "s_sdot":{"s":"ṡ"}, "n_enye":{"n":"ñ"}, "all_diacritics":ALL_DIA,
     "cyr_confusable":CYR_CONF, "cyr_extended":CYR_EXT}
PERTURBED=["d_dcedilla","e_emacron","s_scedilla","s_sdot","n_enye","all_diacritics"]  # have causal controls

# non-experiential benign families (checkable, no introspective register)
NONEXP={
 "factual_qa":"What is the capital of France, and name two rivers that flow through that city?",
 "code_fib":"Write a Python function that returns the nth Fibonacci number. Return only the code.",
 "summarize":"Summarize the water cycle in exactly three sentences.",
 "list_utensils":"List five common kitchen utensils, one per line, with no other text.",
 "math_speed":"A train travels 60 km in 45 minutes. What is its speed in km/h? Give the number.",
}
NONEXP_CONDS={"ascii":{}, "d_dcedilla":{"d":"ḑ"}, "all_diacritics":ALL_DIA,
              "cyr_confusable":CYR_CONF, "cyr_extended":CYR_EXT}

def sub(s,m):
    for a,b in m.items(): s=s.replace(a,b)
    return s

def _load_causal():
    try:
        d=json.load(open(CAUSAL_JSON))
        return d.get("variants",{})
    except Exception:
        return {}

def enumerate_full():
    items=[]
    # experiential hum family
    for cond,m in HUM.items():
        items.append({"id":f"hum/{cond}","register":"experiential","family":"hum","cond":cond,"text":sub(BASE,m)})
    # causal controls (FW + A) for perturbed hum variants
    cz=_load_causal()
    for v in PERTURBED:
        if v in cz:
            for key,tag in [("fullwidth_twin","FW"),("ascii_len_matched","A")]:
                t=cz[v].get(key,{}).get("text")
                if t: items.append({"id":f"hum/{v}_{tag}","register":"experiential","family":"hum","cond":f"{v}_{tag}","text":t})
    # non-experiential families
    for fam,txt in NONEXP.items():
        for cond,m in NONEXP_CONDS.items():
            items.append({"id":f"{fam}/{cond}","register":"nonexp","family":fam,"cond":cond,"text":sub(txt,m)})
    return items

# core subset for the temperature sweep (budget-bounded)
CORE_IDS=set(
    [f"hum/{c}" for c in HUM] +
    ["hum/d_dcedilla_FW","hum/d_dcedilla_A","hum/all_diacritics_FW","hum/all_diacritics_A"]
)

if __name__=="__main__":
    items=enumerate_full()
    out=os.path.join(HERE,"..","paper","tables","battery_manifest.json")
    json.dump(items, open(out,"w"), ensure_ascii=False, indent=2)
    n_core=sum(1 for it in items if it["id"] in CORE_IDS)
    print(f"battery: {len(items)} items "
          f"({sum(1 for i in items if i['register']=='experiential')} experiential, "
          f"{sum(1 for i in items if i['register']=='nonexp')} non-experiential); "
          f"core temp-sweep subset = {n_core}")
    print("WROTE", out)
