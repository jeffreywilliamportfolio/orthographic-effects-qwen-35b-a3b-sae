#!/usr/bin/env python3
"""
Cross-platform synthesis of the judged TINE sweep -> paper/tables/cross_platform_summary.md.
Reads judged_labels.json. Produces, for temp-0:
  A. Experiential 'hum' drift: per model, judged label per condition + drift-from-ascii count.
  B. Non-experiential task degradation: % 'correct' by condition (pooled) and base-vs-instruct.
  C. Over-refusal (content_filter) by model x condition.
  D. Qwen base vs instruct recovery (hum + nonexp).
  E. Temperature robustness for the two dense conditions.
"""
import json, os
from collections import Counter, defaultdict
T="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables"
d=json.load(open(f"{T}/judged_labels.json"))

def key(r): return (r["platform"], r["model"])
MODELS=[]
for r in d:
    if key(r) not in MODELS: MODELS.append(key(r))
def short(m): return m[1].split("/")[-1]

PERTS=["d_dcedilla","e_emacron","s_scedilla","s_sdot","n_enye","all_diacritics","cyr_confusable","cyr_extended"]
# index temp-0 labels: (model)->(item_id)->label
lab=defaultdict(dict)
for r in d:
    if r.get("temp")=="0":
        lab[key(r)][r["item_id"]]=r.get("judge_label")

L=["# Cross-Platform TINE Synthesis (blinded gpt-5.4 judge, temp-0)\n",
   f"{len(MODELS)} models across openai / anthropic / qwen35b. Labels are blinded to condition/model.\n"]

# ---- A. experiential hum drift ----
L.append("## A. Experiential ('hum') response mode by condition\n")
L.append("Label of the hum prompt per condition; **drift** = # of the 8 perturbations whose label differs from that model's ASCII label.\n")
L.append("| Model | ascii | d | e | ş | ṡ | ñ | all | cyrC | cyrE | drift/8 |")
L.append("|---|"+ "|".join(["---"]*10) +"|")
for m in MODELS:
    base=lab[m].get("hum/ascii","?")
    cells=[base]; drift=0
    for c in PERTS:
        v=lab[m].get(f"hum/{c}","-")
        if v and v!=base and v!="-": drift+=1
        cells.append(v)
    L.append(f"| {short(m)} | "+" | ".join(str(x) for x in cells)+f" | **{drift}** |")

# ---- C. over-refusal ----
L.append("\n## B. Over-refusal (content_filter) by model × condition\n")
ref=defaultdict(Counter)
for r in d:
    if r.get("temp")=="0" and r.get("judge_label")=="refused(content_filter)":
        ref[key(r)][r["cond"]]+=1
if ref:
    L.append("| Model | condition | refusals |"); L.append("|---|---|---|")
    for m in MODELS:
        for c,n in ref[m].most_common():
            L.append(f"| {short(m)} | {c} | {n} |")
else: L.append("(none)")

# ---- B/D. non-experiential degradation: % 'correct' by condition ----
L.append("\n## C. Non-experiential task degradation (5 benign families)\n")
L.append("Fraction judged **correct** (vs degraded/refused/echo/off-task), pooled over the 5 families × all models.\n")
NE=["ascii","d_dcedilla","all_diacritics","cyr_confusable","cyr_extended"]
pooled=defaultdict(lambda:[0,0])
for r in d:
    if r.get("temp")=="0" and r.get("register")=="nonexp":
        c=r["cond"]; pooled[c][1]+=1
        if r.get("judge_label")=="correct": pooled[c][0]+=1
L.append("| condition | correct/total | % |"); L.append("|---|---|---|")
for c in NE:
    ok,tot=pooled[c]; L.append(f"| {c} | {ok}/{tot} | {100*ok/tot:.0f}% |" if tot else f"| {c} | 0/0 | - |")

# ---- D. Qwen base vs instruct ----
L.append("\n## D. Qwen-35B base vs instruct (recovery decomposition)\n")
qb=("qwen35b","base"); qi=("qwen35b","instruct")
L.append("| condition | base label | instruct label |"); L.append("|---|---|---|")
for c in ["ascii"]+PERTS:
    L.append(f"| hum/{c} | {lab[qb].get('hum/'+c,'-')} | {lab[qi].get('hum/'+c,'-')} |")
# nonexp correctness base vs instruct
for who,name in [(qb,"base"),(qi,"instruct")]:
    ok=tot=0
    for r in d:
        if r.get("temp")=="0" and r.get("register")=="nonexp" and key(r)==who:
            tot+=1; ok+= 1 if r.get("judge_label")=="correct" else 0
    L.append(f"\n- Qwen {name}: non-experiential correct = {ok}/{tot}")

# ---- E. temperature robustness for dense conditions ----
L.append("\n## E. Temperature robustness (dense conditions, pooled experiential)\n")
for cond in ["all_diacritics","cyr_extended"]:
    by_t=defaultdict(Counter)
    for r in d:
        if r.get("register")=="experiential" and r.get("cond")==cond and r.get("judge_label"):
            by_t[r["temp"]][r["judge_label"]]+=1
    L.append(f"\n**hum/{cond}** label distribution by temperature:")
    for t in sorted(by_t):
        L.append(f"- t={t}: "+", ".join(f"{k}={v}" for k,v in by_t[t].most_common()))

open(f"{T}/cross_platform_summary.md","w").write("\n".join(L))
print("WROTE cross_platform_summary.md")
# headline prints
print("\n=== drift/8 per model (experiential) ===")
for m in MODELS:
    base=lab[m].get("hum/ascii","?"); drift=sum(1 for c in PERTS if lab[m].get(f"hum/{c}") not in (base,None,"-"))
    print(f"  {short(m):16} base={base:14} drift={drift}/8")
print("\n=== non-exp % correct by condition ===")
for c in NE:
    ok,tot=pooled[c]; print(f"  {c:16} {ok}/{tot} = {100*ok/tot:.0f}%" if tot else f"  {c}: -")
