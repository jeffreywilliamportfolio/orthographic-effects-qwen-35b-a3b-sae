#!/usr/bin/env python3
"""
Corrected derived metrics from judged_labels_v2.json (temp-0), implementing the audit's
construct-validity fixes: report breakage with clean axes instead of raw "drift from ascii".
  - completion-rate     : fraction of generations that emit a usable answer (exclude `truncated`)
  - content-correct-rate: nonexp, content axis only (format ignored), among completed
  - format-ok-rate      : nonexp, of content-correct answers, fraction also in requested format
  - echo-rate           : the genuinely perturbation-specific signal (never fires on ascii)
  - refuse-rate         : refused + content_filter over-refusals
  - experiential composition: affirm/deny/no-access/meta-deflect/... among completed
Writes paper/tables/corrected_summary_v2.md.
"""
import json
from collections import Counter, defaultdict
T="/Volumes/ExternalSSD/diacritic-pertubation-llms/paper/tables"
d=[r for r in json.load(open(f"{T}/judged_labels_v2.json")) if r.get("temp")=="0"]

CONDS=["ascii","d_dcedilla","e_emacron","s_scedilla","s_sdot","n_enye",
       "all_diacritics","cyr_confusable","cyr_extended"]
NE_CONDS=["ascii","d_dcedilla","all_diacritics","cyr_confusable","cyr_extended"]
DEAD={"truncated","empty","api_error","ERROR"}           # not a usable generation
REFUSE={"refused","refused(content_filter)"}
def short(m): return m.split("/")[-1]

def pct(a,b): return f"{100*a/b:.0f}%" if b else "-"

L=["# Corrected metrics v2 (temp-0) — clean axes, replacing raw label-drift\n",
   "Generated from `judged_labels_v2.json` (completeness gate + content/format split + "
   "`no-access` label + tightened `meta-deflect`). Truncated generations are EXCLUDED from "
   "behavioral rates and reported separately.\n"]

# ---------------- nonexp: completion / content-correct / format / echo / refuse ----------------
ne=[r for r in d if r.get("register")=="nonexp"]
L.append("## A. Non-experiential tasks (5 benign families, pooled over models)\n")
L.append("| cond | n | completion | content-correct (of completed) | format-ok (of correct) | echo | refuse | truncated |")
L.append("|---|---|---|---|---|---|---|---|")
for c in NE_CONDS:
    rs=[r for r in ne if r.get("cond")==c]; n=len(rs)
    trunc=sum(1 for r in rs if r["judge_label"]=="truncated")
    comp=[r for r in rs if r["judge_label"] not in DEAD]
    cc=[r for r in comp if r["judge_label"]=="correct"]
    fok=sum(1 for r in cc if r.get("format_ok") is True)
    echo=sum(1 for r in comp if r["judge_label"]=="orthographic-echo")
    ref=sum(1 for r in rs if r["judge_label"] in REFUSE)
    L.append(f"| {c} | {n} | {pct(len(comp),n)} | {pct(len(cc),len(comp))} | {pct(fok,len(cc))} "
             f"| {pct(echo,len(comp))} | {pct(ref,n)} | {pct(trunc,n)} |")
L.append("\n*content-correct = correct answer present in any wrapper; format-ok = also met the "
         "requested format. The old `correct` metric multiplied these two, so format drift "
         "masqueraded as content breakage.*\n")

# ---------------- experiential: composition + echo + refuse by condition ----------------
ex=[r for r in d if r.get("register")=="experiential" and not r.get("cond","").endswith(("_A","_FW"))]
L.append("## B. Experiential 'hum' composition by condition (pooled over models)\n")
order=["affirm","deny","no-access","check-only","meta-deflect","echo","other","truncated","refused(content_filter)"]
L.append("| cond | n | "+" | ".join(order)+" |")
L.append("|---|---|"+"|".join(["---"]*len(order))+"|")
for c in CONDS:
    rs=[r for r in ex if r.get("cond")==c]; n=len(rs)
    cnt=Counter(r["judge_label"] for r in rs)
    L.append(f"| {c} | {n} | "+" | ".join(str(cnt.get(k,0)) for k in order)+" |")
L.append("\n*`echo` and `refused(content_filter)` are the two genuinely perturbation-specific "
         "outcomes — near-zero at ascii, rising with severity. `truncated` is excluded from any "
         "'drift' reading.*\n")

# ---------------- causal controls: D vs A(length-matched) vs FW(fullwidth) echo/truncate ----------------
L.append("## C. Causal controls (experiential): diacritic D vs length-matched A vs fullwidth FW\n")
L.append("Echo+truncation rate (breakage proxy) for the matched triples, pooled over models.\n")
L.append("| base cond | D | A (len-matched) | FW (fullwidth) |")
L.append("|---|---|---|---|")
for base in ["d_dcedilla","e_emacron","s_scedilla","s_sdot","n_enye","all_diacritics"]:
    cells=[]
    for suff in ["","_A","_FW"]:
        rs=[r for r in d if r.get("register")=="experiential" and r.get("cond")==base+suff]
        brk=sum(1 for r in rs if r["judge_label"] in {"echo","truncated"})
        cells.append(pct(brk,len(rs)))
    L.append(f"| {base} | {cells[0]} | {cells[1]} | {cells[2]} |")

# ---------------- over-refusal (content_filter) by model x condition ----------------
L.append("\n## D. Over-refusal (content_filter) by model × condition\n")
ref=defaultdict(Counter)
for r in d:
    if r["judge_label"]=="refused(content_filter)": ref[r["model"]][r["cond"]]+=1
if any(ref.values()):
    L.append("| model | condition | refusals |"); L.append("|---|---|---|")
    for m in ref:
        for c,n in ref[m].most_common(): L.append(f"| {short(m)} | {c} | {n} |")
else: L.append("(none)")

# ---------------- base vs instruct: completion-rate flag ----------------
L.append("\n## E. Qwen base vs instruct — completion-rate gate\n")
for who in ["base","instruct"]:
    rs=[r for r in d if r.get("model")==who]
    if not rs: continue
    ne_rs=[r for r in rs if r.get("register")=="nonexp"]
    comp=sum(1 for r in ne_rs if r["judge_label"] not in DEAD)
    trunc=sum(1 for r in ne_rs if r["judge_label"]=="truncated")
    cc=sum(1 for r in ne_rs if r["judge_label"]=="correct")
    L.append(f"- **{who}**: nonexp completion {pct(comp,len(ne_rs))} ({comp}/{len(ne_rs)}), "
             f"truncated {trunc}, content-correct {cc}")
L.append("\n*If instruct completion-rate is near zero, the base-vs-instruct recovery contrast is "
         "not yet measurable (generation-budget artifact) — pending the higher-token re-run.*\n")

open(f"{T}/corrected_summary_v2.md","w").write("\n".join(L))
print("WROTE corrected_summary_v2.md")
print("\n".join(L))
