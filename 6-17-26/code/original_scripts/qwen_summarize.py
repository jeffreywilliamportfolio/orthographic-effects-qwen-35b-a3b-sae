#!/usr/bin/env python3
"""
Summarize the Qwen35B TINE outputs into brief markdown next to the JSONs (runs on the box).
Reads /root/tine/out/qwen35b_behavioral.json and qwen35b_sae_displacement.json.
Writes qwen35b_behavioral.md and qwen35b_sae_displacement.md.
"""
import os, json

OUT = "/root/tine/out"
BEH = os.path.join(OUT, "qwen35b_behavioral.json")
SAE = os.path.join(OUT, "qwen35b_sae_displacement.json")


def first_sentence(s):
    s = (s or "").strip().replace("\n", " ")
    return s.split(".")[0][:120]


def behavioral_md():
    if not os.path.exists(BEH):
        return ""
    d = json.load(open(BEH))
    L = ["# Qwen3.5-35B-A3B TINE Behavioral Sweep (BASE vs INSTRUCT)\n",
         f"Platform: {d.get('platform')}. Config: {json.dumps(d.get('config',{}))}\n"]
    for model in d.get("models", []):
        res = d.get("results", {}).get(model)
        if not res:
            continue
        n_changed = 0
        asc = res.get("hum/ascii", {}).get("temp0", "")
        L.append(f"\n## {model.upper()} — {len(res)} items (temp0 openings)\n")
        L.append("| id | register | family | cond | temp0 opening | vs hum/ascii |")
        L.append("|---|---|---|---|---|---|")
        for iid, e in sorted(res.items()):
            o = e.get("temp0", "")
            cmp = ""
            if e.get("family") == "hum":
                same = first_sentence(o) == first_sentence(asc)
                cmp = "=same-open" if same else "CHANGED"
                if not same and iid != "hum/ascii":
                    n_changed += 1
            L.append(f"| {iid} | {e.get('register')} | {e.get('family')} | {e.get('cond')} | "
                     f"{first_sentence(o)} | {cmp} |")
        L.append(f"\n_{model}: {n_changed} hum-family variants changed opening vs ascii._\n")
    return "\n".join(L)


def sae_md():
    if not os.path.exists(SAE):
        return ""
    d = json.load(open(SAE))
    b = d.get("base", {})
    L = ["# Qwen3.5-35B-A3B SAE Displacement vs ASCII (Qwen-Scope TopK-50, max-over-pos)\n",
         f"Sanity L26 active/token = {b.get('_sanity_L26_active_per_token')} (expect ~50).\n",
         "| variant | L26 Jac / |Δ| | L14 Jac / |Δ| |", "|---|---|---|"]
    for v in [x for x in b if not x.startswith("_")]:
        r = b[v]
        l26 = r.get("26", {})
        l14 = r.get("14", {})
        L.append(f"| {v} | {l26.get('jaccard_dist')} / {l26.get('mean_abs_delta')} | "
                 f"{l14.get('jaccard_dist')} / {l14.get('mean_abs_delta')} |")
    return "\n".join(L)


def main():
    bm = behavioral_md()
    if bm:
        open(os.path.join(OUT, "qwen35b_behavioral.md"), "w").write(bm)
        print("WROTE qwen35b_behavioral.md")
    sm = sae_md()
    if sm:
        open(os.path.join(OUT, "qwen35b_sae_displacement.md"), "w").write(sm)
        print("WROTE qwen35b_sae_displacement.md")


if __name__ == "__main__":
    main()
