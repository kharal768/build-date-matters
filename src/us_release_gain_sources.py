"""Classify label changes between the two PatentsView releases by source.

For each matched pair whose two-year label changes:
  gain_added   - every qualifying follow-on is a patent absent from the earlier release
  gain_reclass - every qualifying follow-on is a shared patent that gained the pair by reclassification
  gain_mixed   - both kinds
  loss         - label changes from 1 to 0
Input:  extract folder as for us_release_comparison.py.
Output: <out_csv>.
Paper:  Section 5.7.
Usage:  python3 -I us_release_gain_sources.py <extract_dir> <out_csv>
"""
import sys, itertools
import pandas as pd

base, outp = sys.argv[1], sys.argv[2]
fil = pd.read_csv(f"{base}/b25j_filing.tsv", sep="\t", header=None, names=["pid", "fd"], dtype=str)
fy = dict(zip(fil.pid, pd.to_numeric(fil.fd.str[:4], errors="coerce")))

def load(v):
    d = pd.read_csv(f"{base}/b25j_pairs_{v}.tsv", sep="\t", header=None, names=["pid", "sc"], dtype=str)
    S = d.groupby("pid").sc.apply(frozenset).to_dict()
    return {p: (s, fy[p]) for p, s in S.items() if pd.notna(fy.get(p)) and fy[p] >= 2005}

def ledger(K):
    by = {}
    for p, (s, y) in K.items():
        for q in itertools.combinations(sorted(s), 2):
            by.setdefault(q, []).append((p, y))
    out = {}
    for q, rr in by.items():
        d = min(y for _, y in rr)
        out[q] = (d, [p for p, y in rr if 1 <= y - d <= 2])
    return out

a, b = load("2025-03"), load("2026-04")
La, Lb = ledger(a), ledger(b)
rows = []
for c in range(2010, 2024):
    m = [q for q in La if La[q][0] == c and q in Lb and Lb[q][0] == c]
    r = dict(debut=c, matched=len(m), gain_added=0, gain_reclass=0, gain_mixed=0, loss=0)
    for q in m:
        ya, yb = bool(La[q][1]), bool(Lb[q][1])
        if yb and not ya:
            F = Lb[q][1]
            added = [p for p in F if p not in a]
            shared = [p for p in F if p in a]
            r["gain_added" if not shared else "gain_reclass" if not added else "gain_mixed"] += 1
        elif ya and not yb:
            r["loss"] += 1
    rows.append(r)
out = pd.DataFrame(rows); out.to_csv(outp, index=False); print(out.to_string())
