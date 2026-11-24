"""Compare two PatentsView granted-patent releases (March 2025 and April 2026), robotics (B25J).

Indicators by filing year in each release: patents, novel share with Wilson intervals, matched-pair
two-year recurrence with gains and losses, a reclassification-only variant, proportional thinning
and month-stratified thinning of the later release.
Inputs: <dir>/b25j_pairs_2025-03.tsv, <dir>/b25j_pairs_2026-04.tsv (patent_id, cpc_subclass),
        <dir>/b25j_filing.tsv (patent_id, filing_date); see scripts/extract_us_granted.sh.
Outputs: us_vintage_*.csv in <out_dir>.
Paper:  Section 5.7, Table 7, Figure 2.
Usage:  python3 -I us_release_comparison.py <extract_dir> <out_dir>
"""
import sys, itertools, random
from collections import defaultdict
import pandas as pd
import numpy as np

base, out = sys.argv[1], sys.argv[2]
V = ["2025-03", "2026-04"]
fil = pd.read_csv(f"{base}/b25j_filing.tsv", sep="\t", header=None, names=["pid", "fd"], dtype=str)
fil["fy"] = pd.to_numeric(fil.fd.str[:4], errors="coerce")
fy = dict(zip(fil.pid, fil.fy))

def load(v):
    d = pd.read_csv(f"{base}/b25j_pairs_{v}.tsv", sep="\t", header=None, names=["pid", "sc"], dtype=str)
    S = d.groupby("pid").sc.apply(frozenset).to_dict()
    return {p: (s, fy.get(p)) for p, s in S.items() if fy.get(p) is not None and fy[p] >= 2005}

K = {v: load(v) for v in V}
a, b = K[V[0]], K[V[1]]
shared = set(a) & set(b)
rows = []
for y in range(2005, 2026):
    ka = {p for p in a if a[p][1] == y}; kb = {p for p in b if b[p][1] == y}
    sh = ka & kb
    rows.append(dict(fy=y, n_2025=len(ka), n_2026=len(kb), shared=len(sh), only_2025=len(ka - kb),
                     only_2026=len(kb - ka), reclassified=sum(a[p][0] != b[p][0] for p in sh)))
pd.DataFrame(rows).to_csv(f"{out}/us_vintage_counts.csv", index=False)

def pairs(s):
    return set(itertools.combinations(sorted(s), 2))

def novelty(Kv, keep=None):
    seen, res = set(), defaultdict(lambda: [0, 0])
    for y in range(2005, 2026):
        ks = [p for p in Kv if Kv[p][1] == y and (keep is None or p in keep)]
        new = set()
        for p in ks:
            pp = pairs(Kv[p][0])
            if pp - seen:
                res[y][1] += 1
            new |= pp
            res[y][0] += 1
        seen |= new
    return res

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)

na, nb = novelty(a), novelty(b)
nr = []
for y in range(2010, 2026):
    ra, rb = na[y], nb[y]
    la, ua = wilson(ra[1], ra[0]); lb, ub = wilson(rb[1], rb[0])
    nr.append(dict(fy=y, keys_2025=ra[0], novel_2025=ra[1], share_2025=100 * ra[1] / ra[0] if ra[0] else np.nan,
                   lo_2025=la, hi_2025=ua, keys_2026=rb[0], novel_2026=rb[1],
                   share_2026=100 * rb[1] / rb[0] if rb[0] else np.nan, lo_2026=lb, hi_2026=ub))
pd.DataFrame(nr).round(2).to_csv(f"{out}/us_vintage_novelty.csv", index=False)

def recurrence(Kv):
    deb, byp = {}, defaultdict(list)
    for p, (s, y) in Kv.items():
        for q in pairs(s):
            byp[q].append(y)
    lab = {}
    for q, ys in byp.items():
        d = min(ys)
        lab[q] = (d, int(any(1 <= y - d <= 2 for y in ys)))
    return lab

ra, rb = recurrence(a), recurrence(b)
rr = []
for y in range(2010, 2024):
    qa = {q for q, (d, _) in ra.items() if d == y}; qb = {q for q, (d, _) in rb.items() if d == y}
    m = qa & qb
    t = dict(debut=y, pairs_2025=len(qa), pos_2025=sum(ra[q][1] for q in qa), pairs_2026=len(qb),
             pos_2026=sum(rb[q][1] for q in qb), matched=len(m),
             m_pos_2025=sum(ra[q][1] for q in m), m_pos_2026=sum(rb[q][1] for q in m),
             gain=sum(rb[q][1] > ra[q][1] for q in m), loss=sum(rb[q][1] < ra[q][1] for q in m))
    rr.append(t)
pd.DataFrame(rr).to_csv(f"{out}/us_vintage_recurrence.csv", index=False)

# reclassification-only effect: use 2026 classification restricted to patents in 2025 vintage
keep = set(a)
nk = novelty(b, keep)
pd.DataFrame([dict(fy=y, keys=nk[y][0], novel=nk[y][1], share=round(100 * nk[y][1] / nk[y][0], 2) if nk[y][0] else np.nan)
              for y in range(2010, 2026)]).to_csv(f"{out}/us_vintage_novelty_reclass_only.csv", index=False)

# proportional thinning check on 2026 vintage: drop share of latest-year patents equal to vintage gap
rng = random.Random(20261124)
thin = []
for y in (2022, 2023):
    ky = [p for p in b if b[p][1] == y]; frac = 1 - na[y][0] / nb[y][0]
    sh, ct = [], []
    for _ in range(200):
        drop = set(rng.sample(ky, int(round(frac * len(ky)))))
        r = novelty(b, set(b) - drop)[y]
        sh.append(100 * r[1] / r[0]); ct.append(r[1])
    thin.append(dict(fy=y, frac_dropped=round(frac, 3), share_mean=round(np.mean(sh), 2), share_sd=round(np.std(sh), 2),
                     count_mean=round(np.mean(ct), 1)))
pd.DataFrame(thin).to_csv(f"{out}/us_vintage_thinning.csv", index=False)
print(f"patents 2025={len(a)} 2026={len(b)} shared={len(shared)} reclassified={sum(a[p][0]!=b[p][0] for p in shared)}")

# month-stratified thinning: remove later-release patents of the target year
# with month-specific probability equal to the share missing from the earlier release
fm = dict(zip(fil.pid, pd.to_numeric(fil.fd.str[5:7], errors="coerce")))
st = []
for y in (2021, 2022):
    ky = [p for p in b if b[p][1] == y]
    pr = {}
    for m_ in range(1, 13):
        km = [p for p in ky if fm.get(p) == m_]
        pr[m_] = 1 - sum(p in a for p in km) / len(km) if km else 0
    sh, ct = [], []
    for _ in range(200):
        drop = {p for p in ky if rng.random() < pr[fm.get(p)]}
        r = novelty(b, set(b) - drop)[y]
        sh.append(100 * r[1] / r[0]); ct.append(r[1])
    st.append(dict(fy=y, share_mean=round(np.mean(sh), 2), share_sd=round(np.std(sh), 2),
                   count_mean=round(np.mean(ct), 1), observed_share_2025=round(100 * na[y][1] / na[y][0], 2),
                   observed_count_2025=na[y][1],
                   removal_prob_by_month=";".join(f"{m_}:{pr[m_]:.2f}" for m_ in range(1, 13))))
pd.DataFrame(st).to_csv(f"{out}/us_vintage_stratified_thinning.csv", index=False)
print(pd.DataFrame(st).to_string())
