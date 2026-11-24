"""Cluster bootstrap of the 2022 rate difference and month-stratified thinning of the novelty share.

a) Cluster bootstrap (2,000 draws) of the matched 2022 rate difference: each changed pair is
   assigned to the cluster of its first later-build follow-on key; unchanged pairs are singletons.
b) Removal of 2024 keys from the later build with month-specific probability equal to the share of
   later-build keys missing from the earlier build; novel count and share are recomputed.
Inputs: earlier build (.dta), later build (.xlsx).
Outputs: <prefix>_cluster_bootstrap.csv, <prefix>_stratified_thinning.csv.
Paper:  Section 5.1 (95% intervals), Section 5.5 (stratified thinning).
Usage:  python3 -I cluster_bootstrap_and_thinning.py <earlier_dta> <later_xlsx> <out_prefix>
"""
import os, sys
from itertools import combinations
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recurrence import load, keys  # noqa
from cohort_trend_and_validation import ledger  # noqa

rng = np.random.default_rng(20261124)
B = 2000


def main(orig, refr, pref):
    do, dr = load(orig), load(refr)
    out = []
    for rule in (False, True):
        nm = "restricted" if rule else "all_links"
        Lo, Lr = ledger(keys(do, rule)), ledger(keys(dr, rule))
        m = [p for p in Lo if Lo[p][0] == 2022 and p in Lr and Lr[p][0] == 2022]
        yo = {p: int(bool(Lo[p][1])) for p in m}; yr = {p: int(bool(Lr[p][1])) for p in m}
        oldkeys = {k for F in (Lo[p][1] for p in m) for k, _ in F}
        cl = {}
        for p in m:
            if yr[p] != yo[p]:
                ks = sorted(k for k, _ in Lr[p][1] if k not in oldkeys)
                cl.setdefault(ks[0], []).append(p)
            else:
                cl[("single", p)] = [p]
        C = list(cl.values())
        diffs = []
        for _ in range(B):
            s = [C[i] for i in rng.integers(0, len(C), len(C))]
            ps = [p for c in s for p in c]
            diffs.append(100 * (np.mean([yr[p] for p in ps]) - np.mean([yo[p] for p in ps])))
        est = 100 * (np.mean(list(yr.values())) - np.mean(list(yo.values())))
        changed_clusters = sum(1 for k in cl if not (isinstance(k, tuple) and k[0] == "single"))
        out.append(dict(rule=nm, n=len(m), est=round(est, 1), lo=round(np.percentile(diffs, 2.5), 1),
                        hi=round(np.percentile(diffs, 97.5), 1), supporting_clusters=changed_clusters,
                        cluster_sizes=";".join(str(len(v)) for k, v in cl.items() if not (isinstance(k, tuple) and k[0] == "single"))))
    pd.DataFrame(out).to_csv(f"{pref}_cluster_bootstrap.csv", index=False)
    print(pd.DataFrame(out).to_string())

    # b) month-stratified thinning
    def kd(d):
        g = d.groupby("key")["apdt"].first()
        return pd.to_datetime(g)
    Ko, Kr = keys(do, False), keys(dr, False)
    mo = kd(do).dt.month; mr = kd(dr).dt.month
    k24r = [k for k, v in Kr.items() if v[2] == 2024]
    k24o = {k for k, v in Ko.items() if v[2] == 2024}
    prob = {}
    for mth in range(1, 13):
        kr_m = [k for k in k24r if mr[k] == mth]
        prob[mth] = 1 - sum(k in k24o for k in kr_m) / len(kr_m) if kr_m else 0
    items = sorted(Kr.items(), key=lambda x: x[1][2])
    shares, counts = [], []
    for _ in range(200):
        keep = [(k, v) for k, v in items if not (v[2] == 2024 and rng.random() < prob[mr[k]])]
        first = {}
        for k, (I, S, y) in keep:
            for p in combinations(sorted(I), 2):
                first.setdefault(p, y)
        ks = [I for k, (I, S, y) in keep if y == 2024]
        new = sum(any(first[p] == 2024 for p in combinations(sorted(I), 2)) for I in ks)
        shares.append(100 * new / len(ks)); counts.append(new)
    res = dict(share_mean=round(np.mean(shares), 1), share_sd=round(np.std(shares), 1),
               count_mean=round(np.mean(counts), 1), count_p2_5=np.percentile(counts, 2.5),
               count_p97_5=np.percentile(counts, 97.5),
               share_p2_5=round(np.percentile(shares, 2.5), 1), share_p97_5=round(np.percentile(shares, 97.5), 1),
               removal_prob_by_month=";".join(f"{m}:{prob[m]:.2f}" for m in range(1, 13)))
    pd.Series(res).to_csv(f"{pref}_stratified_thinning.csv", header=["value"])
    print(res)


if __name__ == "__main__":
    main(*sys.argv[1:4])
