"""Cohort recurrence trends, pair-level check of the cutoff simulation, and firm follow-on counts.

1. Two-year recurrence rate by debut cohort in each build.
2. Pair-level check: does the later build, truncated at a 2024-12-31 publication cutoff,
   reproduce the earlier build's 2022 labels?
3. Stock codes ranked by qualifying follow-on keys (2022 cohort), compared across builds.
Shared module: ledger() is imported by other scripts.
Inputs: earlier build (.dta), later build (.xlsx), recovered dates (CSV), support_dates.csv.
Outputs: <prefix>_cohort_trend.csv, <prefix>_sim_validation.csv, <prefix>_firm_counts.csv.
Paper:  Sections 5.1 and 5.3; Conclusion (decline between the two latest cohorts).
Usage:  python3 -I cohort_trend_and_validation.py <earlier_dta> <later_xlsx> <metadata_csv> <support_dates_csv> <out_prefix>
"""
import sys
from itertools import combinations

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from recurrence import load, keys  # noqa: E402
from publication_truncation_sim import first_public  # noqa: E402


def ledger(K, h=2):
    pairs = {}
    for k, (I, S, y) in K.items():
        for p in combinations(sorted(I), 2):
            pairs.setdefault(p, []).append((k, S, y))
    out = {}
    for p, rr in pairs.items():
        d = min(y for _, _, y in rr)
        if not 2018 <= d <= 2022:
            continue
        D = frozenset().union(*[S for _, S, y in rr if y == d])
        F = [(k, S) for k, S, y in rr if 1 <= y - d <= h and not (S & D)]
        out[p] = (d, F)
    return out


def main(orig, refr, meta_csv, sup_csv, pref):
    do, dr = load(orig), load(refr)
    meta = pd.read_csv(meta_csv, dtype=str)
    e = pd.read_csv(sup_csv, dtype=str)
    pub = first_public(dr, meta, dict(zip(e.key, e.publication_date)))
    T = pd.Timestamp("2024-12-31")
    trend, val, firms = [], [], []
    for rule in (False, True):
        name = "restricted" if rule else "all_links"
        Lo, Lr = ledger(keys(do, rule)), ledger(keys(dr, rule))
        for c in range(2018, 2023):
            for lab, L in (("original", Lo), ("refreshed", Lr)):
                v = [bool(F) for d, F in L.values() if d == c]
                trend.append(dict(rule=name, cohort=c, extract=lab,
                                  pairs=len(v), positives=sum(v),
                                  rate=round(100 * sum(v) / len(v), 1)))
        # validation on matched 2022 pairs
        m = [p for p in Lo if Lo[p][0] == 2022 and p in Lr and Lr[p][0] == 2022]
        for p in m:
            Fr = [k for k, _ in Lr[p][1]]
            lo = any(k in pub.index and pub[k] <= T for k in Fr)
            hi = any(k not in pub.index or pub[k] <= T for k in Fr)
            val.append(dict(rule=name, pair="-".join(p),
                            original=int(bool(Lo[p][1])),
                            refreshed=int(bool(Fr)),
                            sim_lower=int(lo), sim_upper=int(hi)))
        # firm counts: follow-on keys credited to each stock code, 2022 cohort
        for lab, L in (("original", Lo), ("refreshed", Lr)):
            cnt = {}
            for p, (d, F) in L.items():
                if d != 2022:
                    continue
                for k, S in F:
                    for s in S:
                        cnt.setdefault(s, set()).add(k)
            for s, ks in cnt.items():
                firms.append(dict(rule=name, extract=lab, scode=s,
                                  followon_keys=len(ks)))
    tr, va, fi = map(pd.DataFrame, (trend, val, firms))
    tr.to_csv(f"{pref}_cohort_trend.csv", index=False)
    va.to_csv(f"{pref}_sim_validation.csv", index=False)
    fi.to_csv(f"{pref}_firm_counts.csv", index=False)
    print(tr.pivot_table(index=["rule", "cohort"], columns="extract",
                         values="rate"))
    for r, g in va.groupby("rule"):
        print(r, "matched 2022 pairs", len(g),
              "| original pos", g.original.sum(),
              "| sim@2024-12-31 lower/upper", g.sim_lower.sum(), g.sim_upper.sum(),
              "| agree with original (lower)", (g.original == g.sim_lower).sum(),
              "(upper)", (g.original == g.sim_upper).sum())
    for r, g in fi.groupby("rule"):
        w = g.pivot_table(index="scode", columns="extract",
                          values="followon_keys", fill_value=0)
        rho = spearmanr(w.original, w.refreshed).statistic
        top_o = set(w.original.nlargest(10).index)
        top_r = set(w.refreshed.nlargest(10).index)
        print(r, "firms", len(w), "| only in refreshed",
              int(((w.original == 0) & (w.refreshed > 0)).sum()),
              "| Spearman", round(rho, 3),
              "| top-10 overlap", len(top_o & top_r))


if __name__ == "__main__":
    main(*sys.argv[1:6])
