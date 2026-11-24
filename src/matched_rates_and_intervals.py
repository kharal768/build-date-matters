"""Matched cohort rates, visibility of dated positives, novelty intervals, controller variant, leverage bootstrap.

a) Matched-pair cohort rates and paired bootstrap of the 2022 rate difference and of the
   2021-2022 decline difference between builds.
b) Paired (firm) bootstrap of the leverage-coefficient difference on the common sample.
c) Monthly visibility among positives with at least one dated follow-on.
d) Share of keys versus share of pair positives visible at 2024-12-31.
e) Novelty share by filing year with Wilson intervals, and a proportional-thinning check.
f) Controller-disjoint follow-on variant using CERD annual controller names.
Inputs: as for cutoff_scan_and_novelty.py.
Outputs: <prefix>_matched_rates.csv, <prefix>_monthly_dated.csv, <prefix>_novelty_ci.csv,
         <prefix>_controller_disjoint.csv, <prefix>_summary.csv.
Paper:  Table 2, Figure 1, Table 5 (controller-disjoint), Table 6, Appendix E.
Usage:  python3 -I matched_rates_and_intervals.py <earlier_dta> <later_xlsx> <metadata_csv> <support_csv>
        <cerd_xlsx> <firm_apps_dta> <covariates_csv> <out_prefix>
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recurrence import load, keys  # noqa: E402
from cohort_trend_and_validation import ledger  # noqa: E402
from publication_truncation_sim import first_public  # noqa: E402
import firm_regression as fr  # noqa: E402

rng = np.random.default_rng(20261124)
B = 2000


def matched(Lo, Lr, c):
    return [p for p in Lo if Lo[p][0] == c and p in Lr and Lr[p][0] == c]


def main(orig, refr, meta_csv, sup_csv, cerd, apps, cov_csv, pref):
    do, dr = load(orig), load(refr)
    out = {}
    # a) matched rates + bootstrap
    rows = []
    for rule in (False, True):
        nm = "restricted" if rule else "all_links"
        Lo, Lr = ledger(keys(do, rule)), ledger(keys(dr, rule))
        res = {}
        for c in (2021, 2022):
            m = matched(Lo, Lr, c)
            yo = np.array([bool(Lo[p][1]) for p in m]); yr = np.array([bool(Lr[p][1]) for p in m])
            res[c] = (yo, yr)
            rows.append(dict(rule=nm, cohort=c, n=len(m), earlier=yo.sum(), later=yr.sum(),
                             rate_earlier=round(100 * yo.mean(), 1), rate_later=round(100 * yr.mean(), 1)))
        (o21, r21), (o22, r22) = res[2021], res[2022]
        d22, dd = [], []
        for _ in range(B):
            i = rng.integers(0, len(o22), len(o22)); j = rng.integers(0, len(o21), len(o21))
            d22.append(100 * (r22[i].mean() - o22[i].mean()))
            dd.append(100 * ((o21[j].mean() - o22[i].mean()) - (r21[j].mean() - r22[i].mean())))
        out[f"{nm}_rate_diff_2022"] = (round(100 * (r22.mean() - o22.mean()), 1),
                                       np.round(np.percentile(d22, [2.5, 97.5]), 1).tolist())
        out[f"{nm}_decline_diff"] = (round(100 * ((o21.mean() - o22.mean()) - (r21.mean() - r22.mean())), 1),
                                     np.round(np.percentile(dd, [2.5, 97.5]), 1).tolist())
    pd.DataFrame(rows).to_csv(f"{pref}_matched_rates.csv", index=False)

    # b) paired bootstrap leverage difference
    cov = fr.covariates(cerd, apps).join(pd.read_csv(cov_csv, dtype={"scode": str}).set_index("scode"), how="inner")
    df = cov.join(fr.outcome(orig).rename("yo"), how="inner").join(fr.outcome(refr).rename("yr"), how="inner").dropna()
    X = sm.add_constant(df[["log_assets", "leverage"]]).values
    yo_ = (df.yo > 0).astype(int).values; yr_ = (df.yr > 0).astype(int).values

    def lev(Xb, y):
        return sm.Logit(y, Xb).fit(disp=0).params[2]
    base = lev(X, yr_) - lev(X, yo_)
    diffs = []
    for _ in range(B):
        i = rng.integers(0, len(df), len(df))
        try:
            diffs.append(lev(X[i], yr_[i]) - lev(X[i], yo_[i]))
        except Exception:
            pass
    out["leverage_diff_later_minus_earlier"] = (round(base, 2), np.round(np.percentile(diffs, [2.5, 97.5]), 2).tolist(),
                                                round(float(np.mean(np.array(diffs) <= 0)), 3), len(diffs))

    # c) monthly visibility among dated positives; d) keys vs positives at 2024-12-31
    meta = pd.read_csv(meta_csv, dtype=str); e = pd.read_csv(sup_csv, dtype=str)
    pub = first_public(dr, meta, dict(zip(e.key, e.publication_date)))
    vis = []
    for rule in (False, True):
        nm = "restricted" if rule else "all_links"
        Lr = ledger(keys(dr, rule))
        F22 = [[k for k, _ in F] for d, F in Lr.values() if d == 2022 and F]
        dated = [F for F in F22 if any(k in pub.index for k in F)]
        for T in pd.date_range("2024-06-30", "2025-12-31", freq="ME"):
            v = sum(any(k in pub.index and pub[k] <= T for k in F) for F in dated)
            vis.append(dict(rule=nm, cutoff=T.date(), dated_positives=len(dated), visible=v,
                            share=round(100 * v / len(dated), 1)))
    pd.DataFrame(vis).to_csv(f"{pref}_monthly_dated.csv", index=False)
    K = keys(dr, False)
    late = [k for k, (I, S, y) in K.items() if y in (2023, 2024)]
    T = pd.Timestamp("2024-12-31")
    lo = np.mean([k in pub.index and pub[k] <= T for k in late]); hi = np.mean([k not in pub.index or pub[k] <= T for k in late])
    out["keys_2023_2024_visible_at_2024_12_31"] = (round(100 * lo, 1), round(100 * hi, 1), len(late))

    # e) novelty share + thinning
    nov = []
    for lab, d in (("earlier", do), ("later", dr)):
        K = keys(d, False)
        first = {}
        for k, (I, S, y) in sorted(K.items(), key=lambda x: x[1][2]):
            for p in combinations(sorted(I), 2):
                first.setdefault(p, y)
        for yr in range(2016, 2025):
            ks = [I for k, (I, S, y) in K.items() if y == yr]
            new = sum(any(first[p] == yr for p in combinations(sorted(I), 2)) for I in ks)
            a, b = proportion_confint(new, len(ks), method="wilson")
            nov.append(dict(extract=lab, year=yr, keys=len(ks), novel=new, share=round(100 * new / len(ks), 1),
                            ci_low=round(100 * a, 1), ci_high=round(100 * b, 1)))
    pd.DataFrame(nov).to_csv(f"{pref}_novelty_ci.csv", index=False)
    K = keys(dr, False)
    items = sorted(K.items(), key=lambda x: x[1][2])
    shares, counts = [], []
    for _ in range(200):
        keep = [(k, v) for k, v in items if not (v[2] == 2024 and rng.random() < 0.35)]
        first = {}
        for k, (I, S, y) in keep:
            for p in combinations(sorted(I), 2):
                first.setdefault(p, y)
        ks = [I for k, (I, S, y) in keep if y == 2024]
        new = sum(any(first[p] == 2024 for p in combinations(sorted(I), 2)) for I in ks)
        shares.append(100 * new / len(ks)); counts.append(new)
    out["thinning_2024_35pct"] = dict(share_mean=round(np.mean(shares), 1), share_sd=round(np.std(shares), 1),
                                      count_mean=round(np.mean(counts), 1))

    # f) controller-disjoint variant
    c = pd.read_excel(cerd, dtype=str).iloc[1:]
    c["names"] = c.ACNm.fillna("").apply(lambda s: frozenset(x.strip() for x in s.replace("；", ";").split(";") if x.strip()))
    ctrl = {(r.SCode, int(r.Year)): r.names for r in c.itertuples() if r.Year and r.Year.isdigit()}
    cr = []
    for rule in (False, True):
        nm = "restricted" if rule else "all_links"
        Ko, Kr = keys(do, rule), keys(dr, rule)

        def lab(K):
            pairs = {}
            for k, (I, S, y) in K.items():
                for p in combinations(sorted(I), 2):
                    pairs.setdefault(p, []).append((k, S, y))
            res = {}
            for p, rr in pairs.items():
                d = min(y for _, _, y in rr)
                if d != 2022:
                    continue
                D = frozenset().union(*[S for _, S, y in rr if y == d])
                DN = frozenset().union(*[ctrl.get((s, d), frozenset()) for s in D])
                ok = False
                for k, S, y in rr:
                    if 1 <= y - d <= 2 and not (S & D):
                        N = frozenset().union(*[ctrl.get((s, y), frozenset()) for s in S])
                        if not (N & DN):
                            ok = True
                res[p] = ok
            return res
        lo_, lr_ = lab(Ko), lab(Kr)
        m = [p for p in lo_ if p in lr_]
        cr.append(dict(rule=nm, n=len(m), earlier=sum(lo_[p] for p in m), later=sum(lr_[p] for p in m),
                       gains=sum((not lo_[p]) and lr_[p] for p in m), losses=sum(lo_[p] and not lr_[p] for p in m)))
    pd.DataFrame(cr).to_csv(f"{pref}_controller_disjoint.csv", index=False)
    pd.Series({k: str(v) for k, v in out.items()}).to_csv(f"{pref}_summary.csv", header=["value"])
    for f in ("matched_rates", "monthly_dated", "novelty_ci", "controller_disjoint"):
        print("---", f); print(pd.read_csv(f"{pref}_{f}.csv").to_string())
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main(*sys.argv[1:9])
