"""Month-end cutoff scan, monthly visibility, Wilson intervals, common-sample logit and novelty share.

1. Agreement between the earlier build's 2022 labels and the later build truncated at each
   month-end publication cutoff (lower and upper bounds).
2. Monthly visibility of the later build's 2022 cohort positives.
3. Wilson 95% intervals for matched 2022 two-year rates.
4. Common-sample logit (size, leverage) with 95% intervals.
5. Share of invention keys introducing a subclass pair not seen earlier in the build, by filing year.
Inputs: both builds, recovered dates, support_dates.csv, CERD controllers, firm applications,
        firm covariates (from build_covariates.py).
Outputs: <prefix>_cutoff_scan.csv, <prefix>_monthly_visibility.csv, <prefix>_wilson.csv,
         <prefix>_common_logit.csv, <prefix>_novelty.csv.
Paper:  Table 4, Section 5.3, Table 6, Appendix E.
Usage:  python3 -I cutoff_scan_and_novelty.py <earlier_dta> <later_xlsx> <metadata_csv> <support_dates_csv>
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


def main(orig, refr, meta_csv, sup_csv, cerd, apps, cov_csv, pref):
    do, dr = load(orig), load(refr)
    meta = pd.read_csv(meta_csv, dtype=str)
    e = pd.read_csv(sup_csv, dtype=str)
    pub = first_public(dr, meta, dict(zip(e.key, e.publication_date)))
    cut = pd.date_range("2024-06-30", "2025-12-31", freq="ME")
    scan, vis, rates = [], [], []
    for rule in (False, True):
        name = "restricted" if rule else "all_links"
        Lo, Lr = ledger(keys(do, rule)), ledger(keys(dr, rule))
        m = [p for p in Lo if Lo[p][0] == 2022 and p in Lr and Lr[p][0] == 2022]
        yo = {p: bool(Lo[p][1]) for p in m}
        ref22 = [F for d, F in Lr.values() if d == 2022]
        npos = sum(bool(F) for F in ref22)
        for T in cut:
            lo = {p: any(k in pub.index and pub[k] <= T for k, _ in Lr[p][1]) for p in m}
            hi = {p: any(k not in pub.index or pub[k] <= T for k, _ in Lr[p][1]) for p in m}
            scan.append(dict(rule=name, cutoff=T.date(), n=len(m),
                             agree_lower=sum(lo[p] == yo[p] for p in m),
                             agree_upper=sum(hi[p] == yo[p] for p in m)))
            vl = sum(any(k in pub.index and pub[k] <= T for k, _ in F) for F in ref22)
            vu = sum(any(k not in pub.index or pub[k] <= T for k, _ in F) for F in ref22)
            vis.append(dict(rule=name, cutoff=T.date(), ref=npos, lower=vl, upper=vu))
        for lab, y in (("earlier", yo), ("later", {p: bool(Lr[p][1]) for p in m})):
            k = sum(y.values())
            lo_, hi_ = proportion_confint(k, len(m), method="wilson")
            rates.append(dict(rule=name, extract=lab, positives=k, n=len(m),
                              rate=round(100 * k / len(m), 1),
                              ci_low=round(100 * lo_, 1), ci_high=round(100 * hi_, 1)))
    pd.DataFrame(scan).to_csv(f"{pref}_cutoff_scan.csv", index=False)
    pd.DataFrame(vis).to_csv(f"{pref}_monthly_visibility.csv", index=False)
    pd.DataFrame(rates).to_csv(f"{pref}_wilson.csv", index=False)

    # 4. common-sample parsimonious logit
    cov = fr.covariates(cerd, apps).join(
        pd.read_csv(cov_csv, dtype={"scode": str}).set_index("scode"), how="inner")
    yo_ = fr.outcome(orig).rename("y_o")
    yr_ = fr.outcome(refr).rename("y_r")
    df = cov.join(yo_, how="inner").join(yr_, how="inner").dropna()
    reg = []
    for lab, col in (("earlier", "y_o"), ("later", "y_r")):
        X = sm.add_constant(df[["log_assets", "leverage"]])
        res = sm.Logit((df[col] > 0).astype(int), X).fit(disp=0, cov_type="HC1")
        ci = res.conf_int()
        for t in ("log_assets", "leverage"):
            reg.append(dict(extract=lab, term=t, coef=res.params[t], se=res.bse[t],
                            ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1], p=res.pvalues[t],
                            n=len(df), events=int((df[col] > 0).sum())))
    pd.DataFrame(reg).to_csv(f"{pref}_common_logit.csv", index=False)

    # 5. novelty-in-recombination indicator
    nov = []
    for lab, d in (("earlier", do), ("later", dr)):
        K = keys(d, False)
        first = {}
        for k, (I, S, y) in sorted(K.items(), key=lambda x: x[1][2]):
            for p in combinations(sorted(I), 2):
                first.setdefault(p, y)
        for yr in range(2018, 2025):
            ks = [(k, I) for k, (I, S, y) in K.items() if y == yr]
            new = sum(any(first[p] == yr for p in combinations(sorted(I), 2)) for k, I in ks)
            nov.append(dict(extract=lab, year=yr, keys=len(ks), novel=new,
                            share=round(100 * new / len(ks), 1)))
    pd.DataFrame(nov).to_csv(f"{pref}_novelty.csv", index=False)
    for f in ("cutoff_scan", "monthly_visibility", "wilson", "common_logit", "novelty"):
        print(f"--- {f}")
        print(pd.read_csv(f"{pref}_{f}.csv").round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:9])
