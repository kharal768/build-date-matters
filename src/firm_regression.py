"""Firm-level models estimated separately on each CNRDS build.

Unit: listed stock code with at least one robotics invention key filed 2020-2024 (all links).
Outcome: number of 2022-cohort IPC pairs for which the firm is linked to a qualifying two-year
follow-on. Covariates are fixed across builds (controller type, largest stake, prior robot
applications, optional 2021 financial covariates). Poisson GLM with HC1 errors; logit on any
follow-on as a check.
Shared module: model helpers are imported by cutoff_scan_and_novelty.py and
matched_rates_and_intervals.py.
Inputs: both builds, CERD controllers, firm applications, optional covariates CSV.
Output: <out_csv>.
Paper:  Appendix E.
Usage:  python3 -I firm_regression.py <earlier_dta> <later_xlsx> <cerd_xlsx> <firm_apps_dta> <out_csv>
        [covariates_csv] [main|cird]
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from recurrence import load, keys  # noqa: E402
from cohort_trend_and_validation import ledger  # noqa: E402

STATE = ("国", "政府", "中央", "集体")


def covariates(cerd_xlsx, apps_dta):
    c = pd.read_excel(cerd_xlsx, dtype=str).iloc[1:]
    c = c[c.Year == "2021"].drop_duplicates("SCode")
    cov = pd.DataFrame({
        "scode": c.SCode,
        "soe": c.ACNtr.fillna("").apply(lambda s: int(any(t in s for t in STATE))),
        "top1": pd.to_numeric(c.LrgHldRt, errors="coerce"),
    }).set_index("scode")
    a = pd.read_stata(apps_dta)
    a = a[(a.Ftyp == "集团公司合计") & a.Year.astype(int).between(2019, 2021)]
    a = a.assign(RInvia=pd.to_numeric(a.RInvia, errors="coerce").fillna(0))
    prior = a.groupby("Scode").RInvia.sum()
    cov["log_prior"] = np.log1p(prior.reindex(cov.index).fillna(0))
    return cov


def outcome(path):
    d = load(path)
    K = keys(d, False)
    universe = {s for k, (I, S, y) in K.items() if 2020 <= y <= 2024 for s in S}
    y = dict.fromkeys(universe, 0)
    for p, (dp, F) in ledger(K).items():
        if dp != 2022:
            continue
        for s in {s for _, S in F for s in S}:
            y[s] = y.get(s, 0) + 1
    return pd.Series(y, name="followon_pairs")


TERMS = ["soe", "top1", "log_prior"]


def fit(df, label):
    X = sm.add_constant(df[TERMS])
    rows = []
    pois = sm.GLM(df.followon_pairs, X, family=sm.families.Poisson()).fit(cov_type="HC1")
    logit = sm.Logit((df.followon_pairs > 0).astype(int), X).fit(disp=0, cov_type="HC1")
    for name, m in (("poisson", pois), ("logit_any", logit)):
        for v in X.columns[1:]:
            rows.append(dict(extract=label, model=name, term=v,
                             coef=m.params[v], se=m.bse[v], p=m.pvalues[v],
                             n=int(m.nobs),
                             positives=int((df.followon_pairs > 0).sum())))
    return rows


def main(orig, refr, cerd, apps, out, balance=None, spec="main"):
    cov = covariates(cerd, apps)
    if balance:  # optional 2021 CNFS covariates (log_assets, leverage, rd_int)
        b = pd.read_csv(balance, dtype={"scode": str}).set_index("scode")
        cov = cov.join(b, how="inner")
        if spec == "main":
            b = b.drop(columns=["log_inv_all"], errors="ignore")
        else:  # "cird": total invention applications replace robot-only prior
            TERMS.remove("log_prior")
        TERMS.extend(b.columns.tolist())
    res = []
    for label, path in (("original", orig), ("refreshed", refr)):
        df = cov.join(outcome(path), how="inner").dropna()
        res += fit(df, label)
    r = pd.DataFrame(res)
    r.to_csv(out, index=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(r.pivot_table(index=["model", "term"], columns="extract",
                            values=["coef", "p"]))
        print(r.groupby("extract")[["n", "positives"]].first())


if __name__ == "__main__":
    main(*sys.argv[1:8])
