"""Held-out H01M test of the two-month rule and fixed-wait baseline across all forecast sets.

The H01M field was fixed before its results were computed (data/derived/heldout_field_choice.txt).
Inputs: forecast CSVs in the current folder, including rule_forecast_H01M.csv.
Outputs: heldout_H01M_summary.csv, fixed_horizon_baseline.csv.
Paper:  Section 5.8.
Usage:  python3 -I heldout_and_fixed_horizon.py   (run in the results folder)
"""
import numpy as np, pandas as pd
def m90(s, months):
    x = months[s >= 0.9]; return x.min() if len(x) else np.nan
r = pd.read_csv("rule_forecast_H01M.csv"); rows = []
for meth in ("naive", "structural"):
    for k in (0, 2):
        early = 0; d = []; nev = 0; det = []
        for c, g in r.groupby("cohort"):
            a = m90(g.actual, g.month); a = 13 if np.isnan(a) else a
            p = m90(g[meth], g.month)
            if np.isnan(p): nev += 1; det.append(f"{c}:never"); continue
            p += k; early += p < a; d.append(p - a); det.append(f"{c}:{int(p-a):+d}")
        rows.append(dict(method=meth, margin=k, cohorts=r.cohort.nunique(), early=early,
                         median_delay=np.median(d) if d else np.nan, never=nev, by_cohort=" ".join(det)))
pd.DataFrame(rows).to_csv("heldout_H01M_summary.csv", index=False)
sets = {"US B25J": "rule_forecast_B25J.csv", "US A61B": "rule_forecast_A61B.csv", "US G06N": "rule_forecast_G06N.csv",
        "CN Google B25J": "cn_google_forecast.csv", "CN Google B23K": "cn_google_forecast_B23K.csv", "US H01M": "rule_forecast_H01M.csv"}
rows = []
for name, f in sets.items():
    r = pd.read_csv(f)
    acts = []
    for c, g in r.groupby("cohort"):
        a = m90(g.actual, g.month); acts.append(13 if np.isnan(a) else a)
        nd = m90(g.naive, g.month); acts[-1] = (acts[-1], nd)
    for k in (0, 3, 6, 12):
        A = np.array([a for a, _ in acts])
        rows.append(dict(data=name, rule=f"fixed {k} months", cohorts=len(A), early=int((k < A).sum()), median_delay=float(np.median(k - A))))
    A = np.array([a for a, _ in acts]); N = np.array([n for _, n in acts], float)
    ok = ~np.isnan(N)
    rows.append(dict(data=name, rule="naive + 2 months", cohorts=len(A), early=int(((N[ok] + 2) < A[ok]).sum()),
                     median_delay=float(np.median(N[ok] + 2 - A[ok])) if ok.any() else np.nan))
pd.DataFrame(rows).to_csv("fixed_horizon_baseline.csv", index=False)
