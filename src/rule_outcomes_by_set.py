"""Outcomes of the two-month rule (naive estimate) by data set.

Counts cohorts declared early, on time or later, and never reaching 90% within twelve months,
for the tuning set, the B23K held-out field and the CNRDS cohorts.
Inputs: forecast CSVs in the current folder.
Output: <out_csv>.
Paper:  Section 5.8.
Usage:  python3 -I rule_outcomes_by_set.py <out_csv>   (run in the results folder)
"""
import sys
import numpy as np, pandas as pd
sets = {"tuning": ["rule_forecast_B25J.csv", "rule_forecast_A61B.csv", "rule_forecast_G06N.csv", "cn_google_forecast.csv"],
        "heldout": ["cn_google_forecast_B23K.csv"], "cnrds": ["cn_rule_forecast.csv"]}
rows = []
for name, fs in sets.items():
    E = S = N = 0; d = []
    for f in fs:
        r = pd.read_csv(f); keys = ["rule", "cohort"] if "rule" in r else ["cohort"]
        for _, g in r.groupby(keys):
            a = g.month[g.actual >= 0.9].min(); p = g.month[g.naive >= 0.9].min()
            if np.isnan(p): N += 1; continue
            p += 2; a = 13 if np.isnan(a) else a
            if p < a: E += 1
            else: S += 1; d.append(p - a)
    rows.append(dict(set=name, early=E, not_early=S, never_reached_within_12=N, median_delay=np.median(d), max_delay=max(d)))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
