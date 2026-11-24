"""Early declarations and delays by estimate and margin (0-3 months) across the forecast tests.

Inputs: rule_forecast_{B25J,A61B,G06N}.csv, cn_google_forecast.csv, cn_google_forecast_B23K.csv
        in the current folder.
Output: <out_csv>.
Paper:  Table 8, Section 5.8.
Usage:  python3 -I forecast_margin_summary.py <out_csv>   (run in the results folder)
"""
import sys
import numpy as np, pandas as pd
sets = {"US B25J": "rule_forecast_B25J.csv", "US A61B": "rule_forecast_A61B.csv", "US G06N": "rule_forecast_G06N.csv",
        "CN Google B25J": "cn_google_forecast.csv", "CN Google B23K (held out)": "cn_google_forecast_B23K.csv"}
rows = []
for name, f in sets.items():
    r = pd.read_csv(f)
    for meth in ("naive", "structural"):
        e = r[meth] - r.actual
        for k in (0, 1, 2, 3):
            early = 0; d = []; nev = 0
            for c, g in r.groupby("cohort"):
                a = g.month[g.actual >= 0.9].min(); p = g.month[g[meth] >= 0.9].min()
                a = 13 if np.isnan(a) else a
                if np.isnan(p): nev += 1; continue
                p = p + k; early += p < a; d.append(p - a)
            rows.append(dict(data=name, method=meth, margin=k, cohorts=r.cohort.nunique(), early=early,
                             median_delay=np.median(d), max_delay=max(d), forecast_never_90=nev, mae=round(e.abs().mean(), 3)))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
