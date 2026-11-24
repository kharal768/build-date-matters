"""Share of keys introducing a new IPC subclass pair, January-September vs October-December filings.

Later CNRDS build, all links, Wilson 95% intervals.
Input:  later build (.xlsx).
Output: <out_csv>.
Paper:  Section 5.5.
Usage:  python3 -I novelty_by_quarter.py <later_xlsx> <out_csv>
"""
import os, sys
from itertools import combinations
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recurrence import load, keys  # noqa
dr = load(sys.argv[1]); K = keys(dr, False)
mo = pd.to_datetime(dr.groupby("key").apdt.first()).dt.month
first = {}
for k, (I, S, y) in sorted(K.items(), key=lambda x: x[1][2]):
    for p in combinations(sorted(I), 2):
        first.setdefault(p, y)
rows = []
for yrs in ((2024,), (2020, 2021, 2022, 2023, 2024)):
    for lab, ms in (("Jan-Sep", range(1, 10)), ("Oct-Dec", range(10, 13))):
        ks = [k for k, (I, S, y) in K.items() if y in yrs and mo[k] in ms]
        n = sum(any(first[p] == K[k][2] for p in combinations(sorted(K[k][0]), 2)) for k in ks)
        a, b = proportion_confint(n, len(ks), method="wilson")
        rows.append(dict(years="-".join(map(str, yrs)), months=lab, keys=len(ks), novel=n,
                         share=round(100 * n / len(ks), 1), lo=round(100 * a, 1), hi=round(100 * b, 1)))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
