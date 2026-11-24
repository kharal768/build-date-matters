"""Completeness forecast test on Chinese invention applications from Google Patents Public Data.

Input is an export of sql/google_patents_query.sql (application, filing date, first publication
date, IPC subclasses, hashed applicant names). Pairs are IPC subclass pairs; a follow-on is an
application filed one or two years after the debut whose applicants are disjoint from the
debut-year applicants. Cohorts 2016-2022 are forecast from the previous cohort with the naive and
structural estimates (lag window GLO-GHI months).
Input:  export CSV (B25J, or B23K for the held-out field).
Output: <out_csv>.
Paper:  Section 5.8, Table 8.
Usage:  python3 -I cn_google_forecast.py <export_csv> <out_csv>
"""
import os, sys
from itertools import combinations
import numpy as np, pandas as pd

GHI, GLO = int(os.environ.get("GHI", 36)), int(os.environ.get("GLO", 24))
d = pd.read_csv(sys.argv[1], dtype=str)
if "app" not in d.columns: d["app"] = d.index.astype(str)
d["fdt"] = pd.to_datetime(d.fd, format="%Y%m%d"); d["pdt"] = pd.to_datetime(d.pd, format="%Y%m%d")
d["y"] = d.fdt.dt.year
d["S"] = d.asg.fillna("").map(lambda s: frozenset(x for x in s.split("|") if x))
d["I"] = d["sub"].fillna("").map(lambda s: sorted(set(x for x in s.split(";") if x)))
pairs = {}
for r in d.itertuples():
    for p in combinations(r.I, 2):
        pairs.setdefault(p, []).append((r.y, r.S, r.fdt, r.pdt))
FO = {}
for p, rr in pairs.items():
    c = min(x[0] for x in rr)
    D = frozenset().union(*[S for y, S, _, _ in rr if y == c])
    F = [(fd, pdt) for y, S, fd, pdt in rr if 1 <= y - c <= 2 and S and not (S & D)]
    if F:
        FO.setdefault(c, []).append(F)
fd_all, lag_all = d.fdt.values, ((d.pdt - d.fdt).dt.days / 30.4375).values

def G_at(b):
    m = (d.fdt > b - pd.DateOffset(months=GHI)) & (d.fdt <= b - pd.DateOffset(months=GLO)) & (d.pdt <= b)
    lag = np.sort(lag_all[m.values])
    return lambda t: np.searchsorted(lag, np.asarray(t), side="right") / len(lag)

SUP = sorted({(fd, pdt) for L in FO.values() for F in L for fd, pdt in F})
sup_fd = np.array([x[0] for x in SUP], dtype="datetime64[ns]"); sup_pd = np.array([x[1] for x in SUP], dtype="datetime64[ns]")
sup_lag = (sup_pd - sup_fd).astype("timedelta64[D]").astype(float) / 30.4375

def Gs_at(b):
    bb = np.datetime64(b)
    m = (sup_fd > np.datetime64(b - pd.DateOffset(months=GHI))) & (sup_fd <= np.datetime64(b - pd.DateOffset(months=GLO))) & (sup_pd <= bb)
    lag = np.sort(sup_lag[m])
    return lambda t: np.searchsorted(lag, np.asarray(t), side="right") / len(lag)

rows = []
for c in range(2016, 2023):
    prev, cur = FO.get(c - 1, []), FO.get(c, [])
    for m in range(-6, 13):
        b = pd.Timestamp(f"{c + 2}-12-31") + pd.offsets.MonthEnd(m)
        bp = b - pd.DateOffset(years=1)
        actual = np.mean([any(x <= b for _, x in F) for F in cur])
        naive = np.mean([any(x <= bp for _, x in F) for F in prev])
        G = G_at(b)
        st = np.mean([1 - np.prod(1 - G(np.array([(b - (fd + pd.DateOffset(years=1))).days / 30.4375 for fd, _ in F]))) for F in prev])
        Gs = Gs_at(b)
        ss = np.mean([1 - np.prod(1 - Gs(np.array([(b - (fd + pd.DateOffset(years=1))).days / 30.4375 for fd, _ in F]))) for F in prev])
        rows.append(dict(cohort=c, month=m, positives=len(cur), actual=round(actual, 4), naive=round(naive, 4), structural=round(float(st), 4), support=round(float(ss), 4)))
r = pd.DataFrame(rows); r.to_csv(sys.argv[2], index=False)
for c, g in r.groupby("cohort"):
    f = lambda s: g.month[g[s] >= 0.9].min()
    print(c, "n", g.positives.iloc[0], "m0 a/n/s", g[g.month == 0][["actual", "naive", "structural", "support"]].values.round(3).tolist()[0], "month90 a/n/s/sup", f("actual"), f("naive"), f("structural"), f("support"))
