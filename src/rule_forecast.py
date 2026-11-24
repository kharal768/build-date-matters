"""Forecast indicator completeness for a cohort from information available at the forecast date.

Naive forecast: the previous cohort's completeness curve, aligned on the end of the window.
Structural forecast: C(b) = mean over the previous cohort's positive pairs of
    1 - prod_f [1 - G_b(b - (fd_f + 1 year))],
the previous cohort's follow-on filing pattern shifted forward one year, combined with a
publication-lag distribution G_b estimated only from applications filed GLO-GHI months before b
(default 24-36) and published by b. A calibrated variant is also written.
Inputs: folder holding pg_published_application.tsv.zip and pg_cpc_at_issue.tsv.zip.
Output: <out_csv> with forecasts by cohort and month relative to the window's close.
Paper:  Section 5.8, Table 8, Figure 3 (fields B25J, A61B, G06N; held-out field H01M).
Usage:  python3 -I rule_forecast.py <pg_folder> <out_csv> [cpc_subclass] [first_cohort]
"""
import sys, zipfile, itertools
import numpy as np, pandas as pd

base, outp = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(f"{base}/pg_cpc_at_issue.tsv.zip")
parts = []
with z.open(z.namelist()[0]) as fh:
    for ch in pd.read_csv(fh, sep="\t", usecols=["pgpub_id", "cpc_subclass"], dtype=str, chunksize=2_000_000):
        parts.append(ch.drop_duplicates())
cpc = pd.concat(parts).drop_duplicates()
SUB = sys.argv[3] if len(sys.argv) > 3 else "B25J"
C0 = int(sys.argv[4]) if len(sys.argv) > 4 else 2019
rob = set(cpc.loc[cpc.cpc_subclass == SUB, "pgpub_id"]); cpc = cpc[cpc.pgpub_id.isin(rob)]
z = zipfile.ZipFile(f"{base}/pg_published_application.tsv.zip")
with z.open(z.namelist()[0]) as fh:
    app = pd.read_csv(fh, sep="\t", usecols=["pgpub_id", "application_id", "filing_date", "published_date"], dtype=str)
app = app[app.pgpub_id.isin(rob)]
app["fd"] = pd.to_datetime(app.filing_date, errors="coerce"); app["pd"] = pd.to_datetime(app.published_date, errors="coerce")
app = app.dropna(subset=["fd", "pd"]).sort_values("pd").drop_duplicates("application_id")
app = app[app.fd.dt.year >= 2005]
subs = cpc.groupby("pgpub_id").cpc_subclass.apply(lambda s: sorted(set(s)))
app = app[app.pgpub_id.isin(subs.index)]; app["subs"] = app.pgpub_id.map(subs); app["y"] = app.fd.dt.year
pairs = {}
for r in app.itertuples():
    for p in itertools.combinations(r.subs, 2):
        pairs.setdefault(p, []).append((r.y, r.fd, r.pd))

def followons(c):
    out = []
    for p, rr in pairs.items():
        if min(y for y, _, _ in rr) != c: continue
        F = [(fd, pdt) for y, fd, pdt in rr if 1 <= y - c <= 2]
        if F: out.append(F)
    return out

import os
GHI, GLO = int(os.environ.get("GHI", 36)), int(os.environ.get("GLO", 24))
def G_at(b):
    w = app[(app.fd > b - pd.DateOffset(months=GHI)) & (app.fd <= b - pd.DateOffset(months=GLO)) & (app.pd <= b)]
    lag = ((w.pd - w.fd).dt.days / 30.4375).values
    return lambda t: np.mean(lag[None, :] <= np.asarray(t)[:, None], axis=1) if len(lag) else np.zeros(len(t))

def struct(F_list, b, shift_years, G):
    s = []
    for F in F_list:
        t = np.array([(b - (fd + pd.DateOffset(years=shift_years))).days / 30.4375 for fd, _ in F])
        s.append(1 - np.prod(1 - G(t)))
    return float(np.mean(s))

rows = []
FO = {c: followons(c) for c in range(C0 - 2, 2022)}
for c in range(C0, 2022):
    prev, prev2, cur = FO[c - 1], FO[c - 2], FO[c]
    for m in range(-6, 13):
        b = pd.Timestamp(f"{c + 2}-12-31") + pd.offsets.MonthEnd(m)
        bp = b - pd.DateOffset(years=1)
        actual = np.mean([any(x <= b for _, x in F) for F in cur])
        naive = np.mean([any(x <= bp for _, x in F) for F in prev])
        G = G_at(b)
        st = struct(prev, b, 1, G)
        # calibration on cohort c-1 at aligned date bp, using only data visible at b
        prev_vis = [F for F in prev if any(x <= b for _, x in F)]
        obs_prev = np.mean([any(x <= bp for _, x in F) for F in prev_vis]) if prev_vis else np.nan
        st_prev = struct(prev2, bp, 1, G_at(bp))
        cal = min(1.0, st * obs_prev / st_prev) if st_prev > 0 else st
        rows.append(dict(field=SUB, cohort=c, month=m, actual=round(actual, 3), naive=round(naive, 3),
                         structural=round(st, 3), calibrated=round(cal, 3), positives=len(cur)))
r = pd.DataFrame(rows); r.to_csv(outp, index=False)
for meth in ("naive", "structural", "calibrated"):
    e = r[meth] - r.actual
    late = [(g.month[g[meth] >= 0.9].min(), g.month[g.actual >= 0.9].min()) for _, g in r.groupby("cohort")]
    print(SUB, meth, "MAE", round(e.abs().mean(), 3), "over>2pt", int((e > 0.02).sum()), "/", len(r), "max over", round(e.max(), 3), "month90 offsets", late)
