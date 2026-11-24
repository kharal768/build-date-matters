"""Completeness forecast test on the CNRDS robotics table.

Reference: the later build. A follow-on key is visible at b if its first public date (earlier of
the recovered publication date and the CNRDS grant date) is on or before b. Positives are pairs with
at least one dated qualifying follow-on. Cohorts 2019-2022 are forecast from the previous cohort
with the naive and structural estimates (lag window GLO-GHI months, default 24-36).
Inputs: later build (.xlsx), recovered dates (CSV), support_dates.csv.
Output: <out_csv>.
Paper:  Section 5.8 (CNRDS cohorts).
Usage:  python3 -I cnrds_rule_forecast.py <later_xlsx> <metadata_csv> <support_dates_csv> <out_csv>
"""
import os, sys
from itertools import combinations
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recurrence import load, keys  # noqa
from publication_truncation_sim import first_public  # noqa

GHI, GLO = int(os.environ.get("GHI", 36)), int(os.environ.get("GLO", 24))


def main(refr, meta_csv, sup_csv, outp):
    dr = load(refr)
    meta = pd.read_csv(meta_csv, dtype=str)
    e = pd.read_csv(sup_csv, dtype=str)
    pub = first_public(dr, meta, dict(zip(e.key, e.publication_date)))
    fdate = pd.to_datetime(dr.groupby("key").apdt.first())
    lagdf = pd.DataFrame({"fd": fdate, "pd": pub.reindex(fdate.index)}).dropna()
    rows = []
    for rule in (False, True):
        K = keys(dr, rule)
        pairs = {}
        for k, (I, S, y) in K.items():
            for p in combinations(sorted(I), 2):
                pairs.setdefault(p, []).append((k, S, y))
        FO = {}
        for p, rr in pairs.items():
            d = min(y for _, _, y in rr)
            D = frozenset().union(*[S for _, S, y in rr if y == d])
            F = [k for k, S, y in rr if 1 <= y - d <= 2 and not (S & D)]
            F = [(fdate[k], pub[k]) for k in F if k in pub.index and pd.notna(pub.get(k))]
            if F:
                FO.setdefault(d, []).append(F)

        def G_at(b):
            w = lagdf[(lagdf.fd > b - pd.DateOffset(months=GHI)) & (lagdf.fd <= b - pd.DateOffset(months=GLO)) & (lagdf.pd <= b)]
            lag = ((w.pd - w.fd).dt.days / 30.4375).values
            return lambda t: np.mean(lag[None, :] <= np.asarray(t)[:, None], axis=1)

        for c in range(2019, 2023):
            prev, cur = FO.get(c - 1, []), FO.get(c, [])
            for m in range(-6, 13):
                b = pd.Timestamp(f"{c + 2}-12-31") + pd.offsets.MonthEnd(m)
                bp = b - pd.DateOffset(years=1)
                actual = np.mean([any(x <= b for _, x in F) for F in cur])
                naive = np.mean([any(x <= bp for _, x in F) for F in prev])
                G = G_at(b)
                st = np.mean([1 - np.prod(1 - G(np.array([(b - (fd + pd.DateOffset(years=1))).days / 30.4375 for fd, _ in F]))) for F in prev])
                rows.append(dict(rule="restricted" if rule else "all_links", cohort=c, month=m, positives=len(cur),
                                 prev_positives=len(prev), actual=round(actual, 3), naive=round(naive, 3), structural=round(float(st), 3)))
    r = pd.DataFrame(rows); r.to_csv(outp, index=False)
    for (rl, c), g in r.groupby(["rule", "cohort"]):
        f = lambda s: g.month[g[s] >= 0.9].min()
        print(rl, c, "n", g.positives.iloc[0], "m0 a/n/s", g[g.month == 0][["actual", "naive", "structural"]].values.tolist()[0],
              "month90 a/n/s", f("actual"), f("naive"), f("structural"))


if __name__ == "__main__":
    main(*sys.argv[1:5])
