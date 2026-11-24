"""Publication-cutoff simulation on the later CNRDS build.

The later build is the reference. A follow-on key is visible at cutoff T if its first public date
(recovered application publication date, or the CNRDS grant date if earlier) is on or before T.
Keys without a recovered date are treated as not yet visible ('lower') or visible ('upper').
Debut years, debut links and eligible pairs stay at their reference values.
Shared module: first_public() is imported by other scripts.
Inputs: later build (.xlsx), recovered publication dates (CSV), data/derived/support_dates.csv.
Output: <out_csv> with retained positives by cutoff and cohort.
Paper:  Section 5.3 (publication cutoff).
Usage:  python3 -I publication_truncation_sim.py <later_build_xlsx> <metadata_csv> <support_dates_csv> <out_csv>
"""
import sys
from itertools import combinations

import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from recurrence import load, keys  # noqa: E402


def first_public(d, meta, extra):
    pub = (meta.dropna(subset=["cn_a_publication_date"])
           .drop_duplicates("provisional_id")
           .set_index("provisional_id")["cn_a_publication_date"])
    pub = pd.to_datetime(pub)
    for k, v in extra.items():
        pub.loc[k] = pd.Timestamp(v)
    grant = pd.to_datetime(d.loc[d.grdt.str.len() > 0].groupby("key").grdt.min())
    both = pd.concat([pub, grant], axis=1, keys=["a", "b"])
    return both.min(axis=1)


def main(xlsx, meta_csv, extra_csv, out):
    d = load(xlsx)
    meta = pd.read_csv(meta_csv, dtype=str)
    extra = {}
    if extra_csv != "-":
        e = pd.read_csv(extra_csv, dtype=str)
        extra = dict(zip(e.key, e.publication_date))
    pub = first_public(d, meta, extra)
    cutoffs = pd.date_range("2024-06-30", "2026-06-30", freq="QE")
    rows = []
    for rule in (False, True):
        K = keys(d, rule)
        pairs = {}
        for k, (I, S, y) in K.items():
            for p in combinations(sorted(I), 2):
                pairs.setdefault(p, []).append((k, S, y))
        ref = []
        for p, rr in pairs.items():
            dp = min(y for _, _, y in rr)
            if not 2018 <= dp <= 2022:
                continue
            D = frozenset().union(*[S for _, S, y in rr if y == dp])
            f = [k for k, S, y in rr if 1 <= y - dp <= 2 and not (S & D)]
            ref.append((dp, f))
        for T in cutoffs:
            for cohort in range(2018, 2023):
                sub = [f for dp, f in ref if dp == cohort]
                lo = sum(any(k in pub.index and pub[k] <= T for k in f)
                         for f in sub)
                hi = sum(any((k not in pub.index) or pub[k] <= T for k in f)
                         for f in sub)
                rows.append(dict(rule="restricted" if rule else "all_links",
                                 cutoff=T.date(), cohort=cohort,
                                 pairs=len(sub),
                                 ref_positives=sum(bool(f) for f in sub),
                                 visible_lower=lo, visible_upper=hi))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    for r in ("all_links", "restricted"):
        x = df[df.rule == r]
        print(r, "reference:", x.groupby("cohort").ref_positives.first().to_dict())
        x = x.assign(cell=x.visible_lower.astype(str) + "-" + x.visible_upper.astype(str))
        print(x.pivot(index="cutoff", columns="cohort", values="cell"))
    missing = sorted(set(d.key) - set(pub.index))
    print("keys without any public date:", len(missing))


if __name__ == "__main__":
    main(*sys.argv[1:5])
