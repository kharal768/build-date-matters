"""Definition checks for the matched 2022 cohort.

Rebuilt checks recompute keys, debuts and eligibility in both builds. Fixed checks keep the primary
matched 2022 pairs and debut links and change only the follow-on rule or the follow-on set.
Inputs: earlier build (.dta), later build (.xlsx).
Output: <out_csv>.
Paper:  Table 5; definitions in Appendix C.
Usage:  python3 -I definition_checks.py <earlier_dta> <later_xlsx> <out_csv>
"""
import sys
from itertools import combinations

import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from recurrence import load, SUBCLASS, RESTRICTED  # noqa: E402

SUPPORT = ["CN119262098", "CN119394150", "CN119579061",
           "CN119682705", "CN119703765", "CN119989630"]
NO_FRONT_PAGE = ["CN119579061", "CN119989630"]


def build_keys(d, restricted, mode="first", start=2015, exclude=()):
    d = d[~d.key.isin(exclude)]
    if mode == "apponly":
        d = d[d.kind_rank == 0]
    out = {}
    for k, g in d.groupby("key"):
        if mode == "union":
            subs = frozenset(s for t in g.P_IPC for s in SUBCLASS.findall(t))
        else:
            first = g.sort_values(["kind_rank", "Scode", "P_IPC"]).iloc[0]
            subs = frozenset(SUBCLASS.findall(first["P_IPC"]))
        links = g[g.Ctlprtrlat.isin(RESTRICTED)] if restricted else g
        codes = frozenset(links.Scode)
        y = int(g.year.iloc[0])
        if not codes or y < start:
            continue
        out[k] = (subs, codes, y, pd.Timestamp(g.apdt.iloc[0]))
    return out


def pair_index(K):
    pairs = {}
    for k, (I, S, y, dt) in K.items():
        for p in combinations(sorted(I), 2):
            pairs.setdefault(p, []).append((k, S, y, dt))
    return pairs


def debuts(P):
    out = {}
    for p, rr in P.items():
        d = min(r[2] for r in rr)
        if 2018 <= d <= 2022:
            D = frozenset().union(*[r[1] for r in rr if r[2] == d])
            dmin = min(r[3] for r in rr if r[2] == d)
            out[p] = (d, D, dmin)
    return out


def followons(rr, d, D, dmin=None, drop=(), anniv=False):
    f = [r[0] for r in rr
         if 1 <= r[2] - d <= 2 and not (r[1] & D) and r[0] not in drop]
    if anniv:
        lim = dmin + pd.DateOffset(years=2)
        f = [r[0] for r in rr if r[0] in f and r[3] <= lim]
    return f


def matched_2022(Po, Pr, Do, Dr):
    return [p for p in Do if Do[p][0] == 2022 and p in Dr and Dr[p][0] == 2022]


def rebuilt(do, dr, rule, **kw):
    Ko, Kr = build_keys(do, rule, **kw), build_keys(dr, rule, **kw)
    Po, Pr = pair_index(Ko), pair_index(Kr)
    Do, Dr = debuts(Po), debuts(Pr)
    m = matched_2022(Po, Pr, Do, Dr)
    yo = {p: bool(followons(Po[p], *Do[p][:2])) for p in m}
    yr = {p: bool(followons(Pr[p], *Dr[p][:2])) for p in m}
    return summarize(m, yo, yr)


def summarize(m, yo, yr):
    return dict(n=len(m), orig=sum(yo.values()), refr=sum(yr.values()),
                gains=sum((not yo[p]) and yr[p] for p in m),
                losses=sum(yo[p] and not yr[p] for p in m))


def fixed(do, dr, rule, thresh=1, anniv=False, drop=()):
    Ko, Kr = build_keys(do, rule), build_keys(dr, rule)
    Po, Pr = pair_index(Ko), pair_index(Kr)
    Do, Dr = debuts(Po), debuts(Pr)
    m = matched_2022(Po, Pr, Do, Dr)
    yo = {p: len(followons(Po[p], *Do[p], anniv=anniv)) >= thresh for p in m}
    yr = {p: len(followons(Pr[p], *Dr[p], drop=drop, anniv=anniv)) >= thresh
          for p in m}
    return summarize(m, yo, yr)


def discordant(d):
    s = d.groupby(["key", "P_IPC"]).size().reset_index()
    s["subs"] = s.P_IPC.apply(lambda t: frozenset(SUBCLASS.findall(t)))
    n = s.groupby("key").subs.nunique()
    return set(n[n > 1].index)


def main(orig, refr, out):
    do, dr = load(orig), load(refr)
    disc = discordant(do) | discordant(dr)
    rows = []
    for rule in (False, True):
        name = "restricted" if rule else "all_links"
        checks = [
            ("Primary", "reference", lambda: fixed(do, dr, rule)),
            ("History from 2016", "rebuilt", lambda: rebuilt(do, dr, rule, start=2016)),
            ("History from 2017", "rebuilt", lambda: rebuilt(do, dr, rule, start=2017)),
            (f"Exclude {len(disc)} IPC-discordant keys", "rebuilt",
             lambda: rebuilt(do, dr, rule, exclude=disc)),
            ("Union of subclasses", "rebuilt", lambda: rebuilt(do, dr, rule, mode="union")),
            ("Application rows only", "rebuilt", lambda: rebuilt(do, dr, rule, mode="apponly")),
            ("Filing-anniversary cutoff", "fixed", lambda: fixed(do, dr, rule, anniv=True)),
            ("Anniversary + omit two keys without front pages", "fixed",
             lambda: fixed(do, dr, rule, anniv=True, drop=NO_FRONT_PAGE)),
            ("Two qualifying keys", "fixed", lambda: fixed(do, dr, rule, thresh=2)),
            ("Three qualifying keys", "fixed", lambda: fixed(do, dr, rule, thresh=3)),
            ("Omit key without corroboration", "fixed",
             lambda: fixed(do, dr, rule, drop=["CN119579061"])),
            ("Omit two keys without front pages", "fixed",
             lambda: fixed(do, dr, rule, drop=NO_FRONT_PAGE)),
        ]
        for label, kind, f in checks:
            rows.append(dict(rule=name, check=label, handling=kind, **f()))
        one = [fixed(do, dr, rule, drop=[k])["refr"] for k in SUPPORT]
        two = [fixed(do, dr, rule, drop=[a, b])["refr"]
               for a, b in combinations(SUPPORT, 2)]
        rows.append(dict(rule=name, check="Omit any one supporting key", handling="fixed",
                         refr_range=f"{min(one)}-{max(one)}"))
        rows.append(dict(rule=name, check="Omit any two supporting keys", handling="fixed",
                         refr_range=f"{min(two)}-{max(two)}"))
    r = pd.DataFrame(rows)
    r.to_csv(out, index=False)
    print(r.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
