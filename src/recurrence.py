"""Patent keys, IPC subclass pairs, debut years and two-year recurrence labels for one CNRDS build.

Shared module: the other CNRDS scripts import load() and keys() from here. Run on its own, it
prints the pipeline counts.
Input:  one build of the CNRDS listed-company industrial robot patent table (.dta or .xlsx).
Output: counts printed to stdout.
Paper:  Table 1 (pipeline counts); definitions in Section 4.1 and Appendix A.
Usage:  python3 -I recurrence.py <build_file>
"""
import re
import sys
from itertools import combinations

import pandas as pd

SUBCLASS = re.compile(r"[A-HY]\d{2}[A-Z]")
KEY = re.compile(r"^(CN\d+)")
RESTRICTED = {"上市公司本身", "子公司"}


def load(path):
    if str(path).endswith(".dta"):
        d = pd.read_stata(path).astype(str).replace({"nan": "", "NaT": ""})
        for c in ("apdt", "grdt"):
            d[c] = d[c].str[:10]
    else:
        d = pd.read_excel(path, dtype=str).fillna("")
    d = d[d["Scode"].str.fullmatch(r"\d{6}")]
    d = d[d["P_Type"].str.strip().str.startswith("发明")].copy()
    d["key"] = d["Pateanno"].str.extract(KEY, expand=False)
    d["kind_rank"] = (d["P_Type"].str.strip() != "发明公布").astype(int)  # application (A) publication before grant (B)
    d["year"] = pd.to_datetime(d["apdt"]).dt.year
    return d


def keys(d, restricted):
    out = {}
    for k, g in d.groupby("key"):
        first = g.sort_values(["kind_rank", "Scode", "P_IPC"]).iloc[0]
        subs = frozenset(SUBCLASS.findall(first["P_IPC"]))
        links = g if not restricted else g[g["Ctlprtrlat"].isin(RESTRICTED)]
        codes = frozenset(links["Scode"])
        if not codes:
            continue
        out[k] = (subs, codes, int(g["year"].iloc[0]))
    return out


def labels(K, h):
    pairs = {}
    for k, (I, S, y) in K.items():
        for p in combinations(sorted(I), 2):
            pairs.setdefault(p, []).append((k, S, y))
    res = {}
    for p, rows in pairs.items():
        d = min(y for _, _, y in rows)
        if not 2018 <= d <= 2022:
            continue
        D = frozenset().union(*[S for _, S, y in rows if y == d])
        F = [k for k, S, y in rows if 1 <= y - d <= h and not (S & D)]
        res[p] = (d, int(len(F) >= 1))
    return res


if __name__ == "__main__":
    d = load(sys.argv[1])
    print("invention rows", len(d))
    for rule in (False, True):
        K = keys(d, rule)
        name = "restricted" if rule else "all links"
        L1, L2 = labels(K, 1), labels(K, 2)
        print(f"{name}: keys {len(K)}, eligible {len(L2)}, "
              f"1y pos {sum(v for _, v in L1.values())}, "
              f"2y pos {sum(v for _, v in L2.values())}")
