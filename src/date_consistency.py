"""Consistency checks on recovered Chinese application publication dates.

1. Weekday distribution (invention applications are published on fixed weekdays).
2. Publication after filing.
3. Application publication on or before the CNRDS grant date, for keys with a grant row.
Inputs: recovered dates (CSV), earlier build (.dta), later build (.xlsx).
Output: <out_csv>.
Paper:  Appendix B.
Usage:  python3 -I date_consistency.py <metadata_csv> <earlier_dta> <later_xlsx> <out_csv>
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recurrence import load  # noqa: E402

meta_csv, orig, refr, out = sys.argv[1:5]
m = pd.read_csv(meta_csv, dtype=str).dropna(subset=["cn_a_publication_date"])
m = m.drop_duplicates("provisional_id")
m["pub"] = pd.to_datetime(m.cn_a_publication_date)
m["fil"] = pd.to_datetime(m.filing_date)
grants = pd.concat([load(orig), load(refr)])
grants = grants[grants.grdt.str.len() > 0]
g = pd.to_datetime(grants.groupby("key").grdt.min())
m["grant"] = m.provisional_id.map(g)
wd = m.pub.dt.day_name().value_counts()
res = {
    "dated_keys": len(m),
    "weekday_tuesday_or_friday_share": round(100 * m.pub.dt.dayofweek.isin([1, 4]).mean(), 2),
    "published_after_filing_share": round(100 * (m.pub > m.fil).mean(), 2),
    "keys_with_cnrds_grant": int(m.grant.notna().sum()),
    "publication_on_or_before_grant_share": round(
        100 * (m.loc[m.grant.notna(), "pub"] <= m.loc[m.grant.notna(), "grant"]).mean(), 2),
}
pd.Series(res).to_csv(out, header=["value"])
print(res)
print(wd.to_dict())
bad = m[(m.grant.notna()) & (m.pub > m.grant)]
print("publication after grant:", bad[["provisional_id", "cn_a_publication_date", "grant"]].to_string())
