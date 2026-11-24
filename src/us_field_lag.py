"""Publication timing across technology fields in the PatentsView pre-grant data.

For applications filed in 2022-2023 (the follow-up window of the 2021 cohort): share published by
31 December 2023, by CPC section and for robotics (B25J), and the fourth-quarter filing share.
Input:  folder holding pg_published_application.tsv.zip and pg_cpc_at_issue.tsv.zip.
Output: <out_csv>.
Paper:  Section 5.6.
Usage:  python3 -I us_field_lag.py <pg_folder> <out_csv>
"""
import sys
import zipfile

import pandas as pd

base, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(f"{base}/pg_published_application.tsv.zip")
with z.open(z.namelist()[0]) as fh:
    app = pd.read_csv(fh, sep="\t", dtype=str,
                      usecols=["pgpub_id", "application_id", "filing_date", "published_date"])
app["fd"] = pd.to_datetime(app.filing_date, errors="coerce")
app["pd"] = pd.to_datetime(app.published_date, errors="coerce")
app = app.dropna(subset=["fd", "pd"]).sort_values("pd").drop_duplicates("application_id")
app = app[app.fd.dt.year.isin([2022, 2023])]
ids = set(app.pgpub_id)
z = zipfile.ZipFile(f"{base}/pg_cpc_at_issue.tsv.zip")
parts = []
with z.open(z.namelist()[0]) as fh:
    for ch in pd.read_csv(fh, sep="\t", dtype=str, usecols=["pgpub_id", "cpc_section", "cpc_subclass"],
                          chunksize=2_000_000):
        parts.append(ch[ch.pgpub_id.isin(ids)].drop_duplicates())
cpc = pd.concat(parts)
cut = pd.Timestamp("2023-12-31")
rows = []
groups = [(s, set(cpc.loc[cpc.cpc_section == s, "pgpub_id"])) for s in sorted(cpc.cpc_section.dropna().unique())]
groups.append(("B25J robotics", set(cpc.loc[cpc.cpc_subclass == "B25J", "pgpub_id"])))
groups.append(("All", ids))
for name, g in groups:
    a = app[app.pgpub_id.isin(g)]
    rows.append(dict(field=name, applications=len(a),
                     published_by_2023_end=round(100 * (a.pd <= cut).mean(), 1),
                     q4_filing_share=round(100 * (a.fd.dt.quarter == 4).mean(), 1)))
r = pd.DataFrame(rows)
r.to_csv(out, index=False)
print(r.to_string())
