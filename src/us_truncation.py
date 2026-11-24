"""Publication-cutoff simulation on PatentsView pre-grant publications (release of 10 April 2026).

Robotics set: applications with CPC-at-issue subclass B25J. Pairs: unordered CPC subclass pairs.
Debut: first filing year of the pair among robotics applications (from 2005). Label: the pair
appears on another robotics application filed one or two years later. Visible at cutoff T if
published on or before T.
Inputs: folder holding pg_published_application.tsv.zip and pg_cpc_at_issue.tsv.zip.
Outputs: us_robotics_apps.csv, us_truncation.csv and us_q4_share.csv in <pg_folder>/us_analysis/ (the folder must exist).
Paper:  Section 5.6, Figure 1.
Usage:  python3 -I us_truncation.py <pg_folder> [first_cutoff] [frequency]
"""
import sys, zipfile, itertools
import pandas as pd

base = sys.argv[1]
z = zipfile.ZipFile(f"{base}/pg_cpc_at_issue.tsv.zip")
cols = ["pgpub_id", "cpc_subclass"]
parts = []
with z.open(z.namelist()[0]) as fh:
    for ch in pd.read_csv(fh, sep="\t", usecols=cols, dtype=str, chunksize=2_000_000):
        parts.append(ch.drop_duplicates())
cpc = pd.concat(parts).drop_duplicates()
rob = set(cpc.loc[cpc.cpc_subclass == "B25J", "pgpub_id"])
cpc = cpc[cpc.pgpub_id.isin(rob)]
z = zipfile.ZipFile(f"{base}/pg_published_application.tsv.zip")
with z.open(z.namelist()[0]) as fh:
    app = pd.read_csv(fh, sep="\t", usecols=["pgpub_id", "application_id", "filing_date", "published_date"], dtype=str)
app = app[app.pgpub_id.isin(rob)]
app["fd"] = pd.to_datetime(app.filing_date, errors="coerce")
app["pd"] = pd.to_datetime(app.published_date, errors="coerce")
app = app.dropna(subset=["fd", "pd"]).sort_values("pd").drop_duplicates("application_id")
app = app[app.fd.dt.year >= 2005]
subs = cpc.groupby("pgpub_id").cpc_subclass.apply(lambda s: sorted(set(s)))
app = app[app.pgpub_id.isin(subs.index)]
app["subs"] = app.pgpub_id.map(subs)
app["y"] = app.fd.dt.year
app[["pgpub_id", "application_id", "filing_date", "published_date"]].to_csv(f"{base}/us_analysis/us_robotics_apps.csv", index=False)
pairs = {}
for r in app.itertuples():
    for p in itertools.combinations(r.subs, 2):
        pairs.setdefault(p, []).append((r.y, r.pd, r.fd))
q4 = app.groupby("y").apply(lambda g: (g.fd.dt.quarter == 4).mean()).round(3)
rows = []
cutoffs = pd.date_range(sys.argv[2] if len(sys.argv) > 2 else "2022-06-30", "2025-12-31", freq=(sys.argv[3] if len(sys.argv) > 3 else "QE"))
for cohort in range(2015, 2023):
    ref = []
    for p, rr in pairs.items():
        d = min(y for y, _, _ in rr)
        if d != cohort:
            continue
        ref.append([pdte for y, pdte, _ in rr if 1 <= y - d <= 2])
    n = len(ref); pos = sum(bool(f) for f in ref)
    for T in cutoffs:
        vis = sum(any(x <= T for x in f) for f in ref)
        rows.append(dict(cohort=cohort, cutoff=T.date(), pairs=n, ref_positives=pos, visible=vis,
                         share=round(vis / pos, 3) if pos else None))
out = pd.DataFrame(rows)
out.to_csv(f"{base}/us_analysis/us_truncation.csv", index=False)
q4.to_csv(f"{base}/us_analysis/us_q4_share.csv")
print("robotics applications:", len(app))
print("Q4 share by filing year:", q4.to_dict())
print(out.pivot(index="cutoff", columns="cohort", values="share").to_string())
print(out.groupby("cohort")[["pairs", "ref_positives"]].first().to_string())
