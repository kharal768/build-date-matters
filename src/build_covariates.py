"""Build 2021 firm covariates from CNRDS exports (CSV members extracted into folders).

Output columns: scode, log_assets, leverage, rd_int, ind_elec, ind_mach, ind_nonmfg, ind_trans
(reference: other manufacturing), log_inv_all.
Inputs: folders with the balance-sheet, income-statement, company-information and firm
        patent-application exports.
Output: <out_csv> (licensed data: do not redistribute).
Paper:  Appendix E (covariates).
Usage:  python3 -I build_covariates.py <balance_dir> <income_dir> <cbid_dir> <cird_dir> <out_csv>
"""
import glob
import sys

import numpy as np
import pandas as pd


def read(folder):
    files = sorted(glob.glob(f"{folder}/*.csv"))
    return pd.concat([pd.read_csv(f, dtype=str, skiprows=[1], encoding="utf-8-sig")
                      for f in files])


def year_end(d, cols):
    d = d[d.Date == "2021-12-31"].drop_duplicates("Scode").copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.set_index("Scode")


def industry(code):
    code = str(code)
    if not code.startswith("C"):
        return "nonmfg"
    return {"34": "mach", "35": "mach", "38": "elec", "39": "elec", "40": "elec",
            "36": "trans", "37": "trans"}.get(code[1:3], "othmfg")


def main(bal, inc, cbid, cird, out):
    b = year_end(read(bal), ["AT", "LB"])
    b = b[b.AT > 0]
    f = pd.DataFrame({"log_assets": np.log(b.AT), "leverage": b.LB / b.AT})
    i = year_end(read(inc), ["OPREV", "RADEX"])
    i = i[i.OPREV > 0]
    f = f.join((i.RADEX.fillna(0) / i.OPREV).clip(upper=1).rename("rd_int"), how="inner")
    c = read(cbid).set_index("Scode")
    ind = pd.get_dummies(c.IndcodeB.map(industry), prefix="ind", dtype=int)
    f = f.join(ind.drop(columns="ind_othmfg"), how="inner")
    p = read(cird)
    p = p[(p.Ftyp == "集团公司合计") & p.Year.astype(int).between(2019, 2021)]
    inv = sum(pd.to_numeric(p[c], errors="coerce").fillna(0) for c in ("Invia", "Invja"))
    inv = inv.groupby(p.Scode).sum()
    f["log_inv_all"] = np.log1p(inv.reindex(f.index).fillna(0))
    f.index.name = "scode"
    f.reset_index().to_csv(out, index=False)
    print(f.shape)


if __name__ == "__main__":
    main(*sys.argv[1:6])
