"""Bootstrap over cohort-rule combinations of the CNRDS forecast test, with exact binomial intervals.

2,000 draws, seed 20261124.
Input:  output of cnrds_rule_forecast.py.
Output: <out_csv>.
Paper:  Section 5.8.
Usage:  python3 -I cnrds_rule_forecast_bootstrap.py <cn_rule_forecast_csv> <out_csv>
"""
import sys
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint
rng = np.random.default_rng(20261124)
r = pd.read_csv(sys.argv[1]); units = [g for _, g in r.groupby(["rule", "cohort"])]
def stats(us):
    out = {}
    for m in ("naive", "structural"):
        e = 0
        for g in us:
            a = g.month[g.actual >= 0.9].min(); p = g.month[g[m] >= 0.9].min()
            a = 13 if np.isnan(a) else a; p = 13 if np.isnan(p) else p
            e += p < a
        out[m + "_early"] = e / len(us)
        out[m + "_mae"] = np.mean([(g[m] - g.actual).abs().mean() for g in us])
    out["mae_diff"] = out["naive_mae"] - out["structural_mae"]
    return out
obs = stats(units)
B = pd.DataFrame([stats([units[i] for i in rng.integers(0, len(units), len(units))]) for _ in range(2000)])
res = pd.DataFrame({"observed": obs, "lo": B.quantile(.025), "hi": B.quantile(.975)}).round(3)
res.loc["p_mae_diff_le_0"] = [(B.mae_diff <= 0).mean(), np.nan, np.nan]
for name, k, n in (("naive_early_exactCI", 4, 8), ("structural_early_exactCI", 3, 8),
                   ("structural_early_pooled_US_CN", 4, 26), ("naive_early_pooled_US_CN", 10, 26)):
    lo, hi = proportion_confint(k, n, method="beta"); res.loc[name] = [k / n, round(lo, 3), round(hi, 3)]
res.to_csv(sys.argv[2])
