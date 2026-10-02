"""Per-company decision thresholds (cost-sensitive, bootstrap-stabilised).

Idea
----
A customer is 'flagged' (gets a retention offer) when the expected gain is positive.
  gain if flagged = (customer really churns) x success_rate x customer_value  -  offer_cost
We search the threshold that gives the highest total gain for each company (group), on
out-of-fold *calibrated* probabilities. To keep the result stable, the search is repeated on
50 bootstrap re-samples and the median threshold is used ("bagged threshold").
The result is shrunk towards the global threshold (weight = churners / (churners + 200)), and companies
with too few customers / churners simply use the global threshold.
"""
import numpy as np
import pandas as pd

DEFAULT_CFG = dict(
    offer_cost=300.0,        # Rs spent on one retention offer (cashback / discount / call)
    success_rate=0.25,       # share of would-be churners that a retention offer saves
    horizon_months=12,       # customer value = monthly revenue x this many months
    default_value=5000.0,    # used when the data has no revenue column
    min_rows=500,            # a company needs at least this many customers ...
    min_churners=100,        # ... and this many churners to get its own threshold
    shrink_k=200,            # own threshold is pulled towards the global one: weight = churners/(churners+shrink_k)
    n_boot=50,
    seed=42,
)
GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)
VALUE_KEYS = ("arpu", "monthly_charge", "monthlycharge", "monthly charge")
ALL = "All customers"


def customer_value(df, cfg):
    """Rs value of keeping each customer = average monthly revenue (months 6-8) x horizon."""
    cols = [c for c in df.columns
            if any(k in c.lower() for k in VALUE_KEYS)
            and pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_9")]
    if not cols:
        return np.full(len(df), float(cfg["default_value"]))
    monthly = df[cols].clip(lower=0).mean(axis=1)
    return (monthly * cfg["horizon_months"]).to_numpy(float)


def profit_curve(y, p, value, cfg, grid=GRID):
    """Total gain for every threshold in `grid`."""
    gain = y * cfg["success_rate"] * value - cfg["offer_cost"]
    order = np.argsort(-p, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(gain[order])])
    k = np.searchsorted(-p[order], -grid, side="right")      # customers with p >= threshold
    return cum[k]


def profit_at(y, flagged, value, cfg):
    return float((flagged * (y * cfg["success_rate"] * value - cfg["offer_cost"])).sum())


def best_threshold(y, p, value, cfg):
    rng = np.random.default_rng(cfg["seed"])
    n, picks = len(y), []
    for _ in range(int(cfg["n_boot"])):
        i = rng.integers(0, n, n)
        picks.append(GRID[int(np.argmax(profit_curve(y[i], p[i], value[i], cfg)))])
    return float(np.median(picks))


def describe(y, p, value, groups, table, glob, cfg):
    """Add flagged / precision / recall / gain columns to the threshold table."""
    thr = thresholds_for(groups, table, glob)
    rows = []
    for g in table["Company"]:
        m = groups == g
        f = p[m] >= thr[m]
        pos = int(y[m].sum())
        caught = int((f & (y[m] == 1)).sum())
        rows.append(dict(
            Flagged=int(f.sum()), Caught=caught,
            Precision=round(caught / max(int(f.sum()), 1) * 100, 1),
            Recall=round(caught / max(pos, 1) * 100, 1),
            Gain_Rs=round(profit_at(y[m], f, value[m], cfg)),
            Gain_Rs_if_global=round(profit_at(y[m], p[m] >= glob, value[m], cfg))))
    return pd.concat([table.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def fit_thresholds(y, p, value, groups, cfg):
    """Returns (table, global_threshold)."""
    y, p, value, groups = map(np.asarray, (y, p, value, groups))
    glob = best_threshold(y, p, value, cfg)
    rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        n, pos = int(m.sum()), int(y[m].sum())
        own = n >= cfg["min_rows"] and pos >= cfg["min_churners"] and len(pd.unique(groups)) > 1
        t = glob
        if own:                                   # shrink towards the global threshold (more churners = more trust)
            w = pos / (pos + cfg["shrink_k"])
            t = round(w * best_threshold(y[m], p[m], value[m], cfg) + (1 - w) * glob, 2)
        rows.append(dict(
            Company=g, Customers=n, Churners=pos, Churn_Rate=round(pos / max(n, 1) * 100, 1),
            Threshold=t, Method="own (shrunk)" if own else "global threshold"))
    table = pd.DataFrame(rows)
    return describe(y, p, value, groups, table, glob, cfg), glob


def thresholds_for(groups, table, glob):
    lookup = dict(zip(table["Company"], table["Threshold"]))
    return np.array([lookup.get(g, glob) for g in groups], dtype=float)


def risk_level(p, thr):
    high_cut = np.minimum(2 * thr, 0.5)
    return np.where(p >= high_cut, "High", np.where(p >= thr, "Medium", "Low"))
