"""Core logic: cleaning (NaN -> 0), calibrated churn model, per-company thresholds, evaluation."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import thresholds as th

MODEL_PATH = "churn_model.joblib"
POSITIVE = {"yes", "y", "1", "1.0", "true", "churn", "churned", "left"}
NULL_TEXT = {"", "nan", "none", "null", "na", "n/a", "-", "?"}
GROUP_HINTS = ("operator", "partner", "company", "provider", "carrier", "brand", "network_name")
OPERATOR_ALIASES = {
    "jio": "Jio", "reliance jio": "Jio", "reliance jio infocomm": "Jio", "rjio": "Jio",
    "airtel": "Airtel", "bharti airtel": "Airtel", "bharti": "Airtel",
    "vi": "Vi", "vodafone idea": "Vi", "vodafone-idea": "Vi", "vodafone": "Vi", "idea": "Vi",
    "bsnl": "BSNL", "bharat sanchar nigam": "BSNL", "mtnl": "MTNL",
}


# ------------------------------------------------------------------ cleaning
def clean_data(df: pd.DataFrame, drop_rows: bool = True):
    """Fix errors and replace every missing value with 0. Returns (clean_df, report).
    drop_rows=False keeps every row (use it for files you want predictions for)."""
    rep = {"rows_before": len(df), "cols_before": df.shape[1]}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).str.lower().str.startswith("unnamed")]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]        # repeated column names
    df = df.dropna(axis=1, how="all")                          # empty columns

    rep["duplicates_removed"] = 0
    if drop_rows:
        df = df.dropna(axis=0, how="all")
        n = len(df)
        df = df.drop_duplicates()
        rep["duplicates_removed"] = n - len(df)

    # text tidy-up: numbers stored as text -> numbers; junk words -> missing
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        s = df[c].astype(str).str.strip()
        s = s.where(~s.str.lower().isin(NULL_TEXT), np.nan)
        num = pd.to_numeric(s.str.replace(r"[₹$,%\s]", "", regex=True), errors="coerce")
        if s.notna().sum() > 0 and num.notna().sum() >= 0.9 * s.notna().sum():
            df[c] = num
        else:
            df[c] = s

    # infinity -> missing
    numc = df.select_dtypes(include=[np.number]).columns
    rep["inf_replaced"] = int(np.isinf(df[numc].to_numpy(dtype=float)).sum()) if len(numc) else 0
    if len(numc):
        df[numc] = df[numc].replace([np.inf, -np.inf], np.nan)

    # ALL missing values -> 0
    rep["nan_replaced"] = int(df.isna().sum().sum())
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna("0").astype(str)

    df = df.reset_index(drop=True)
    rep["rows_after"], rep["cols_after"] = len(df), df.shape[1]
    return df, rep


# --------------------------------------------------------- column detection
def find_target(df):
    return next((c for c in df.columns if "churn" in c.lower() and "prob" not in c.lower()), None)


def find_id_col(df):
    keys = ("customer", "msisdn", "mobile", "phone", "user_id", "userid", "subscriber")
    for c in df.columns:
        n = c.lower().replace(" ", "_")
        if (any(k in n for k in keys) or n == "id" or n.endswith("_id")) and df[c].nunique() > 0.9 * len(df):
            return c
    return None


def find_group_col(df):
    """Column that holds the telecom company (Jio / Airtel / Vi / BSNL ...), if any."""
    for c in df.columns:
        if any(h in c.lower() for h in GROUP_HINTS) and not pd.api.types.is_numeric_dtype(df[c]) \
                and 1 < df[c].nunique() <= 30:
            return c
    return None


def to_binary(series):
    return series.astype(str).str.strip().str.lower().isin(POSITIVE).astype(int)


def standardise_group(df, group_col):
    """Same company written in different ways (jio / Reliance Jio) -> one name."""
    df = df.copy()
    if group_col and group_col in df.columns:
        def fix(v):
            v = str(v).strip()
            if v.lower() in NULL_TEXT or v == "0":
                return "Unknown"
            return OPERATOR_ALIASES.get(v.lower(), v)
        df[group_col] = df[group_col].map(fix)
    return df


def groups_from(df, group_col):
    if group_col and group_col in df.columns:
        return standardise_group(df, group_col)[group_col].astype(str).to_numpy()
    return np.full(len(df), th.ALL, dtype=object)


# ------------------------------------------------------------------ model
def _pipeline(cat):
    enc = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)],
        remainder="passthrough")
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, class_weight="balanced", random_state=42)
    return Pipeline([("pre", enc), ("clf", clf)])


def _calibrator(raw, y):
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, y)


def train_bundle(df, target, id_col=None, group_col=None, cfg=None, train_mask=None, n_splits=5, seed=42):
    """Train + calibrate + fit per-company thresholds. `df` must already be cleaned."""
    cfg = {**th.DEFAULT_CFG, **(cfg or {})}
    df = standardise_group(df, group_col)
    y_all = to_binary(df[target]).to_numpy()
    mask = np.ones(len(df), bool) if train_mask is None else np.asarray(train_mask, bool)

    X = df.drop(columns=[c for c in (target, id_col) if c and c in df.columns])
    X = X.drop(columns=[c for c in X.columns
                        if not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 50])
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    groups = groups_from(df, group_col)
    value = th.customer_value(df, cfg)
    Xt, yt = X[mask], y_all[mask]

    pipe = _pipeline(cat)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    raw = cross_val_predict(pipe, Xt, yt, cv=skf, method="predict_proba")[:, 1]   # honest, out-of-fold
    cal = _calibrator(raw, yt)
    prob = cal.predict(raw)
    pipe.fit(Xt, yt)

    table, glob = th.fit_thresholds(yt, prob, value[mask], groups[mask], cfg)

    fpr, tpr, _ = roc_curve(yt, raw)
    q = pd.qcut(pd.Series(raw).rank(method="first"), 10, labels=False)
    reliab = pd.DataFrame({"bin": q, "raw": raw, "calibrated": prob, "actual": yt}).groupby("bin").mean()
    metrics = dict(auc=roc_auc_score(yt, raw), pr_auc=average_precision_score(yt, raw),
                   brier_raw=brier_score_loss(yt, raw), brier_calibrated=brier_score_loss(yt, prob),
                   churn_rate=float(yt.mean()), n_train=int(len(yt)),
                   fpr=fpr, tpr=tpr, reliability=reliab.reset_index())

    idx = np.random.default_rng(seed).choice(len(yt), min(2000, len(yt)), replace=False)
    imp = permutation_importance(pipe, Xt.iloc[idx], yt[idx], scoring="roc_auc", n_repeats=2,
                                 random_state=seed, n_jobs=1)
    importance = pd.Series(imp.importances_mean, index=Xt.columns).sort_values(ascending=False)

    return {"model": pipe, "calibrator": cal, "features": list(X.columns), "cat": cat, "num": num,
            "target": target, "id_col": id_col, "group_col": group_col, "cfg": cfg,
            "thr_table": table, "thr_global": glob, "metrics": metrics, "importance": importance,
            "oof": {"y": yt, "raw": raw, "prob": prob, "group": groups[mask], "value": value[mask]}}


def save_bundle(bundle, path=MODEL_PATH):
    joblib.dump(bundle, path)


def load_bundle(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception:
        return None


# ------------------------------------------------------------- prediction
def _align(bundle, df):
    X = df.copy()
    for c in bundle["features"]:
        if c not in X.columns:
            X[c] = "0" if c in bundle["cat"] else 0
    X = X[bundle["features"]].copy()
    for c in bundle["cat"]:
        X[c] = X[c].astype(str)
    for c in bundle["num"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def predict_proba(bundle, df):
    """Calibrated churn probability (0-1) for every row."""
    df = standardise_group(df, bundle["group_col"])
    raw = bundle["model"].predict_proba(_align(bundle, df))[:, 1]
    return bundle["calibrator"].predict(raw)


def decide(bundle, df, prob, table=None, glob=None):
    """Apply each company's own threshold -> Yes/No answer + risk level."""
    table = bundle["thr_table"] if table is None else table
    glob = bundle["thr_global"] if glob is None else glob
    groups = groups_from(df, bundle["group_col"])
    thr = th.thresholds_for(groups, table, glob)
    value = th.customer_value(df, bundle["cfg"])
    return pd.DataFrame({
        "Company": groups,
        "Churn_Probability": (prob * 100).round(1),
        "Threshold_Used": (thr * 100).round(1),
        "Predicted_Churn": np.where(prob >= thr, "Yes", "No"),
        "Risk_Level": th.risk_level(prob, thr),
        "Customer_Value_Rs": value.round(0),
    }, index=df.index)


def honest_evaluation(bundle, cfg=None, repeats=10, seed=42):
    """Tune thresholds on one random half of the customers, judge them on the other half.
    Repeated `repeats` times with different splits and averaged (single splits are too noisy)."""
    cfg = {**bundle["cfg"], **(cfg or {})}
    o = bundle["oof"]
    y, raw, g = o["y"], o["raw"], o["group"]
    v = o["value"] * cfg["horizon_months"] / bundle["cfg"]["horizon_months"]
    strat = pd.Series(g).astype(str) + "_" + pd.Series(y).astype(str)
    acc = {}
    for r in range(repeats):
        try:
            it, ie = train_test_split(np.arange(len(y)), test_size=0.5, random_state=seed + r, stratify=strat)
        except ValueError:
            it, ie = train_test_split(np.arange(len(y)), test_size=0.5, random_state=seed + r)
        cal = _calibrator(raw[it], y[it])
        pt, pe = cal.predict(raw[it]), cal.predict(raw[ie])
        table, glob = th.fit_thresholds(y[it], pt, v[it], g[it], cfg)
        plans = {"Fixed 50% cut-off": np.full(len(ie), 0.5),
                 "One global threshold": np.full(len(ie), glob),
                 "Per-company thresholds": th.thresholds_for(g[ie], table, glob)}
        for name, thr in plans.items():
            f = pe >= thr
            acc.setdefault(name, []).append((int(f.sum()), int((f & (y[ie] == 1)).sum()),
                                             int(y[ie].sum()), th.profit_at(y[ie], f, v[ie], cfg)))
    rows = []
    for name, vals in acc.items():
        a = np.array(vals, float)
        rows.append({"Method": name, "Customers_flagged": round(a[:, 0].mean()),
                     "Churners_caught": round(a[:, 1].mean()), "Churners_total": round(a[:, 2].mean()),
                     "Gain_Rs": round(a[:, 3].mean()), "Gain_Rs_std": round(a[:, 3].std())})
    return pd.DataFrame(rows)
