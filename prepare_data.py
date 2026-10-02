"""Turns the raw India/SEA telecom file (columns ending _6, _7, _8, _9) into training-ready data.

1. high-value customers (avg recharge of months 6-7 above the 70th percentile) = training rows
2. Churn label: no calls in/out AND no mobile data in month 9  -> Yes
3. every month-9 column is dropped (otherwise the model would see the answer = data leakage)
4. date columns and columns that never change are dropped
"""
import numpy as np
import pandas as pd

CHURN_BASIS = ["total_ic_mou_9", "total_og_mou_9", "vol_2g_mb_9", "vol_3g_mb_9"]


def is_raw_india_file(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in CHURN_BASIS) and not any("churn" in c.lower() for c in df.columns)


def _col(df, name):
    return df[name].fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)


def _avg_recharge(df):
    total = sum(_col(df, f"total_rech_amt_{m}") + _col(df, f"total_rech_data_{m}") * _col(df, f"av_rech_amt_data_{m}")
                for m in (6, 7))
    return total / 2


def prepare_india_all(df: pd.DataFrame, quantile: float = 0.70):
    """All customers, with a Churn column. Returns (df, train_mask, report)."""
    df = df.copy()
    rep = {"rows_raw": len(df)}
    avg = _avg_recharge(df)
    train_mask = (avg >= avg.quantile(quantile)).to_numpy()
    rep["rows_high_value"] = int(train_mask.sum())

    no_use = df[CHURN_BASIS].fillna(0).sum(axis=1) == 0          # missing = no usage
    df["Churn"] = np.where(no_use, "Yes", "No")
    rep["churn_rate"] = float((df.loc[train_mask, "Churn"] == "Yes").mean())

    hv = df[train_mask]
    leak = [c for c in df.columns if c.endswith("_9") or c.startswith("sep_")]
    dates = [c for c in df.columns if "date" in c.lower()]
    const = [c for c in df.columns if c != "Churn" and hv[c].nunique(dropna=True) <= 1]
    drop = set(leak + dates + const)
    rep["cols_dropped"] = len(drop)
    return df.drop(columns=drop).reset_index(drop=True), train_mask, rep


def prepare_india(df: pd.DataFrame, quantile: float = 0.70):
    """High-value customers only (used for training in the dashboard)."""
    full, mask, rep = prepare_india_all(df, quantile)
    return full[mask].reset_index(drop=True), rep


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "telecom_churn_data.csv"
    out, r = prepare_india(pd.read_csv(src))
    out.to_csv("india_telecom_prepared.csv", index=False)
    print(r, out.shape)
