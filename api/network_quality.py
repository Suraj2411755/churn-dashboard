import pandas as pd


def classify_rsrp(rsrp):
    if pd.isna(rsrp):
        return "Unknown"
    elif rsrp >= -80:
        return "Excellent"
    elif rsrp >= -90:
        return "Good"
    elif rsrp >= -100:
        return "Fair"
    else:
        return "Poor"


def classify_rsrq(rsrq):
    if pd.isna(rsrq):
        return "Unknown"
    elif rsrq >= -10:
        return "Excellent"
    elif rsrq >= -15:
        return "Good"
    elif rsrq >= -20:
        return "Fair"
    else:
        return "Poor"


def classify_sinr(sinr):
    if pd.isna(sinr):
        return "Unknown"
    elif sinr >= 20:
        return "Excellent"
    elif sinr >= 13:
        return "Good"
    elif sinr >= 0:
        return "Fair"
    else:
        return "Poor"


def add_network_quality_labels(df):

    df = df.copy()

    if "RSRP" in df.columns:
        df["RSRP_Quality"] = df["RSRP"].apply(
            classify_rsrp
        )

    if "RSRQ" in df.columns:
        df["RSRQ_Quality"] = df["RSRQ"].apply(
            classify_rsrq
        )

    if "SINR" in df.columns:
        df["SINR_Quality"] = df["SINR"].apply(
            classify_sinr
        )

    return df
