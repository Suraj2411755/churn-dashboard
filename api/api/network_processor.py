import pandas as pd


def process_cells(cells):
    """
    Convert OpenCelliD cell records into a clean DataFrame.
    """

    if not cells:
        return pd.DataFrame()

    df = pd.DataFrame(cells)

    required_columns = [
        "mcc",
        "mnc",
        "lac",
        "cellid",
        "lat",
        "lon"
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = None

    df = df[required_columns + [
        col for col in df.columns
        if col not in required_columns
    ]]

    return df


def add_distance_from_location(df, user_lat, user_lon):
    """
    Calculate approximate distance from user location.
    """

    if df.empty:
        return df

    df = df.copy()

    df["distance_km"] = (
        ((df["lat"] - user_lat) ** 2 +
         (df["lon"] - user_lon) ** 2) ** 0.5
    ) * 111

    return df.sort_values("distance_km")


def get_network_summary(df):
    """
    Generate a simple network summary.
    """

    if df.empty:
        return {
            "total_cells": 0,
            "average_distance_km": None
        }

    return {
        "total_cells": len(df),
        "average_distance_km": round(
            df["distance_km"].mean(), 2
        )
    }
