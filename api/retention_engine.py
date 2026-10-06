import pandas as pd


def get_risk_level(churn_probability):

    if churn_probability >= 0.70:
        return "High Risk"

    elif churn_probability >= 0.40:
        return "Medium Risk"

    else:
        return "Low Risk"


def get_network_status(rsrp, sinr):

    if rsrp < -100 or sinr < 0:
        return "Poor"

    elif rsrp < -90 or sinr < 13:
        return "Fair"

    else:
        return "Good"


def get_retention_action(
    churn_probability,
    rsrp,
    sinr,
    customer_value
):

    risk = get_risk_level(churn_probability)
    network = get_network_status(rsrp, sinr)

    if risk == "High Risk" and network == "Poor":
        return "Priority Retention + Network Improvement"

    elif risk == "High Risk":
        return "Priority Retention Offer"

    elif risk == "Medium Risk" and network == "Poor":
        return "Network Quality Improvement"

    elif risk == "Medium Risk":
        return "Targeted Retention Offer"

    elif customer_value == "High":
        return "Loyalty Benefit"

    else:
        return "Normal Customer Engagement"


def generate_retention_recommendations(df):

    df = df.copy()

    df["Risk_Level"] = (
        df["Churn_Probability"]
        .apply(get_risk_level)
    )

    df["Network_Status"] = df.apply(
        lambda row: get_network_status(
            row["rsrp"],
            row["sinr"]
        ),
        axis=1
    )

    df["Retention_Action"] = df.apply(
        lambda row: get_retention_action(
            row["Churn_Probability"],
            row["rsrp"],
            row["sinr"],
            row.get("customer_value", "Normal")
        ),
        axis=1
    )

    return df
