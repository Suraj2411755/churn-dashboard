import hashlib
import urllib.parse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import churn_core as cc
import prepare_data as pdp
import thresholds as th


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Indian Telecom Churn AI",
    page_icon="📡",
    layout="wide"
)


# =========================================================
# LOGIN
# =========================================================

_PW_HASH = hashlib.sha256(
    st.secrets.get("APP_PASSWORD", "churn2026").encode()
).hexdigest()


def _check_login():

    if st.session_state.get("authed"):
        return True

    st.markdown(
        "## 🔒 Indian Telecom Churn Predictor — Login"
    )

    st.caption(
        "Enter the shared password given to your class/team."
    )

    pw = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Log in", type="primary"):

        if hashlib.sha256(pw.encode()).hexdigest() == _PW_HASH:

            st.session_state.authed = True
            st.rerun()

        else:

            st.error("Incorrect password.")

    st.stop()


_check_login()


# =========================================================
# SIDEBAR LOGOUT
# =========================================================

with st.sidebar:

    if st.button("🚪 Log out"):

        st.session_state.authed = False
        st.rerun()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .hero {
        background: linear-gradient(
            120deg,
            #4f46e5,
            #7c3aed 55%,
            #db2777
        );

        padding: 28px 34px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        margin: 6px 0 0;
        opacity: .9;
    }

    .kpi {
        background: white;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,.08);
        border-left: 6px solid var(--c);
    }

    .kpi .v {
        font-size: 1.7rem;
        font-weight: 700;
        color: #111827;
    }

    .kpi .l {
        color: #6b7280;
        font-size: .85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <h1>📡 Indian Telecom Churn & Retention AI</h1>

        <p>
        Customer churn prediction → churn insights →
        retention recommendations → network intelligence →
        operator comparison
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CONSTANTS
# =========================================================

COLORS = {
    "Low": "#10b981",
    "Medium": "#f59e0b",
    "High": "#ef4444"
}

NONE = "(none - one threshold for everyone)"


# =========================================================
# KPI FUNCTION
# =========================================================

def kpi(col, label, value, color):

    col.markdown(
        f"""
        <div class="kpi" style="--c:{color}">
            <div class="v">{value}</div>
            <div class="l">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# THRESHOLD VIEW
# =========================================================

def threshold_view(table, key=""):

    show = table.copy()

    show["Threshold"] = (
        show["Threshold"] * 100
    ).round(0)

    show = show.rename(
        columns={
            "Threshold": "Threshold_%",
            "Churn_Rate": "Churn_Rate_%"
        }
    )

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        table,
        x="Company",
        y=table["Threshold"] * 100,
        text=(table["Threshold"] * 100).round(0),
        color="Company",
        title="Churn threshold used for each company (%)"
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title="Threshold %"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="thr_bar" + key
    )


# =========================================================
# MODEL QUALITY
# =========================================================

def show_model(bundle):

    m = bundle["metrics"]

    c = st.columns(4)

    kpi(
        c[0],
        "ROC-AUC",
        f"{m['auc']:.3f}",
        "#4f46e5"
    )

    kpi(
        c[1],
        "PR-AUC",
        f"{m['pr_auc']:.3f}",
        "#10b981"
    )

    kpi(
        c[2],
        "Brier Score",
        f"{m['brier_raw']:.3f} → {m['brier_calibrated']:.3f}",
        "#f59e0b"
    )

    kpi(
        c[3],
        "Training Churn Rate",
        f"{m['churn_rate']:.1%}",
        "#db2777"
    )

    st.caption(
        "Metrics are based on out-of-fold predictions. "
        "Lower Brier score indicates better probability calibration."
    )

    g1, g2, g3 = st.columns(3)

    # Feature importance

    imp = bundle["importance"].head(10)[::-1]

    fig_imp = px.bar(
        x=imp.values,
        y=imp.index,
        orientation="h",
        title="Top factors associated with churn",
        color=imp.values,
        color_continuous_scale="Purples"
    )

    fig_imp.update_layout(
        coloraxis_showscale=False,
        xaxis_title="AUC drop when shuffled",
        yaxis_title=""
    )

    g1.plotly_chart(
        fig_imp,
        use_container_width=True
    )

    # ROC

    roc = go.Figure(
        go.Scatter(
            x=m["fpr"],
            y=m["tpr"],
            fill="tozeroy",
            line=dict(color="#7c3aed")
        )
    )

    roc.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            dash="dash",
            color="gray"
        )
    )

    roc.update_layout(
        title="ROC curve",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate"
    )

    g2.plotly_chart(
        roc,
        use_container_width=True
    )

    # Reliability

    r = m["reliability"]

    rel = go.Figure()

    rel.add_trace(
        go.Scatter(
            x=r["raw"],
            y=r["actual"],
            mode="lines+markers",
            name="Before calibration"
        )
    )

    rel.add_trace(
        go.Scatter(
            x=r["calibrated"],
            y=r["actual"],
            mode="lines+markers",
            name="After calibration"
        )
    )

    rel.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            dash="dash",
            color="gray"
        )
    )

    rel.update_layout(
        title="Are probabilities trustworthy?",
        xaxis_title="Predicted churn probability",
        yaxis_title="Actual churn rate"
    )

    g3.plotly_chart(
        rel,
        use_container_width=True
    )


# =========================================================
# LOAD MODEL
# =========================================================

if "bundle" not in st.session_state:

    st.session_state.bundle = cc.load_bundle()


# =========================================================
# SIDEBAR MENU
# =========================================================

page = st.sidebar.radio(
    "Menu",
    [
        "1️⃣ Train model",
        "2️⃣ Company thresholds",
        "3️⃣ Predict churn",
        "4️⃣ Churn insights",
        "5️⃣ Network intelligence",
        "6️⃣ Tower lookup",
        "7️⃣ Market trends",
        "ℹ️ Help"
    ]
)


# =========================================================
# TRAI DATA
# =========================================================

TRAI_DATA = {

    "month": "March 2026",

    "source":
        "TRAI Telecom Subscription Data, March 2026",

    "rows": [

        {
            "Company": "Jio",
            "Subscribers (million)": 481.2,
            "Monthly Change": "+2.1M"
        },

        {
            "Company": "Airtel",
            "Subscribers (million)": 412.8,
            "Monthly Change": "+5.1M"
        },

        {
            "Company": "Vi (Vodafone Idea)",
            "Subscribers (million)": 196.4,
            "Monthly Change": "-0.16M"
        },

        {
            "Company": "BSNL",
            "Subscribers (million)": 92.3,
            "Monthly Change": "+0.4M"
        }

    ]
}


# =========================================================
# CIRCLE CENTROIDS
# =========================================================

CIRCLE_CENTROIDS = {

    "Delhi": (28.6139, 77.2090),

    "Mumbai": (19.0760, 72.8777),

    "Maharashtra": (19.7515, 75.7139),

    "Karnataka": (15.3173, 75.7139),

    "Tamil Nadu": (11.1271, 78.6569),

    "Kerala": (10.8505, 76.2711),

    "Andhra Pradesh": (15.9129, 79.7400),

    "Gujarat": (22.2587, 71.1924),

    "Rajasthan": (27.0238, 74.2179),

    "Punjab": (31.1471, 75.3412),

    "UP West": (28.3670, 79.4304),

    "UP East": (26.8467, 80.9462),

    "West Bengal": (22.9868, 87.8550),

    "Madhya Pradesh": (22.9734, 78.6569),

    "Bihar": (25.0961, 85.3131),

    "Assam": (26.2006, 92.9376),

    "Himachal Pradesh": (31.1048, 77.1734),

    "Orissa": (20.9517, 85.0985)
}


st.sidebar.caption(
    "Model status: "
    + (
        "✅ ready"
        if st.session_state.bundle
        else "❌ not trained yet"
    )
)


# =========================================================
# 1. TRAIN MODEL
# =========================================================

if page.startswith("1"):

    st.subheader(
        "Train the model with your historical data"
    )

    st.write(
        "Upload a CSV that shows who churned "
        "(a column such as `Churn`), or the raw "
        "Indian/SEA telecom file."
    )

    up = st.file_uploader(
        "Training CSV",
        type="csv",
        key="train"
    )

    if up:

        raw = pd.read_csv(
            up,
            low_memory=False
        )

        if pdp.is_raw_india_file(raw):

            raw = (
                raw
                .dropna(how="all")
                .drop_duplicates()
                .reset_index(drop=True)
            )

            raw, prep = pdp.prepare_india(raw)

            st.info(
                f"Indian/SEA telecom file detected and prepared "
                f"automatically: kept {prep['rows_high_value']:,} "
                f"high-value customers, created the Churn label "
                f"({prep['churn_rate']:.1%} churned) and removed "
                f"month-9 columns to avoid data leakage."
            )

        df, rep = cc.clean_data(raw)

        c = st.columns(4)

        kpi(
            c[0],
            "Rows (raw → clean)",
            f"{rep['rows_before']:,} → {rep['rows_after']:,}",
            "#4f46e5"
        )

        kpi(
            c[1],
            "Duplicates removed",
            rep["duplicates_removed"],
            "#7c3aed"
        )

        kpi(
            c[2],
            "Missing values set to 0",
            f"{rep['nan_replaced']:,}",
            "#db2777"
        )

        kpi(
            c[3],
            "Columns kept",
            rep["cols_after"],
            "#0ea5e9"
        )

        with st.expander("Preview cleaned data"):

            st.dataframe(
                df.head(50),
                use_container_width=True
            )

        cols = list(df.columns)

        tg = cc.find_target(df)
        ig = cc.find_id_col(df)
        gg = cc.find_group_col(df)

        a, b_, c_ = st.columns(3)

        target = a.selectbox(
            "Churn column",
            cols,
            index=cols.index(tg)
            if tg else 0
        )

        id_opts = ["(none)"] + cols

        id_col = b_.selectbox(
            "Customer ID column",
            id_opts,
            index=id_opts.index(ig)
            if ig else 0
        )

        grp_opts = [NONE] + cols

        group_col = c_.selectbox(
            "Telecom company column",
            grp_opts,
            index=grp_opts.index(gg)
            if gg else 0
        )

        id_col = (
            None
            if id_col == "(none)"
            else id_col
        )

        group_col = (
            None
            if group_col == NONE
            else group_col
        )

        if group_col is None:

            st.warning(
                "No company column selected → the same threshold "
                "will be used for every customer. Add a column "
                "with Jio / Airtel / Vi / BSNL to get a threshold "
                "for each company."
            )

        st.markdown(
            "**Business settings** "
            "(used to choose the thresholds)"
        )

        s1, s2, s3 = st.columns(3)

        offer = s1.number_input(
            "Cost of one retention offer (₹)",
            0.0,
            100000.0,
            th.DEFAULT_CFG["offer_cost"],
            50.0
        )

        succ = s2.slider(
            "Offer success rate (%)",
            5,
            95,
            int(
                th.DEFAULT_CFG["success_rate"] * 100
            )
        ) / 100

        hor = s3.slider(
            "Customer value horizon (months)",
            1,
            36,
            th.DEFAULT_CFG["horizon_months"]
        )

        if st.button(
            "🚀 Train model",
            type="primary"
        ):

            with st.spinner(
                "Training with 5-fold validation..."
            ):

                bundle = cc.train_bundle(
                    df,
                    target,
                    id_col,
                    group_col,
                    dict(
                        offer_cost=offer,
                        success_rate=succ,
                        horizon_months=hor
                    )
                )

                cc.save_bundle(bundle)

                st.session_state.bundle = bundle

            st.success(
                "Model trained and saved."
            )

        if st.session_state.bundle:

            st.markdown(
                "### Model quality"
            )

            show_model(
                st.session_state.bundle
            )


# =========================================================
# 2. COMPANY THRESHOLDS
# =========================================================

elif page.startswith("2"):

    bundle = st.session_state.bundle

    if not bundle:

        st.warning(
            "Please train the model first."
        )

        st.stop()

    st.subheader(
        "Churn threshold for each telecom company"
    )

    st.write(
        "A customer is flagged when the expected gain "
        "of a retention offer is positive:"
    )

    st.code(
        "chance of churn × offer success rate × "
        "customer value − offer cost"
    )

    cfg0 = bundle["cfg"]

    c1, c2, c3 = st.columns(3)

    offer = c1.number_input(
        "Cost of one retention offer (₹)",
        0.0,
        100000.0,
        float(cfg0["offer_cost"]),
        50.0
    )

    succ = c2.slider(
        "Offer success rate (%)",
        5,
        95,
        int(
            round(
                cfg0["success_rate"] * 100
            )
        )
    ) / 100

    hor = c3.slider(
        "Customer value horizon (months)",
        1,
        36,
        int(cfg0["horizon_months"])
    )

    cfg = {
        **cfg0,
        "offer_cost": offer,
        "success_rate": succ,
        "horizon_months": hor,
        "n_boot": 30
    }

    o = bundle["oof"]

    value = (
        o["value"]
        * hor
        / cfg0["horizon_months"]
    )

    with st.spinner(
        "Searching best thresholds..."
    ):

        table, glob = th.fit_thresholds(
            o["y"],
            o["prob"],
            value,
            o["group"],
            cfg
        )

    if len(table) == 1:

        st.info(
            "The training data has no company column, "
            "so there is one threshold for all customers."
        )

    threshold_view(table)

    grid = th.GRID

    fig = go.Figure()

    for _, row in table.iterrows():

        m = (
            o["group"]
            == row["Company"]
        )

        fig.add_trace(
            go.Scatter(
                x=grid * 100,
                y=th.profit_curve(
                    o["y"][m],
                    o["prob"][m],
                    value[m],
                    cfg
                ),
                name=row["Company"],
                mode="lines"
            )
        )

    fig.update_layout(
        title="Total gain at every threshold",
        xaxis_title="Threshold %",
        yaxis_title="Gain ₹"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    b1, b2 = st.columns(2)

    if b1.button(
        "✅ Use these thresholds for predictions and save"
    ):

        bundle["thr_table"] = table
        bundle["thr_global"] = glob

        bundle["cfg"] = {
            **cfg,
            "n_boot": cfg0["n_boot"]
        }

        cc.save_bundle(bundle)

        st.success(
            "Thresholds saved."
        )

    if b2.button(
        "🧪 Honest test"
    ):

        with st.spinner(
            "Running evaluation..."
        ):

            ev = cc.honest_evaluation(
                bundle,
                cfg
            )

        st.dataframe(
            ev,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            ev,
            x="Method",
            y="Gain_Rs",
            error_y="Gain_Rs_std",
            color="Method",
            title="Average gain on customers not used for tuning"
        )

        fig.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# 3. PREDICT CHURN
# =========================================================

elif page.startswith("3"):

    bundle = st.session_state.bundle

    if not bundle:

        st.warning(
            "Please train the model first."
        )

        st.stop()

    st.subheader(
        "Upload customers to predict churn"
    )

    up = st.file_uploader(
        "Customer CSV",
        type="csv",
        key="pred"
    )

    if up:

        raw = pd.read_csv(
            up,
            low_memory=False
        )

        df, rep = cc.clean_data(
            raw,
            drop_rows=False
        )

        missing = [
            c
            for c in bundle["features"]
            if c not in df.columns
        ]

        if missing:

            st.warning(
                f"{len(missing)} of "
                f"{len(bundle['features'])} training "
                f"columns are missing. They will be treated as 0."
            )

        gc = bundle["group_col"]

        if gc and gc not in df.columns:

            st.warning(
                f"Company column '{gc}' not found. "
                "Global threshold will be used."
            )

        prob = cc.predict_proba(
            bundle,
            df
        )

        res = cc.decide(
            bundle,
            df,
            prob
        )

        idc = (
            bundle["id_col"]
            if bundle["id_col"] in df.columns
            else cc.find_id_col(df)
        )

        if idc:

            out = pd.concat(
                [
                    df[[idc]],
                    res
                ],
                axis=1
            )

        else:

            out = res.copy()

        out = out.sort_values(
            "Churn_Probability",
            ascending=False
        )

        flagged = (
            out["Predicted_Churn"]
            == "Yes"
        )

        at_risk = float(
            (
                out.loc[
                    flagged,
                    "Churn_Probability"
                ] / 100
                *
                out.loc[
                    flagged,
                    "Customer_Value_Rs"
                ]
            ).sum()
        )

        c = st.columns(4)

        kpi(
            c[0],
            "Customers analysed",
            f"{len(out):,}",
            "#4f46e5"
        )

        kpi(
            c[1],
            "Likely to churn",
            f"{int(flagged.sum()):,}",
            "#ef4444"
        )

        kpi(
            c[2],
            "Flagged share",
            f"{flagged.mean():.1%}",
            "#f59e0b"
        )

        kpi(
            c[3],
            "Expected revenue at risk",
            f"₹{at_risk:,.0f}",
            "#db2777"
        )

        st.session_state["prediction_df"] = df
        st.session_state["prediction_result"] = res
        st.session_state["prediction_output"] = out

        st.success(
            "Churn prediction completed."
        )

        g1, g2 = st.columns(2)

        fig1 = px.pie(
            out,
            names="Risk_Level",
            hole=.55,
            title="Customers by risk level",
            color="Risk_Level",
            color_discrete_map=COLORS,
            category_orders={
                "Risk_Level": [
                    "Low",
                    "Medium",
                    "High"
                ]
            }
        )

        g1.plotly_chart(
            fig1,
            use_container_width=True
        )

        multi = (
            "Company" in out.columns
            and out["Company"].nunique() > 1
        )

        fig2 = px.histogram(
            out,
            x="Churn_Probability",
            nbins=25,
            color="Company"
            if multi else None,
            title="Churn probability distribution"
        )

        g2.plotly_chart(
            fig2,
            use_container_width=True
        )

        if multi:

            grp = (
                out
                .assign(
                    f=flagged.astype(int)
                )
                .groupby("Company")
                .agg(
                    Customers=("f", "size"),
                    Flagged_pct=("f", "mean"),
                    Threshold=(
                        "Threshold_Used",
                        "first"
                    )
                )
                .reset_index()
            )

            grp["Flagged_pct"] = (
                grp["Flagged_pct"] * 100
            ).round(1)

            fig3 = px.bar(
                grp,
                x="Company",
                y="Flagged_pct",
                text="Flagged_pct",
                color="Company",
                title="Share of customers flagged by company (%)"
            )

            fig3.update_layout(
                showlegend=False,
                yaxis_title="Flagged %"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        # -------------------------------------------------
        # CATEGORY INSIGHT
        # -------------------------------------------------

        both = df.join(
            res.drop(
                columns=["Company"],
                errors="ignore"
            )
        )

        cat_cols = [
            c
            for c in df.columns
            if (
                not pd.api.types.is_numeric_dtype(
                    df[c]
                )
                and df[c].nunique() <= 15
            )
        ]

        if cat_cols:

            st.subheader(
                "Customer segment churn analysis"
            )

            by = st.selectbox(
                "Compare flagged customers by",
                cat_cols
            )

            g = (
                both
                .assign(
                    f=(
                        both["Predicted_Churn"]
                        == "Yes"
                    ).astype(int)
                )
                .groupby(by)["f"]
                .mean()
                .mul(100)
                .round(1)
                .reset_index()
                .sort_values(
                    "f",
                    ascending=False
                )
            )

            fig4 = px.bar(
                g,
                x=by,
                y="f",
                text="f",
                title=f"Flagged customers by {by} (%)"
            )

            fig4.update_layout(
                yaxis_title="Flagged %"
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )

        # -------------------------------------------------
        # CUSTOMER TABLE
        # -------------------------------------------------

        st.subheader(
            "🚨 Customers most likely to churn"
        )

        show = st.radio(
            "Show",
            [
                "Will churn (flagged)",
                "High risk only",
                "Everyone"
            ],
            horizontal=True
        )

        if show.startswith("Will"):

            view = out[flagged]

        elif show.startswith("High"):

            view = out[
                out["Risk_Level"]
                == "High"
            ]

        else:

            view = out

        st.dataframe(
            view,
            use_container_width=True,
            height=380,
            column_config={
                "Churn_Probability":
                    st.column_config.ProgressColumn(
                        "Churn %",
                        min_value=0,
                        max_value=100,
                        format="%.1f"
                    )
            }
        )

        st.download_button(
            "⬇️ Download predictions",
            out.to_csv(
                index=False
            ).encode(),
            "churn_predictions.csv",
            "text/csv"
        )

        # -------------------------------------------------
        # CUSTOMER EXPLANATION
        # -------------------------------------------------

        if (
            "explain_stats" in bundle
            and flagged.any()
        ):

            st.subheader(
                "🔍 Why is this customer at risk?"
            )

            if idc:

                flagged_ids = (
                    out.loc[
                        flagged,
                        idc
                    ]
                )

                pick = st.selectbox(
                    "Pick a flagged customer",
                    flagged_ids.astype(
                        str
                    ).tolist()
                )

                row_idx = flagged_ids[
                    flagged_ids.astype(
                        str
                    ) == pick
                ].index[0]

            else:

                flagged_indices = (
                    out.loc[
                        flagged
                    ].index
                )

                pick = st.selectbox(
                    "Pick a flagged customer",
                    [
                        str(x)
                        for x in flagged_indices
                    ]
                )

                row_idx = int(pick)

            reasons = cc.explain_customer(
                bundle,
                df.loc[row_idx],
                top_n=4
            )

            prob_pct = out.loc[
                row_idx,
                "Churn_Probability"
            ]

            st.info(
                f"**{prob_pct:.1f}% predicted "
                "churn probability.**"
            )

            st.markdown(
                "**Main factors increasing this customer's risk:**"
            )

            for reason in reasons:

                st.markdown(
                    f"- {reason}"
                )


# =========================================================
# 4. CHURN INSIGHTS
# =========================================================

elif page.startswith("4"):

    st.subheader(
        "📊 Churn Insights"
    )

    if "prediction_output" not in st.session_state:

        st.info(
            "First go to **3️⃣ Predict churn** "
            "and upload a customer CSV."
        )

        st.stop()

    out = st.session_state[
        "prediction_output"
    ]

    df = st.session_state[
        "prediction_df"
    ]

    res = st.session_state[
        "prediction_result"
    ]

    flagged = (
        out["Predicted_Churn"]
        == "Yes"
    )

    total_customers = len(out)

    high_risk = (
        out["Risk_Level"]
        == "High"
    ).sum()

    medium_risk = (
        out["Risk_Level"]
        == "Medium"
    ).sum()

    low_risk = (
        out["Risk_Level"]
        == "Low"
    ).sum()

    average_risk = (
        out["Churn_Probability"]
        .mean()
    )

    churn_share = (
        flagged.mean() * 100
        if total_customers
        else 0
    )

    # ---------------------------------------------
    # KPI
    # ---------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    kpi(
        c1,
        "Total customers",
        f"{total_customers:,}",
        "#4f46e5"
    )

    kpi(
        c2,
        "High-risk customers",
        f"{high_risk:,}",
        "#ef4444"
    )

    kpi(
        c3,
        "Average churn risk",
        f"{average_risk:.1f}%",
        "#f59e0b"
    )

    kpi(
        c4,
        "Customers flagged",
        f"{churn_share:.1f}%",
        "#db2777"
    )

    # ---------------------------------------------
    # RISK DISTRIBUTION
    # ---------------------------------------------

    st.markdown(
        "### Risk distribution"
    )

    risk_df = pd.DataFrame(
        {
            "Risk Level": [
                "High",
                "Medium",
                "Low"
            ],
            "Customers": [
                high_risk,
                medium_risk,
                low_risk
            ]
        }
    )

    g1, g2 = st.columns(2)

    fig_risk = px.bar(
        risk_df,
        x="Risk Level",
        y="Customers",
        color="Risk Level",
        color_discrete_map=COLORS,
        title="Customer risk distribution"
    )

    g1.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    fig_pie = px.pie(
        risk_df,
        names="Risk Level",
        values="Customers",
        hole=.55,
        color="Risk Level",
        color_discrete_map=COLORS,
        title="Risk composition"
    )

    g2.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # ---------------------------------------------
    # AUTOMATIC BUSINESS INSIGHT
    # ---------------------------------------------

    st.markdown(
        "### 🔎 Key business insights"
    )

    if churn_share >= 20:

        st.error(
            f"High attention required: "
            f"{churn_share:.1f}% of customers are "
            "currently flagged for churn."
        )

    elif churn_share >= 10:

        st.warning(
            f"Moderate attention required: "
            f"{churn_share:.1f}% of customers are "
            "currently flagged for churn."
        )

    else:

        st.success(
            f"Current predicted churn exposure is "
            f"{churn_share:.1f}% of customers."
        )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### Customer risk status"
        )

        st.write(
            f"🔴 **High risk:** {high_risk:,} customers"
        )

        st.write(
            f"🟠 **Medium risk:** {medium_risk:,} customers"
        )

        st.write(
            f"🟢 **Low risk:** {low_risk:,} customers"
        )

    with col2:

        st.markdown(
            "#### Recommended priority"
        )

        st.write(
            "1. Contact high-risk customers first."
        )

        st.write(
            "2. Monitor medium-risk customers."
        )

        st.write(
            "3. Maintain engagement with low-risk customers."
        )

    # ---------------------------------------------
    # COMPANY INSIGHTS
    # ---------------------------------------------

    if "Company" in out.columns:

        company_count = (
            out["Company"]
            .nunique()
        )

        if company_count > 1:

            st.markdown(
                "### 🏢 Churn risk by operator"
            )

            company_insights = (
                out
                .assign(
                    Flagged=flagged.astype(int)
                )
                .groupby("Company")
                .agg(
                    Customers=(
                        "Flagged",
                        "size"
                    ),
                    High_Risk=(
                        "Risk_Level",
                        lambda x:
                        (x == "High").sum()
                    ),
                    Average_Risk=(
                        "Churn_Probability",
                        "mean"
                    ),
                    Flagged_Rate=(
                        "Flagged",
                        "mean"
                    )
                )
                .reset_index()
            )

            company_insights[
                "Average_Risk"
            ] = (
                company_insights[
                    "Average_Risk"
                ].round(1)
            )

            company_insights[
                "Flagged_Rate"
            ] = (
                company_insights[
                    "Flagged_Rate"
                ] * 100
            ).round(1)

            st.dataframe(
                company_insights,
                use_container_width=True,
                hide_index=True
            )

            fig_company = px.bar(
                company_insights,
                x="Company",
                y="Flagged_Rate",
                text="Flagged_Rate",
                color="Company",
                title="Predicted churn exposure by operator (%)"
            )

            fig_company.update_layout(
                yaxis_title="Flagged customers (%)"
            )

            st.plotly_chart(
                fig_company,
                use_container_width=True
            )

    # ---------------------------------------------
    # TOP FEATURES
    # ---------------------------------------------

    bundle = st.session_state.bundle

    if bundle and "importance" in bundle:

        st.markdown(
            "### 🧠 Main factors associated with churn"
        )

        importance = (
            bundle["importance"]
            .head(10)
            .reset_index()
        )

        importance.columns = [
            "Feature",
            "Importance"
        ]

        st.dataframe(
            importance,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are model-derived feature importance "
            "measures. They indicate predictive association, "
            "not proof that a feature causes churn."
        )

    # ---------------------------------------------
    # RETENTION ACTION
    # ---------------------------------------------

    st.markdown(
        "### 🎯 Retention action"
    )

    if high_risk > 0:

        st.error(
            f"Priority action: focus retention campaigns "
            f"on {high_risk:,} high-risk customers."
        )

        st.write(
            "Recommended actions:"
        )

        st.write(
            "• Prioritize high-risk customers."
        )

        st.write(
            "• Investigate service and network-quality problems."
        )

        st.write(
            "• Review declining customer usage."
        )

        st.write(
            "• Consider customer-value-based retention offers."
        )

    else:

        st.success(
            "No customers are currently classified as high risk."
        )


# =========================================================
# 5. NETWORK INTELLIGENCE
# =========================================================

elif page.startswith("5"):

    st.subheader(
        "📡 Network Intelligence"
    )

    st.write(
        "Use OpenCelliD to retrieve nearby cellular "
        "cell records for an authorized test location."
    )

    try:

        from api.opencellid_api import (
            get_cells_in_area
        )

        from api.network_processor import (
            process_cells,
            add_distance_from_location
        )

        api_available = True

    except Exception as e:

        api_available = False

        st.error(
            "OpenCelliD modules are not available yet."
        )

        st.code(
            "api/opencellid_api.py\n"
            "api/network_processor.py"
        )

    if api_available:

        c1, c2 = st.columns(2)

        lat = c1.number_input(
            "Latitude",
            value=18.5204,
            format="%.6f"
        )

        lon = c2.number_input(
            "Longitude",
            value=73.8567,
            format="%.6f"
        )

        if st.button(
            "🔎 Find Nearby Cellular Cells",
            type="primary"
        ):

            try:

                with st.spinner(
                    "Fetching cellular data..."
                ):

                    cells = get_cells_in_area(
                        lat,
                        lon,
                        radius=0.01
                    )

                    network_df = process_cells(
                        cells
                    )

                    network_df = (
                        add_distance_from_location(
                            network_df,
                            lat,
                            lon
                        )
                    )

                if network_df.empty:

                    st.warning(
                        "No cellular cells were found "
                        "for this location."
                    )

                else:

                    st.success(
                        f"{len(network_df)} "
                        "cellular cells found."
                    )

                    display_columns = [
                        "mcc",
                        "mnc",
                        "lac",
                        "cellid",
                        "lat",
                        "lon",
                        "distance_km"
                    ]

                    available_columns = [
                        c
                        for c in display_columns
                        if c in network_df.columns
                    ]

                    st.dataframe(
                        network_df[
                            available_columns
                        ],
                        use_container_width=True
                    )

                    nearest = (
                        network_df.iloc[0]
                    )

                    st.markdown(
                        "### Nearest cellular cell"
                    )

                    n1, n2, n3, n4 = st.columns(4)

                    kpi(
                        n1,
                        "Distance",
                        f"{nearest['distance_km']:.2f} km",
                        "#4f46e5"
                    )

                    kpi(
                        n2,
                        "Cell ID",
                        str(nearest["cellid"]),
                        "#10b981"
                    )

                    kpi(
                        n3,
                        "MCC",
                        str(nearest["mcc"]),
                        "#f59e0b"
                    )

                    kpi(
                        n4,
                        "MNC",
                        str(nearest["mnc"]),
                        "#db2777"
                    )

                    st.info(
                        "OpenCelliD records represent cellular "
                        "cell information. They should not "
                        "automatically be interpreted as confirmed "
                        "physical tower locations or live signal "
                        "measurements."
                    )

            except Exception as e:

                st.error(
                    f"Network data error: {e}"
                )


# =========================================================
# 6. TOWER LOOKUP
# =========================================================

elif page.startswith("6"):

    st.subheader(
        "📍 Mobile tower lookup"
    )

    st.write(
        "Use the official government portal to "
        "check available telecom infrastructure."
    )

    c1, c2 = st.columns(2)

    place = c1.text_input(
        "Place name",
        placeholder="e.g. Shivaji Nagar, Pune"
    )

    company = c2.selectbox(
        "Company",
        [
            "Jio",
            "Airtel",
            "Vi (Vodafone Idea)",
            "BSNL",
            "MTNL",
            "Other"
        ]
    )

    if st.button(
        "🔎 Open tower information",
        type="primary"
    ):

        if not place.strip():

            st.warning(
                "Enter a place name first."
            )

        else:

            st.success(
                f"Search the Tarang Sanchar map for "
                f"**{place}** and filter by **{company}**."
            )

            st.link_button(
                "🌐 Open Tarang Sanchar",
                "https://www.tarangsanchar.gov.in/",
                use_container_width=True
            )

            maps_q = urllib.parse.quote(
                place
            )

            st.link_button(
                "🗺️ Open place on Google Maps",
                f"https://www.google.com/maps/search/?api=1&query={maps_q}",
                use_container_width=True
            )


# =========================================================
# 7. MARKET TRENDS
# =========================================================

elif page.startswith("7"):

    st.subheader(
        "📈 Market trends — TRAI subscriber data"
    )

    st.info(
        "The figures below are stored in the application "
        "as a demonstration dataset. Update them with "
        "verified official TRAI figures before presenting "
        "them as current market statistics."
    )

    trai_df = pd.DataFrame(
        TRAI_DATA["rows"]
    )

    c1, c2 = st.columns([3, 2])

    fig = px.bar(
        trai_df,
        x="Company",
        y="Subscribers (million)",
        color="Company",
        text="Subscribers (million)",
        title=(
            f"Wireless subscribers — "
            f"{TRAI_DATA['month']}"
        )
    )

    fig.update_layout(
        showlegend=False
    )

    c1.plotly_chart(
        fig,
        use_container_width=True
    )

    c2.dataframe(
        trai_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Example circle-wise visualization"
    )

    demo_map = pd.DataFrame(
        [
            {
                "Circle": circle,
                "lat": lat,
                "lon": lon,
                "Sample value":
                    (hash(circle) % 50) + 50
            }
            for circle, (lat, lon)
            in CIRCLE_CENTROIDS.items()
        ]
    )

    fig_map = px.scatter_geo(
        demo_map,
        lat="lat",
        lon="lon",
        size="Sample value",
        color="Sample value",
        hover_name="Circle",
        scope="asia",
        title=(
            "Illustrative circle-wise map"
        )
    )

    fig_map.update_geos(
        center={
            "lat": 22,
            "lon": 80
        },
        projection_scale=4,
        showcountries=True
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

    st.caption(
        "The map values above are illustrative and "
        "must not be presented as actual operator "
        "subscriber/churn measurements."
    )


# =========================================================
# HELP
# =========================================================

else:

    st.markdown(
        """
        ## How to use

        ### 1. Train model

        Upload historical telecom customer data and train
        the churn model.

        ### 2. Company thresholds

        Optimize retention thresholds based on customer
        value and offer economics.

        ### 3. Predict churn

        Upload customer data to obtain:

        - Churn probability
        - Risk level
        - Predicted churn
        - Revenue at risk
        - Customer-level explanation

        ### 4. Churn insights

        Analyze:

        - High-risk customers
        - Medium-risk customers
        - Low-risk customers
        - Churn exposure
        - Operator-wise risk
        - Important churn factors
        - Retention priorities

        ### 5. Network intelligence

        Use OpenCelliD to inspect nearby cellular-cell
        records.

        ### 6. Tower lookup

        Use the official Tarang Sanchar portal for
        infrastructure information.

        ### 7. Market trends

        Compare operator subscriber information using
        verified TRAI data.

        ---

        ## Important data note

        OpenCelliD does not provide a live measurement of
        a user's RSRP, RSRQ or SINR.

        Those measurements should come from authorized
        device/test measurements.

        Also, OpenCelliD cellular-cell records should not
        automatically be described as confirmed physical
        towers.
        """
    )
