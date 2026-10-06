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

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")

# ------------------------------------------------------------- login gate
# Shared password for the whole class/team. Set your own in
# .streamlit/secrets.toml as:  APP_PASSWORD = "your-password-here"
# Falls back to "churn2026" only when no secret is configured (local testing).
_DEFAULT_PW_HASH = hashlib.sha256("churn2026".encode()).hexdigest()
_PW_HASH = hashlib.sha256(st.secrets.get("APP_PASSWORD", "churn2026").encode()).hexdigest()


def _check_login():
    if st.session_state.get("authed"):
        return True
    st.markdown("## 🔒 Indian Telecom Churn Predictor — Login")
    st.caption("Enter the shared password given to your class/team.")
    pw = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        if hashlib.sha256(pw.encode()).hexdigest() == _PW_HASH:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


_check_login()

with st.sidebar:
    if st.button("🚪 Log out"):
        st.session_state.authed = False
        st.rerun()

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#4f46e5,#7c3aed 55%,#db2777);padding:28px 34px;
      border-radius:18px;color:#fff;margin-bottom:22px}
.hero h1{margin:0;font-size:2rem} .hero p{margin:6px 0 0;opacity:.9}
.kpi{background:#fff;border-radius:16px;padding:18px 20px;box-shadow:0 4px 18px rgba(0,0,0,.08);
     border-left:6px solid var(--c);}
.kpi .v{font-size:1.7rem;font-weight:700;color:#111827} .kpi .l{color:#6b7280;font-size:.85rem}
</style>
<div class="hero"><h1>📡 Indian Telecom Churn Predictor</h1>
<p>Upload a CSV → automatic cleaning (missing = 0) → churn risk with a separate threshold for each telecom company.</p></div>
""", unsafe_allow_html=True)

COLORS = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
NONE = "(none - one threshold for everyone)"


def kpi(col, label, value, color):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="v">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


def threshold_view(table, key=""):
    show = table.copy()
    show["Threshold"] = (show["Threshold"] * 100).round(0)
    show = show.rename(columns={"Threshold": "Threshold_%", "Churn_Rate": "Churn_Rate_%"})
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(table, x="Company", y=table["Threshold"] * 100, text=(table["Threshold"] * 100).round(0),
                           color="Company", title="Churn threshold used for each company (%)")
                    .update_layout(showlegend=False, yaxis_title="Threshold %"),
                    use_container_width=True, key="thr_bar" + key)


def show_model(b):
    m = b["metrics"]
    c = st.columns(4)
    kpi(c[0], "ROC-AUC (out-of-fold)", f"{m['auc']:.3f}", "#4f46e5")
    kpi(c[1], "PR-AUC", f"{m['pr_auc']:.3f}", "#10b981")
    kpi(c[2], "Probability error (Brier)", f"{m['brier_raw']:.3f} → {m['brier_calibrated']:.3f}", "#f59e0b")
    kpi(c[3], "Churn rate in training data", f"{m['churn_rate']:.1%}", "#db2777")
    st.caption("Every number comes from predictions on customers the model had not seen (5-fold cross-validation). "
               "Brier: lower is better; the second value is after calibration.")
    g1, g2, g3 = st.columns(3)
    imp = b["importance"].head(10)[::-1]
    g1.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Top factors driving churn",
                           color=imp.values, color_continuous_scale="Purples")
                    .update_layout(coloraxis_showscale=False, xaxis_title="AUC drop when shuffled", yaxis_title=""),
                    use_container_width=True)
    roc = go.Figure(go.Scatter(x=m["fpr"], y=m["tpr"], fill="tozeroy", line=dict(color="#7c3aed")))
    roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    g2.plotly_chart(roc, use_container_width=True)
    r = m["reliability"]
    rel = go.Figure()
    rel.add_trace(go.Scatter(x=r["raw"], y=r["actual"], mode="lines+markers", name="before calibration"))
    rel.add_trace(go.Scatter(x=r["calibrated"], y=r["actual"], mode="lines+markers", name="after calibration"))
    rel.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    rel.update_layout(title="Are probabilities trustworthy?", xaxis_title="Predicted churn probability",
                      yaxis_title="Actual churn rate")
    g3.plotly_chart(rel, use_container_width=True)


if "bundle" not in st.session_state:
    st.session_state.bundle = cc.load_bundle()

page = st.sidebar.radio("Menu", ["1️⃣ Train model", "2️⃣ Company thresholds", "3️⃣ Predict churn",
                                 "4️⃣ Tower lookup", "5️⃣ Market trends", "ℹ️ Help"])

# TRAI official monthly subscriber data (wireless, India, millions). Update this table each month
# from https://www.trai.gov.in/release-publication/reports/telecom-subscriptions-reports — there is
# no live API, TRAI publishes a new PDF/report roughly 4 weeks after each month ends.
TRAI_DATA = {"month": "March 2026", "source": "TRAI Telecom Subscription Data, March 2026",
            "rows": [
                {"Company": "Jio", "Subscribers (million)": 481.2, "Monthly Change": "+2.1M"},
                {"Company": "Airtel", "Subscribers (million)": 412.8, "Monthly Change": "+5.1M"},
                {"Company": "Vi (Vodafone Idea)", "Subscribers (million)": 196.4, "Monthly Change": "-0.16M"},
                {"Company": "BSNL", "Subscribers (million)": 92.3, "Monthly Change": "+0.4M"},
            ]}
# Approximate centroid of each telecom circle (LSA), for demo mapping only — replace with your own
# circle-wise data once you have it (e.g. from TRAI's circle-wise tables).
CIRCLE_CENTROIDS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777), "Maharashtra": (19.7515, 75.7139),
    "Karnataka": (15.3173, 75.7139), "Tamil Nadu": (11.1271, 78.6569), "Kerala": (10.8505, 76.2711),
    "Andhra Pradesh": (15.9129, 79.7400), "Gujarat": (22.2587, 71.1924), "Rajasthan": (27.0238, 74.2179),
    "Punjab": (31.1471, 75.3412), "UP West": (28.3670, 79.4304), "UP East": (26.8467, 80.9462),
    "West Bengal": (22.9868, 87.8550), "Madhya Pradesh": (22.9734, 78.6569), "Bihar": (25.0961, 85.3131),
    "Assam": (26.2006, 92.9376), "Himachal Pradesh": (31.1048, 77.1734), "Orissa": (20.9517, 85.0985),
}
st.sidebar.caption("Model status: " + ("✅ ready" if st.session_state.bundle else "❌ not trained yet"))

# =============================================================== TRAIN
if page.startswith("1"):
    st.subheader("Train the model with your historical data")
    st.write("Upload a CSV that shows who churned (a column such as `Churn`), or the raw Indian/SEA telecom file.")
    up = st.file_uploader("Training CSV", type="csv", key="train")
    if up:
        raw = pd.read_csv(up, low_memory=False)
        if pdp.is_raw_india_file(raw):
            raw = raw.dropna(how="all").drop_duplicates().reset_index(drop=True)
            raw, prep = pdp.prepare_india(raw)
            st.info(f"Indian/SEA telecom file detected and prepared automatically: kept {prep['rows_high_value']:,} "
                    f"high-value customers, created the Churn label ({prep['churn_rate']:.1%} churned) and removed "
                    f"month-9 columns to avoid data leakage.")
        df, rep = cc.clean_data(raw)

        c = st.columns(4)
        kpi(c[0], "Rows (raw → clean)", f"{rep['rows_before']:,} → {rep['rows_after']:,}", "#4f46e5")
        kpi(c[1], "Duplicates removed", rep["duplicates_removed"], "#7c3aed")
        kpi(c[2], "Missing values set to 0", f"{rep['nan_replaced']:,}", "#db2777")
        kpi(c[3], "Columns kept", rep["cols_after"], "#0ea5e9")
        with st.expander("Preview cleaned data"):
            st.dataframe(df.head(50), use_container_width=True)

        cols = list(df.columns)
        tg, ig, gg = cc.find_target(df), cc.find_id_col(df), cc.find_group_col(df)
        a, b_, c_ = st.columns(3)
        target = a.selectbox("Churn column", cols, index=cols.index(tg) if tg else 0)
        id_opts = ["(none)"] + cols
        id_col = b_.selectbox("Customer ID column", id_opts, index=id_opts.index(ig) if ig else 0)
        grp_opts = [NONE] + cols
        group_col = c_.selectbox("Telecom company column", grp_opts, index=grp_opts.index(gg) if gg else 0)
        id_col = None if id_col == "(none)" else id_col
        group_col = None if group_col == NONE else group_col
        if group_col is None:
            st.warning("No company column selected → the same threshold will be used for every customer. "
                       "Add a column with Jio / Airtel / Vi / BSNL to get a threshold for each company.")

        st.markdown("**Business settings** (used to choose the thresholds)")
        s1, s2, s3 = st.columns(3)
        offer = s1.number_input("Cost of one retention offer (₹)", 0.0, 100000.0, th.DEFAULT_CFG["offer_cost"], 50.0)
        succ = s2.slider("Offer success rate (%)", 5, 95, int(th.DEFAULT_CFG["success_rate"] * 100)) / 100
        hor = s3.slider("Customer value horizon (months)", 1, 36, th.DEFAULT_CFG["horizon_months"])

        if st.button("🚀 Train model", type="primary"):
            with st.spinner("Training with 5-fold validation - can take a few minutes on big files..."):
                bundle = cc.train_bundle(df, target, id_col, group_col,
                                         dict(offer_cost=offer, success_rate=succ, horizon_months=hor))
                cc.save_bundle(bundle)
                st.session_state.bundle = bundle
            st.success("Model trained and saved. See the 'Company thresholds' page.")

    if st.session_state.bundle:
        st.markdown("### Model quality")
        show_model(st.session_state.bundle)

# ====================================================== THRESHOLDS
elif page.startswith("2"):
    bundle = st.session_state.bundle
    if not bundle:
        st.warning("Please train the model first (menu → 1️⃣ Train model).")
        st.stop()
    st.subheader("Churn threshold for each telecom company")
    st.write("A customer is flagged when the expected gain of a retention offer is positive: "
             "`chance of churn × offer success rate × customer value − offer cost`. "
             "The best cut-off is searched for every company on out-of-fold calibrated probabilities.")
    cfg0 = bundle["cfg"]
    c1, c2, c3 = st.columns(3)
    offer = c1.number_input("Cost of one retention offer (₹)", 0.0, 100000.0, float(cfg0["offer_cost"]), 50.0)
    succ = c2.slider("Offer success rate (%)", 5, 95, int(round(cfg0["success_rate"] * 100))) / 100
    hor = c3.slider("Customer value horizon (months)", 1, 36, int(cfg0["horizon_months"]))
    cfg = {**cfg0, "offer_cost": offer, "success_rate": succ, "horizon_months": hor, "n_boot": 30}

    o = bundle["oof"]
    value = o["value"] * hor / cfg0["horizon_months"]
    with st.spinner("Searching best thresholds..."):
        table, glob = th.fit_thresholds(o["y"], o["prob"], value, o["group"], cfg)
    if len(table) == 1:
        st.info("The training data has no company column, so there is one threshold for all customers.")
    threshold_view(table)

    grid = th.GRID
    fig = go.Figure()
    for _, row in table.iterrows():
        m = o["group"] == row["Company"]
        fig.add_trace(go.Scatter(x=grid * 100, y=th.profit_curve(o["y"][m], o["prob"][m], value[m], cfg),
                                 name=row["Company"], mode="lines"))
    fig.update_layout(title="Total gain (₹) at every threshold - the peak is the best cut-off",
                      xaxis_title="Threshold %", yaxis_title="Gain ₹")
    st.plotly_chart(fig, use_container_width=True)

    b1, b2 = st.columns(2)
    if b1.button("✅ Use these thresholds for predictions and save"):
        bundle["thr_table"], bundle["thr_global"], bundle["cfg"] = table, glob, {**cfg, "n_boot": cfg0["n_boot"]}
        cc.save_bundle(bundle)
        st.success("Saved.")
    if b2.button("🧪 Honest test (tune on half, judge on the other half)"):
        with st.spinner("Running 10 random splits..."):
            ev = cc.honest_evaluation(bundle, cfg)
        st.dataframe(ev, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(ev, x="Method", y="Gain_Rs", error_y="Gain_Rs_std", color="Method",
                               title="Average gain (₹) on customers not used for tuning")
                        .update_layout(showlegend=False), use_container_width=True)
        st.caption("If 'Per-company thresholds' is not clearly higher than 'One global threshold', "
                   "company thresholds add little for this data.")

# ============================================================ PREDICT
elif page.startswith("3"):
    bundle = st.session_state.bundle
    if not bundle:
        st.warning("Please train the model first (menu → 1️⃣ Train model).")
        st.stop()
    st.subheader("Upload customers to predict churn")
    up = st.file_uploader("Customer CSV (a churn column is not needed)", type="csv", key="pred")
    if up:
        raw = pd.read_csv(up, low_memory=False)
        df, rep = cc.clean_data(raw, drop_rows=False)          # keeps every row
        missing = [c for c in bundle["features"] if c not in df.columns]
        if missing:
            st.warning(f"{len(missing)} of {len(bundle['features'])} columns used in training are missing in this file "
                       f"(treated as 0). Predictions may be weaker. Example: {', '.join(missing[:5])}")
        gc = bundle["group_col"]
        if gc and gc not in df.columns:
            st.warning(f"Company column '{gc}' not found → the global threshold is used for everyone.")

        prob = cc.predict_proba(bundle, df)
        res = cc.decide(bundle, df, prob)
        idc = bundle["id_col"] if bundle["id_col"] in df.columns else cc.find_id_col(df)
        out = pd.concat([df[[idc]] if idc else pd.DataFrame(index=df.index), res], axis=1) \
            .sort_values("Churn_Probability", ascending=False)
        # NOTE: index is intentionally kept aligned with df's index (not reset) so a row in `out`
        # can always be mapped back to the matching row in `df` — used by the explain-a-customer feature.

        flagged = out["Predicted_Churn"] == "Yes"
        at_risk = float((out.loc[flagged, "Churn_Probability"] / 100 * out.loc[flagged, "Customer_Value_Rs"]).sum())
        c = st.columns(4)
        kpi(c[0], "Customers analysed", f"{len(out):,}", "#4f46e5")
        kpi(c[1], "Likely to churn", f"{int(flagged.sum()):,}", "#ef4444")
        kpi(c[2], "Flagged share", f"{flagged.mean():.1%}", "#f59e0b")
        kpi(c[3], "Expected revenue at risk", f"₹{at_risk:,.0f}", "#db2777")
        st.write("")

        g1, g2 = st.columns(2)
        g1.plotly_chart(px.pie(out, names="Risk_Level", hole=.55, title="Customers by risk level",
                               color="Risk_Level", color_discrete_map=COLORS,
                               category_orders={"Risk_Level": ["Low", "Medium", "High"]}),
                        use_container_width=True)
        multi = out["Company"].nunique() > 1
        g2.plotly_chart(px.histogram(out, x="Churn_Probability", nbins=25, color="Company" if multi else None,
                                     title="Churn probability distribution",
                                     color_discrete_sequence=px.colors.qualitative.Bold), use_container_width=True)

        if multi:
            grp = (out.assign(f=flagged.astype(int)).groupby("Company")
                   .agg(Customers=("f", "size"), Flagged_pct=("f", "mean"), Threshold=("Threshold_Used", "first"))
                   .reset_index())
            grp["Flagged_pct"] = (grp["Flagged_pct"] * 100).round(1)
            st.plotly_chart(px.bar(grp, x="Company", y="Flagged_pct", text="Flagged_pct", color="Company",
                                   title="Share of customers flagged, by company (%)")
                            .update_layout(showlegend=False), use_container_width=True)

        both = df.join(res.drop(columns=["Company"]))
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 15]
        if cat_cols:
            by = st.selectbox("Compare flagged customers by…", cat_cols)
            g = (both.assign(f=(both["Predicted_Churn"] == "Yes").astype(int)).groupby(by)["f"].mean()
                 .mul(100).round(1).reset_index().sort_values("f", ascending=False))
            st.plotly_chart(px.bar(g, x=by, y="f", text="f", color="f", color_continuous_scale="RdYlGn_r",
                                   title=f"Flagged % by {by}").update_layout(coloraxis_showscale=False,
                                                                             yaxis_title="Flagged %"),
                            use_container_width=True)

        st.markdown("### 🚨 Customers most likely to churn")
        show = st.radio("Show", ["Will churn (flagged)", "High risk only", "Everyone"], horizontal=True)
        view = out[flagged] if show.startswith("Will") else out[out["Risk_Level"] == "High"] \
            if show.startswith("High") else out
        st.dataframe(view, use_container_width=True, height=380,
                     column_config={"Churn_Probability": st.column_config.ProgressColumn(
                         "Churn %", min_value=0, max_value=100, format="%.1f")})
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                           "churn_predictions.csv", "text/csv")

        if "explain_stats" in bundle and flagged.any():
            st.markdown("### 🔍 Why is this customer at risk?")
            flagged_ids = out.loc[flagged, idc] if idc else pd.Series(out.loc[flagged].index.astype(str),
                                                                       index=out.loc[flagged].index)
            pick = st.selectbox("Pick a flagged customer to explain", flagged_ids.astype(str).tolist())
            row_idx = flagged_ids[flagged_ids.astype(str) == pick].index[0]   # label in both out and df
            reasons = cc.explain_customer(bundle, df.loc[row_idx], top_n=4)
            prob_pct = out.loc[row_idx, "Churn_Probability"]
            st.info(f"**{prob_pct:.1f}% predicted churn probability.** Top factors pushing this risk up "
                   f"(compared with a typical retained customer):")
            for r in reasons:
                st.markdown(f"- {r}")
            st.caption("These are the model's most important features, weighted by how far this "
                      "customer's own values differ from a typical retained customer — not a full "
                      "SHAP explanation, but the same idea in a lighter, deployment-safe form.")

# ========================================================= TOWER LOOKUP
elif page.startswith("4"):
    st.subheader("Mobile tower lookup")
    st.write("There is no public live API for telecom tower locations in India — operators don't publish "
             "one, and this isn't data this app can fetch on its own. The only official source is the "
             "Government of India's **Tarang Sanchar** portal, which has 2.36 million base stations mapped "
             "for all operators, but only through its own map interface, not through an API.")

    c1, c2 = st.columns(2)
    place = c1.text_input("Place name", placeholder="e.g. Shivaji Nagar, Pune")
    company = c2.selectbox("Company", ["Jio", "Airtel", "Vi (Vodafone Idea)", "BSNL", "MTNL", "Other"])

    if st.button("🔎 Open Tarang Sanchar for this search", type="primary"):
        if not place.strip():
            st.warning("Enter a place name first.")
        else:
            st.success(f"Opening Tarang Sanchar — search for **{place}** and filter by **{company}** "
                       f"once the map loads.")
            st.link_button("🌐 Open Tarang Sanchar (official government portal)",
                           "https://www.tarangsanchar.gov.in/", use_container_width=True)
            maps_q = urllib.parse.quote(place)
            st.link_button("🗺️ Open this place on Google Maps (for reference)",
                           f"https://www.google.com/maps/search/?api=1&query={maps_q}",
                           use_container_width=True)

    st.caption("On Tarang Sanchar: use its map search for your place name, then filter the results by "
              "telecom service provider to see that company's towers nearby.")

# ========================================================= MARKET TRENDS
elif page.startswith("5"):
    st.subheader("Market trends — official TRAI subscriber data")
    st.write("There is no live API for this either — the numbers below are from TRAI's official monthly "
             f"report (**{TRAI_DATA['source']}**). TRAI publishes a new report about 4 weeks after each "
             "month ends, so update this table by hand each month from "
             "[trai.gov.in](https://www.trai.gov.in/release-publication/reports/telecom-subscriptions-reports).")

    trai_df = pd.DataFrame(TRAI_DATA["rows"])
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(px.bar(trai_df, x="Company", y="Subscribers (million)", color="Company",
                           text="Subscribers (million)", title=f"Wireless subscribers — {TRAI_DATA['month']}")
                    .update_layout(showlegend=False), use_container_width=True)
    c2.dataframe(trai_df, use_container_width=True, hide_index=True, height=200)

    st.markdown("### Example: putting this on a map")
    st.write("Your churn dataset's `circle_id` column has only one value, so there's no real per-circle "
             "breakdown to map yet. Below is a **working example** using approximate circle centroids and "
             "illustrative numbers — once you have real circle-wise data (subscriber counts, tower counts, "
             "or churn rates per circle), swap the numbers in and the map updates the same way.")

    demo_map = pd.DataFrame([
        {"Circle": c, "lat": lat, "lon": lon, "Sample value": (hash(c) % 50) + 50}
        for c, (lat, lon) in CIRCLE_CENTROIDS.items()
    ])
    fig = px.scatter_geo(demo_map, lat="lat", lon="lon", size="Sample value", color="Sample value",
                         hover_name="Circle", color_continuous_scale="Purples",
                         scope="asia", title="Illustrative circle-wise map (replace Sample value with real data)")
    fig.update_geos(center={"lat": 22, "lon": 80}, projection_scale=4, showcountries=True)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Code used for this map"):
        st.code(
'''import plotly.express as px

# your_df needs: a place/circle name, lat, lon, and the value you want to show
fig = px.scatter_geo(your_df, lat="lat", lon="lon", size="value", color="value",
                     hover_name="circle", scope="asia", color_continuous_scale="Purples")
fig.update_geos(center={"lat": 22, "lon": 80}, projection_scale=4, showcountries=True)
st.plotly_chart(fig, use_container_width=True)''', language="python")

# =============================================================== HELP
else:
    st.markdown("""
### How to use
1. **Train model** - upload the historical file (with a churn column, or the raw Indian/SEA telecom file).
2. **Company thresholds** - see and adjust the cut-off used for each telecom company, then save it.
3. **Predict churn** - upload customers; get Yes/No answer, probability, risk level and graphs.
4. **Tower lookup** - enter a place and company, then open the official Tarang Sanchar government portal to check real tower locations (this app has no tower data of its own).

### What the automatic cleaning does
- removes empty rows/columns, `Unnamed` columns and duplicate rows (training only)
- converts numbers stored as text (`"1,499"`, `"₹499"`) into numbers, junk words (`NA`, `null`, `-`) into missing
- replaces infinity and **every missing value with 0**
- writes the same company in one way (`jio`, `Reliance Jio` → `Jio`)

### Tips
- Column names in the prediction file should match the training file.
- No company column in the data = one common threshold. Add a company column to get one per company.
""")
