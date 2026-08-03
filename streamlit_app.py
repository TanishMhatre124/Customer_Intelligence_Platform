"""
Customer Intelligence & Churn Prediction Platform - Streamlit Dashboard
--------------------------------------------------------------------------
Run with:
    streamlit run streamlit_app.py

Reads the processed outputs produced by `python main.py` (data/processed,
reports/, models/) and presents every metric and chart from the project
in a single interactive app.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------
# Custom styling: gradient background, styled cards, typography
# -------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* App background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0f1729 0%, #131b33 45%, #1a2140 100%);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #10182c 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p {
    color: #e6e9f5 !important;
}

/* Headings */
h1, h2, h3 {
    color: #f5f6fb !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}
h1 {
    background: linear-gradient(90deg, #6d8bff, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
    padding-bottom: 4px;
}
p, span, label, .stMarkdown, .stCaption {
    color: #cbd2e6;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 18px 12px 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(109, 139, 255, 0.5);
}
[data-testid="stMetricLabel"] {
    color: #9aa4c7 !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #f5f6fb !important;
    font-weight: 800 !important;
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: nowrap !important;
}
[data-testid="stMetric"] div {
    overflow: visible !important;
}


/* Expander / info / warning boxes */
[data-testid="stExpander"], .stAlert {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Buttons */
.stButton button, .stDownloadButton button {
    background: linear-gradient(90deg, #6d8bff, #22d3ee);
    color: #0b1120;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    transition: filter 0.15s ease, transform 0.15s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}

/* Selectboxes / dropdowns */
[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    min-height: 42px !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="ValueContainer"],
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}
[data-baseweb="select"] svg {
    fill: #e6e9f5 !important;
    stroke: #e6e9f5 !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #000000 !important;
    stroke: #000000 !important;
}
[data-baseweb="select"] input,
[data-baseweb="select"] span {
    color: #e6e9f5 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #000000 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [role="option"] {
    color: #000000 !important;
}

/* Radio (sidebar nav) styled like a menu */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 6px 10px;
    margin-bottom: 4px;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(109, 139, 255, 0.15);
}

/* Section divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
px.defaults.template = PLOTLY_TEMPLATE
px.defaults.color_discrete_sequence = ["#6d8bff", "#22d3ee", "#f59f00", "#2f9e44", "#e8590c", "#a78bfa"]


PROCESSED = os.path.join(BASE_DIR, "data", "processed")
REPORTS = os.path.join(BASE_DIR, "reports")

PRIMARY = "#4C6EF5"
BR_STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
    "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco",
    "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima",
    "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe",
    "TO": "Tocantins",
}


def state_label(code: str) -> str:
    """Return 'São Paulo (SP)' for a state code, or the code itself if unknown."""
    name = BR_STATE_NAMES.get(code)
    return f"{name} ({code})" if name else code


SEGMENT_COLORS = {
    "Champions": "#2F9E44",
    "Loyal Customers": "#4C6EF5",
    "Potential Loyalists": "#22B8CF",
    "Need Attention": "#F59F00",
    "At Risk": "#E8590C",
    "Lost Customers": "#868E96",
}


def render_chart(fig, height=None):
    """Apply consistent transparent styling and render a plotly figure."""
    if height:
        fig.update_layout(height=height)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6e9f5",
        title_font_size=18,
        margin=dict(t=60, l=10, r=10, b=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f5f6fb", size=13),
            title=dict(font=dict(color="#f5f6fb")),
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Data loading (cached so the app stays snappy)
# -------------------------------------------------------------------

@st.cache_data
def load_data():
    missing = []

    def _read(path):
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
            return None
        return pd.read_csv(path)

    rfm = _read(os.path.join(PROCESSED, "rfm_customer_segments_scored.csv"))
    customer_features = _read(os.path.join(PROCESSED, "customer_features.csv"))
    orders = _read(os.path.join(PROCESSED, "orders_features.csv"))
    customers = _read(os.path.join(PROCESSED, "customers_clean.csv"))
    reviews = _read(os.path.join(PROCESSED, "reviews_clean.csv"))
    products = _read(os.path.join(PROCESSED, "products_clean.csv"))
    order_items = _read(os.path.join(PROCESSED, "order_items_clean.csv"))
    model_comparison = _read(os.path.join(REPORTS, "model_comparison.csv"))
    high_risk = _read(os.path.join(REPORTS, "high_risk_customers.csv"))
    retention = _read(os.path.join(PROCESSED, "retention_model_scored.csv"))

    if orders is not None:
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    return {
        "rfm": rfm,
        "customer_features": customer_features,
        "orders": orders,
        "customers": customers,
        "reviews": reviews,
        "products": products,
        "order_items": order_items,
        "model_comparison": model_comparison,
        "high_risk": high_risk,
        "retention": retention,
        "missing": missing,
    }


data = load_data()

if data["rfm"] is None:
    st.error(
        "Couldn't find `data/processed/rfm_customer_segments_scored.csv`.\n\n"
        "Run the pipeline first from the project root:\n\n"
        "```\npython main.py\n```\n\n"
        "then run `jupyter execute notebooks/07_Churn_Prediction.ipynb` "
        "(or just re-run notebook 07), which is what produces the scored file, "
        "and refresh this page."
    )
    st.stop()

rfm = data["rfm"]
customer_features = data["customer_features"]
orders = data["orders"]
customers = data["customers"]

# -------------------------------------------------------------------
# Sidebar - filters
# -------------------------------------------------------------------

st.sidebar.title("📊 Customer Intelligence")
st.sidebar.caption("Olist E-Commerce · RFM Segmentation & Churn")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Customer Segmentation",
        "Churn Prediction",
        "Geography",
        "High-Risk Watchlist",
        "Retention Model (Advanced)",
        "Model Performance",
        "Raw Data Explorer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

all_segments = sorted(rfm["Customer_Segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Customer segment", all_segments, default=all_segments
)

state_codes = []
if customers is not None:
    state_codes = sorted(customers["customer_state"].dropna().unique().tolist())

state_label_to_code = {"All": "All"}
state_label_to_code.update({state_label(code): code for code in state_codes})

selected_state_label = st.sidebar.selectbox("Customer state", list(state_label_to_code.keys()))
selected_state = state_label_to_code[selected_state_label]

# Apply filters
rfm_view = rfm[rfm["Customer_Segment"].isin(selected_segments)].copy()

if selected_state != "All" and customers is not None:
    state_ids = customers.loc[
        customers["customer_state"] == selected_state, "customer_unique_id"
    ].unique()
    rfm_view = rfm_view[rfm_view["customer_unique_id"].isin(state_ids)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(rfm_view):,}** of {len(rfm):,} customers")

if data["missing"]:
    st.sidebar.warning("Missing files: " + ", ".join(data["missing"]))


# -------------------------------------------------------------------
# Shared KPI header
# -------------------------------------------------------------------

def kpi_row(df):
    total_customers = len(df)
    total_revenue = df["Monetary"].sum()
    avg_order_value = (
        customer_features["average_order_value"].mean()
        if customer_features is not None
        else df["Monetary"].sum() / max(df["Frequency"].sum(), 1)
    )
    churn_rate = df["Churn"].mean() * 100 if "Churn" in df else float("nan")
    high_risk_count = int(df["Predicted_Churn"].sum()) if "Predicted_Churn" in df else 0

    top_cols = st.columns(3)
    top_cols[0].metric("Total Customers", f"{total_customers:,}")
    top_cols[1].metric("Total Revenue", f"R$ {total_revenue:,.0f}")
    top_cols[2].metric("Avg Order Value", f"R$ {avg_order_value:,.2f}")

    bottom_cols = st.columns(2)
    bottom_cols[0].metric("Churn Rate", f"{churn_rate:.1f}%")
    bottom_cols[1].metric("High-Risk Customers", f"{high_risk_count:,}")


# =====================================================================
# PAGE: Overview
# =====================================================================

if page == "Overview":
    st.title("Customer Intelligence Overview")
    st.caption(
        "End-to-end view of customer value, behavior, and churn risk — "
        "Olist Brazilian E-Commerce dataset."
    )

    kpi_row(rfm_view)
    st.markdown("---")

    filtered_orders = None
    if orders is not None:
        filtered_orders = orders
        if "customer_unique_id" in orders.columns:
            filtered_orders = orders[orders["customer_unique_id"].isin(rfm_view["customer_unique_id"])].copy()

    col1, col2 = st.columns([2, 1])

    with col1:
        if filtered_orders is not None:
            monthly = (
                filtered_orders.assign(
                    month=filtered_orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
                )
                .groupby("month")["order_value"]
                .sum()
                .reset_index()
                .sort_values("month")
            )
            fig = px.line(
                monthly, x="month", y="order_value", markers=True,
                title="Monthly Revenue Trend",
                labels={"month": "Month", "order_value": "Revenue (R$)"},
            )
            fig.update_layout(xaxis_tickangle=-45, height=380)
            fig.update_traces(line_color=PRIMARY)
            render_chart(fig)

    with col2:
        seg_counts = rfm_view["Customer_Segment"].value_counts().reset_index()
        seg_counts.columns = ["Customer_Segment", "Count"]
        fig = px.pie(
            seg_counts, names="Customer_Segment", values="Count",
            title="Customer Mix",
            color="Customer_Segment", color_discrete_map=SEGMENT_COLORS, hole=0.45,
        )
        fig.update_layout(height=380, showlegend=True)
        render_chart(fig)

    st.subheader("Recency / Frequency / Monetary Distributions")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.histogram(rfm_view, x="Recency", nbins=40, title="Recency (days)")
        fig.update_traces(marker_color=PRIMARY)
        render_chart(fig)
    with c2:
        fig = px.histogram(rfm_view, x="Frequency", nbins=20, title="Frequency (orders)")
        fig.update_traces(marker_color=PRIMARY)
        render_chart(fig)
    with c3:
        fig = px.histogram(
            rfm_view[rfm_view["Monetary"] < rfm_view["Monetary"].quantile(0.99)],
            x="Monetary", nbins=40, title="Monetary (R$, 99th pct capped)"
        )
        fig.update_traces(marker_color=PRIMARY)
        render_chart(fig)


# =====================================================================
# PAGE: Customer Segmentation
# =====================================================================

elif page == "Customer Segmentation":
    st.title("Customer Segmentation (RFM)")
    st.caption(
        "Segments are assigned from Recency, Frequency, and Monetary quintile scores."
    )

    kpi_row(rfm_view)
    st.markdown("---")

    segment_summary = (
        rfm_view.groupby("Customer_Segment")
        .agg(
            Customer_Count=("customer_unique_id", "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
        .reset_index()
    )
    segment_summary["Pct_of_Customers"] = (
        segment_summary["Customer_Count"] / segment_summary["Customer_Count"].sum() * 100
    ).round(1)
    segment_summary["Pct_of_Revenue"] = (
        segment_summary["Total_Revenue"] / segment_summary["Total_Revenue"].sum() * 100
    ).round(1)
    segment_summary = segment_summary.sort_values("Total_Revenue", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            segment_summary, x="Customer_Segment", y="Customer_Count",
            color="Customer_Segment", color_discrete_map=SEGMENT_COLORS,
            text="Pct_of_Customers", title="Customers by Segment",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, height=420)
        render_chart(fig)

    with col2:
        fig = px.bar(
            segment_summary, x="Customer_Segment", y="Total_Revenue",
            color="Customer_Segment", color_discrete_map=SEGMENT_COLORS,
            text="Pct_of_Revenue", title="Revenue by Segment",
            labels={"Total_Revenue": "Revenue (R$)"},
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, height=420)
        render_chart(fig)

    st.info(
        "**How to read this:** segments with a much higher share of revenue than "
        "share of customers (e.g. Champions) are the highest priority to retain. "
        "Segments with many customers but low revenue share are lower priority for "
        "retention spend."
    )

    st.subheader("Segment Summary Table")
    st.dataframe(
        segment_summary.style.format({
            "Total_Revenue": "R$ {:,.0f}",
            "Avg_Recency": "{:.0f}",
            "Avg_Frequency": "{:.1f}",
            "Avg_Monetary": "R$ {:,.2f}",
            "Pct_of_Customers": "{:.1f}%",
            "Pct_of_Revenue": "{:.1f}%",
        }),
        use_container_width=True,
    )

    st.subheader("RFM Scatter (Recency vs Monetary, sized by Frequency)")
    sample = rfm_view.sample(min(5000, len(rfm_view)), random_state=42) if len(rfm_view) > 5000 else rfm_view
    fig = px.scatter(
        sample, x="Recency", y="Monetary", size="Frequency", color="Customer_Segment",
        color_discrete_map=SEGMENT_COLORS, opacity=0.6,
        labels={"Recency": "Recency (days)", "Monetary": "Total Spend (R$)"},
        height=500,
    )
    render_chart(fig)


# =====================================================================
# PAGE: Churn Prediction
# =====================================================================

elif page == "Churn Prediction":
    st.title("Churn Prediction")

    if "Churn_Probability" not in rfm_view.columns:
        st.warning(
            "No churn scores found. Run/re-run `notebooks/07_Churn_Prediction.ipynb` "
            "to generate `rfm_customer_segments_scored.csv`."
        )
    else:
        kpi_row(rfm_view)
        st.markdown("---")

        with st.expander("⚠️ Important caveat on this model", expanded=False):
            st.markdown(
                "Churn is **defined** as `Recency > 180 days AND Frequency <= 2 orders`, "
                "then **predicted** using `Recency`, `Frequency`, and `Monetary`. "
                "Because the label is built from two of the same three input features, "
                "the model scores near-perfectly on this data — that reflects it "
                "faithfully learning the rule, not genuine predictive power on unseen "
                "future behavior. Treat these scores as an automatic, explainable "
                "implementation of the business rule and a prioritization tool, "
                "not a forecast. See `notebooks/07_Churn_Prediction.ipynb` for the full "
                "discussion and what a leakage-free version would require.\n\n"
                "**👉 For a genuinely predictive model** built on real future outcomes "
                "instead of a circular rule, see the **Retention Model (Advanced)** page "
                "in the sidebar."
            )

        col1, col2 = st.columns(2)

        with col1:
            churn_by_segment = (
                rfm_view.groupby("Customer_Segment")["Predicted_Churn"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "High_Risk", "count": "Total"})
                .reset_index()
            )
            churn_by_segment["Risk_Rate_%"] = (
                churn_by_segment["High_Risk"] / churn_by_segment["Total"] * 100
            ).round(1)
            churn_by_segment = churn_by_segment.sort_values("Risk_Rate_%", ascending=False)

            fig = px.bar(
                churn_by_segment, x="Customer_Segment", y="Risk_Rate_%",
                color="Risk_Rate_%", color_continuous_scale="Reds",
                title="Churn Risk Rate by Segment", labels={"Risk_Rate_%": "% Flagged High-Risk"},
            )
            fig.update_layout(height=420)
            render_chart(fig)

        with col2:
            fig = px.histogram(
                rfm_view, x="Churn_Probability", nbins=40, color_discrete_sequence=[PRIMARY],
                title="Predicted Churn Probability Distribution",
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                          annotation_text="Decision threshold")
            fig.update_layout(height=420)
            render_chart(fig)

        st.subheader("Churn Rate vs. Recency")
        recency_bins = pd.cut(rfm_view["Recency"], bins=10)
        recency_churn = (
            rfm_view.assign(Recency_Bin=recency_bins.astype(str))
            .groupby("Recency_Bin")["Predicted_Churn"]
            .mean()
            .reset_index()
        )
        recency_churn["Predicted_Churn"] *= 100
        fig = px.bar(
            recency_churn, x="Recency_Bin", y="Predicted_Churn",
            labels={"Predicted_Churn": "% Predicted Churned", "Recency_Bin": "Recency (days)"},
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(xaxis_tickangle=-30, height=380)
        render_chart(fig)


# =====================================================================
# PAGE: Geography
# =====================================================================

elif page == "Geography":
    st.title("Geographic Distribution")

    if customers is None:
        st.warning("customers_clean.csv not found.")
    else:
        customer_state = customers[["customer_unique_id", "customer_state"]].drop_duplicates(
            "customer_unique_id"
        )
        rfm_geo = rfm_view.merge(customer_state, on="customer_unique_id", how="left")

        state_summary = (
            rfm_geo.groupby("customer_state")
            .agg(
                Customers=("customer_unique_id", "count"),
                Revenue=("Monetary", "sum"),
                Avg_Churn_Risk=("Predicted_Churn", "mean") if "Predicted_Churn" in rfm_geo else ("Monetary", "mean"),
            )
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )
        if "Avg_Churn_Risk" in state_summary:
            state_summary["Avg_Churn_Risk"] = (state_summary["Avg_Churn_Risk"] * 100).round(1)

        state_summary.insert(0, "State", state_summary["customer_state"].map(
            lambda c: BR_STATE_NAMES.get(c, c)
        ))
        state_summary = state_summary.drop(columns=["customer_state"])

        kpi_row(rfm_view)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                state_summary.head(15), x="State", y="Revenue",
                title="Top 15 States by Revenue", color_discrete_sequence=[PRIMARY],
                labels={"Revenue": "Revenue (R$)"},
            )
            fig.update_layout(height=420)
            render_chart(fig)

        with col2:
            fig = px.bar(
                state_summary.head(15), x="State", y="Customers",
                title="Top 15 States by Customer Count", color_discrete_sequence=["#22B8CF"],
            )
            fig.update_layout(height=420)
            render_chart(fig)

        st.subheader("State Summary Table")
        st.dataframe(state_summary, use_container_width=True)


# =====================================================================
# PAGE: High-Risk Watchlist
# =====================================================================

elif page == "High-Risk Watchlist":
    st.title("High-Risk Customer Watchlist")
    st.caption("Customers flagged as likely to churn, ranked by value at risk.")

    if "Predicted_Churn" not in rfm_view.columns:
        st.warning("No churn predictions found. Run notebook 07 first.")
    else:
        watchlist = rfm_view[rfm_view["Predicted_Churn"] == 1].copy()

        min_spend = st.slider(
            "Minimum lifetime spend (R$)", 0, int(watchlist["Monetary"].max()), 0, step=50
        )
        watchlist = watchlist[watchlist["Monetary"] >= min_spend]
        watchlist = watchlist.sort_values(["Monetary", "Churn_Probability"], ascending=[False, False])

        c1, c2, c3 = st.columns(3)
        c1.metric("Customers on Watchlist", f"{len(watchlist):,}")
        c2.metric("Revenue at Risk", f"R$ {watchlist['Monetary'].sum():,.0f}")
        c3.metric("Avg Churn Probability", f"{watchlist['Churn_Probability'].mean() * 100:.1f}%")

        st.dataframe(
            watchlist[
                ["customer_unique_id", "Customer_Segment", "Recency", "Frequency",
                 "Monetary", "Churn_Probability"]
            ].head(200).style.format({
                "Monetary": "R$ {:,.2f}",
                "Churn_Probability": "{:.1%}",
            }),
            use_container_width=True,
            height=500,
        )

        st.download_button(
            "Download full watchlist as CSV",
            watchlist.to_csv(index=False).encode("utf-8"),
            file_name="high_risk_watchlist.csv",
            mime="text/csv",
        )


# =====================================================================
# PAGE: Model Performance
# =====================================================================

elif page == "Retention Model (Advanced)":
    st.title("Advanced Retention Model")
    st.caption(
        "A genuinely predictive, leakage-free alternative to the rule-based churn model — "
        "built only from a customer's first order, predicting a real future event."
    )

    retention = data["retention"]

    if retention is None:
        st.warning(
            "No retention model data found. Run:\n\n"
            "```\npython -m src.retention_model\n```\n\n"
            "or execute `notebooks/09_Improved_Retention_Model.ipynb`, then refresh."
        )
    else:
        with st.expander("How this differs from the Churn Prediction page", expanded=True):
            st.markdown(
                "- **Label:** did the customer place a **second order within 90 days** "
                "of their first? A real, future, verifiable outcome — not a rule built "
                "from the same features used to predict it.\n"
                "- **Features:** only what was knowable from the customer's **first order** "
                "(delivery time, review score, payment method, price, product category, "
                "season) — nothing from their later behavior.\n"
                "- **Result:** a modest but honest ROC-AUC (~0.60) instead of a suspicious "
                "1.00 — this model has to actually find signal, and mostly finds it in "
                "**product category**, not delivery/service quality."
            )

        repeat_rate = retention["repeat_within_90"].mean() * 100
        eligible = len(retention)
        avg_prob = retention["Repeat_Probability"].mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Eligible First-Time Customers", f"{eligible:,}")
        c2.metric("Actual Repeat Rate (90 days)", f"{repeat_rate:.2f}%")
        c3.metric("Avg. Predicted Repeat Probability", f"{avg_prob:.2f}%")

        st.markdown("---")

        st.subheader("What actually predicts repeat purchase?")
        col1, col2 = st.columns(2)

        with col1:
            cat_rate = (
                retention.groupby("category")["repeat_within_90"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            cat_rate["repeat_within_90"] *= 100
            fig = px.bar(
                cat_rate, x="repeat_within_90", y="category", orientation="h",
                title="Repeat-Purchase Rate by Product Category",
                labels={"repeat_within_90": "Repeat Rate (%)", "category": "Category"},
            )
            render_chart(fig, height=420)

        with col2:
            review_rate = (
                retention.assign(review_round=retention["review_score"].round())
                .groupby("review_round")["repeat_within_90"]
                .mean()
                .reset_index()
            )
            review_rate["repeat_within_90"] *= 100
            fig = px.bar(
                review_rate, x="review_round", y="repeat_within_90",
                title="Repeat-Purchase Rate by First-Order Review Score",
                labels={"review_round": "Review Score", "repeat_within_90": "Repeat Rate (%)"},
            )
            render_chart(fig, height=420)

        col3, col4 = st.columns(2)
        with col3:
            delivery_bins = pd.cut(
                retention["delivery_days"], bins=[0, 5, 10, 15, 20, 30, 200],
                labels=["0-5", "6-10", "11-15", "16-20", "21-30", "30+"]
            )
            delivery_rate = (
                retention.assign(bin=delivery_bins)
                .groupby("bin", observed=True)["repeat_within_90"]
                .mean()
                .reset_index()
            )
            delivery_rate["repeat_within_90"] *= 100
            fig = px.bar(
                delivery_rate, x="bin", y="repeat_within_90",
                title="Repeat-Purchase Rate by Delivery Time (days)",
                labels={"bin": "Delivery Days", "repeat_within_90": "Repeat Rate (%)"},
            )
            render_chart(fig, height=380)

        with col4:
            month_rate = (
                retention.groupby("purchase_month")["repeat_within_90"]
                .mean()
                .reset_index()
            )
            month_rate["repeat_within_90"] *= 100
            fig = px.line(
                month_rate, x="purchase_month", y="repeat_within_90", markers=True,
                title="Repeat-Purchase Rate by Purchase Month",
                labels={"purchase_month": "Month", "repeat_within_90": "Repeat Rate (%)"},
            )
            render_chart(fig, height=380)

        st.info(
            "**Reading these charts together:** category has by far the widest spread "
            "(home/decor customers reorder at several times the rate of gadget/gift buyers), "
            "while review score and delivery speed barely move the repeat rate at all. "
            "That's a genuinely useful negative result: for *this* marketplace, service "
            "recovery (faster shipping, better support) is unlikely to be the highest-leverage "
            "retention investment — category-based cross-sell is."
        )

        st.markdown("---")
        st.subheader("Highest-Potential New Customers")
        st.caption(
            "First-time buyers ranked by predicted probability of reordering within 90 days — "
            "good candidates for an early loyalty nudge or second-purchase incentive."
        )
        top_prospects = retention.sort_values("Repeat_Probability", ascending=False).head(100)
        st.dataframe(
            top_prospects[
                ["customer_unique_id", "category", "order_value", "review_score",
                 "delivery_days", "Repeat_Probability"]
            ].style.format({
                "order_value": "R$ {:,.2f}",
                "review_score": "{:.1f}",
                "Repeat_Probability": "{:.1%}",
            }),
            use_container_width=True,
            height=400,
        )


elif page == "Model Performance":
    st.title("Churn Model Performance")

    model_comparison = data["model_comparison"]
    if model_comparison is None:
        st.warning("reports/model_comparison.csv not found. Run `python main.py` first.")
    else:
        st.subheader("Model Comparison")
        st.dataframe(
            model_comparison.style.format({
                col: "{:.3f}" for col in model_comparison.columns if col != "model"
            }).background_gradient(cmap="Greens", subset=[c for c in model_comparison.columns if c != "model"]),
            use_container_width=True,
        )

        metric_cols = [c for c in model_comparison.columns if c != "model"]
        melted = model_comparison.melt(id_vars="model", value_vars=metric_cols,
                                        var_name="Metric", value_name="Score")
        fig = px.bar(
            melted, x="Metric", y="Score", color="model", barmode="group",
            title="Model Comparison by Metric",
        )
        fig.update_layout(height=450)
        render_chart(fig)

        st.info(
            "All models score very highly here because the churn label is derived "
            "directly from Recency/Frequency, which are also the model's inputs — "
            "see the caveat on the Churn Prediction page for details."
        )

    st.subheader("Feature Importance")
    try:
        import joblib
        model_dir = os.path.join(BASE_DIR, "models")
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
        if model_files:
            model = joblib.load(os.path.join(model_dir, model_files[0]))
            features = ["Recency", "Frequency", "Monetary"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = abs(model.coef_[0])
            else:
                importances = None

            if importances is not None:
                imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
                    "Importance", ascending=False
                )
                fig = px.bar(
                    imp_df, x="Importance", y="Feature", orientation="h",
                    title=f"Feature Importance ({model_files[0].replace('.joblib', '')})",
                    color_discrete_sequence=[PRIMARY],
                )
                render_chart(fig)
        else:
            st.info("No saved model found in models/. Run `python main.py` to train and save one.")
    except Exception as e:
        st.info(f"Could not load model for feature importance: {e}")


# =====================================================================
# PAGE: Raw Data Explorer
# =====================================================================

elif page == "Raw Data Explorer":
    st.title("Raw Data Explorer")
    st.caption("Browse any processed table directly.")

    table_options = {
        "RFM + Churn Scores": rfm,
        "Customer Features": customer_features,
        "Orders (with engineered features)": orders,
        "Customers": customers,
        "Reviews": data["reviews"],
        "Products": data["products"],
        "Order Items": data["order_items"],
        "High-Risk Customers (report)": data["high_risk"],
    }
    table_options = {k: v for k, v in table_options.items() if v is not None}

    choice = st.selectbox("Select a table", list(table_options.keys()))
    df = table_options[choice]

    st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(500), use_container_width=True, height=500)

    st.download_button(
        f"Download {choice} as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{choice.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )


st.sidebar.markdown("---")
st.sidebar.caption("Customer Intelligence & Churn Prediction Platform · Tanish Mhatre")
