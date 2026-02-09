import io
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="UN Finance & Indicators Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))
    raise ValueError("Unsupported file type. Upload CSV or Excel.")

def normalize_financial_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected flexible schema:
      - date OR period column (any of: date, period, month, year)
      - account / line_item column (any of: account, line_item, item, category)
      - amount column (any of: amount, value, actual, budget)
    If your file differs, the mapping UI will handle it.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    return df

def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def compute_variance(actual: pd.Series, budget: pd.Series) -> pd.Series:
    return actual - budget

def compute_variance_pct(actual: pd.Series, budget: pd.Series) -> pd.Series:
    denom = budget.replace({0: np.nan})
    return (actual - budget) / denom

# IMF DataMapper API (simple JSON)
# Base URL documented/used broadly: https://www.imf.org/external/datamapper/api/v1/  :contentReference[oaicite:9]{index=9}
@st.cache_data(show_spinner=False, ttl=3600)
def imf_datamapper_get(indicator: str, country: str, periods: str = "") -> pd.DataFrame:
    base = "https://www.imf.org/external/datamapper/api/v1"
    url = f"{base}/{indicator}/{country}"
    params = {}
    if periods:
        params["periods"] = periods
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    # DataMapper returns nested JSON; a common pattern is js['values'][indicator][country]
    values = None
    if isinstance(js, dict):
        values = (js.get("values", {}) or {}).get(indicator, {}).get(country)

    if not values:
        return pd.DataFrame(columns=["period", "value"])

    out = pd.DataFrame(
        [{"period": k, "value": v} for k, v in values.items()]
    )
    out["period"] = pd.to_numeric(out["period"], errors="ignore")
    out = out.sort_values("period")
    return out

# World Bank SDMX API exists for WDI :contentReference[oaicite:10]{index=10}
# This is a placeholder that returns empty unless you wire a concrete SDMX query.
@st.cache_data(show_spinner=False, ttl=3600)
def worldbank_sdmx_placeholder() -> pd.DataFrame:
    return pd.DataFrame()

# -----------------------------
# UI
# -----------------------------
st.title("Finance + Market & Economic Indicators Dashboard (UN Partnerships)")

with st.sidebar:
    st.header("1) Load internal financials")
    fin_file = st.file_uploader("Upload financial statements (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    st.caption("Tip: include columns like Date/Period, Account/Line Item, Actual, Budget (optional).")

    st.header("2) External indicators")
    st.caption("IMF DataMapper is a quick way to fetch core macro series. :contentReference[oaicite:11]{index=11}")
    imf_indicator = st.text_input("IMF indicator code (e.g., NGDP_RPCH, PCPI)", value="PCPI")
    imf_country = st.text_input("Country code (ISO3, e.g., USA, KEN, IND)", value="USA")
    imf_periods = st.text_input("Periods (comma-separated years, optional)", value="2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025")
    fetch_imf = st.button("Fetch IMF series")

# -----------------------------
# Main panels
# -----------------------------
colA, colB = st.columns([1.35, 1.0], gap="large")

with colA:
    st.subheader("Internal financials")
    if fin_file:
        raw = read_table(fin_file)
        df = normalize_financial_df(raw)

        st.write("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("### Map your columns (so the app works with any dataset)")
        cols = list(df.columns)

        date_col = st.selectbox("Date/Period column", options=["(none)"] + cols, index=0)
        item_col = st.selectbox("Account / Line item column", options=["(none)"] + cols, index=0)
        actual_col = st.selectbox("Actual/Value column", options=["(none)"] + cols, index=0)
        budget_col = st.selectbox("Budget column (optional)", options=["(none)"] + cols, index=0)

        if actual_col != "(none)":
            work = df.copy()

            # Build a working period column
            if date_col != "(none)":
                work["period"] = safe_to_datetime(work[date_col])
                # If only year is provided, this still works; otherwise use month.
                work["period"] = work["period"].dt.to_period("M").dt.to_timestamp()
            else:
                work["period"] = pd.NaT

            if item_col != "(none)":
                work["item"] = work[item_col].astype(str)
            else:
                work["item"] = "TOTAL"

            work["actual"] = pd.to_numeric(work[actual_col], errors="coerce").fillna(0.0)

            if budget_col != "(none)":
                work["budget"] = pd.to_numeric(work[budget_col], errors="coerce").fillna(0.0)
                work["variance"] = compute_variance(work["actual"], work["budget"])
                work["variance_pct"] = compute_variance_pct(work["actual"], work["budget"])
            else:
                work["budget"] = np.nan
                work["variance"] = np.nan
                work["variance_pct"] = np.nan

            # Aggregate
            agg = (
                work.groupby(["period", "item"], dropna=False)[["actual", "budget", "variance"]]
                .sum()
                .reset_index()
                .sort_values(["period", "item"])
            )

            st.markdown("### Trend (Actual)")
            if agg["period"].notna().any():
                fig = px.line(
                    agg.dropna(subset=["period"]),
                    x="period",
                    y="actual",
                    color="item",
                    markers=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No usable Date/Period found. Map a date-like column to enable trends.")

            if budget_col != "(none)":
                st.markdown("### Budget vs Actual (Variance)")
                piv = agg.dropna(subset=["period"]).groupby("period")[["actual", "budget", "variance"]].sum().reset_index()
                fig2 = px.bar(piv, x="period", y=["actual", "budget"], barmode="group")
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.line(piv, x="period", y="variance")
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### KPI flags (simple examples)")
            total_actual = float(agg["actual"].sum())
            total_budget = float(agg["budget"].sum()) if budget_col != "(none)" else np.nan

            k1, k2, k3 = st.columns(3)
            k1.metric("Total Actual", f"{total_actual:,.0f}")
            if budget_col != "(none)":
                k2.metric("Total Budget", f"{total_budget:,.0f}")
                k3.metric("Total Variance", f"{(total_actual - total_budget):,.0f}")
            else:
                k2.metric("Total Budget", "—")
                k3.metric("Total Variance", "—")

            st.markdown("### Recommendations (starter logic)")
            recs = []
            if budget_col != "(none)":
                var_pct = (total_actual - total_budget) / (total_budget if total_budget else np.nan)
                if not math.isnan(var_pct):
                    if var_pct > 0.05:
                        recs.append("Spend is running **over budget** (>5%). Review top variance line-items and re-forecast.")
                    elif var_pct < -0.05:
                        recs.append("Spend is **under budget** (<-5%). Check pipeline bottlenecks (procurement / hiring) and delivery risks.")
                    else:
                        recs.append("Budget execution is **within tolerance** (±5%). Continue monitoring burn rate and commitments.")
            else:
                recs.append("Add a Budget column to enable variance analysis and budget execution controls.")

            for r in recs:
                st.write("• " + r)

        else:
            st.warning("Please map at least an Actual/Value column to proceed.")
    else:
        st.info("Upload a financial dataset to start (CSV/XLSX).")

with colB:
    st.subheader("External indicators (macro / market)")
    if fetch_imf:
        try:
            imf_df = imf_datamapper_get(imf_indicator.strip(), imf_country.strip(), imf_periods.strip())
            if imf_df.empty:
                st.warning("No data returned. Double-check indicator/country codes.")
            else:
                st.write("IMF DataMapper series")
                st.dataframe(imf_df, use_container_width=True)
                fig = px.line(imf_df, x="period", y="value", markers=True)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"IMF fetch failed: {e}")

    st.markdown("### Add more sources (recommended)")
    st.write(
        "- **IMF SDMX** for broader financial statistics datasets (documented by IMF). :contentReference[oaicite:12]{index=12}\n"
        "- **World Bank SDMX (WDI)** for development indicators. :contentReference[oaicite:13]{index=13}\n"
        "- **UNSD SDG API** for SDG-series relevant to financing and outcomes. :contentReference[oaicite:14]{index=14}"
    )
