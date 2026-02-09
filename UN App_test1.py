import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="UN Finance & Econ Dashboard (UNSD SDG API v5)", layout="wide")

UNSD_V5_BASE = "https://unstats.un.org/sdgs/UNSDGAPIV5/v1/sdg"

# -----------------------------
# Helpers
# -----------------------------
def _safe_get_json(url: str, params: dict | None = None, timeout: int = 40):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _rows_from_json(j):
    # API sometimes returns {"data":[...], ...} or a direct list
    if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
        return j["data"]
    if isinstance(j, list):
        return j
    # fallback: try common keys
    for k in ["Data", "result", "Results"]:
        if isinstance(j, dict) and k in j and isinstance(j[k], list):
            return j[k]
    return []

def _pick_col(df: pd.DataFrame, candidates: list[str]):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fallback: fuzzy contains
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def _coerce_year(series: pd.Series) -> pd.Series:
    # Try direct numeric first
    y = pd.to_numeric(series, errors="coerce")
    if y.notna().sum() > 0:
        return y

    # Try parsing year from strings (e.g. "2019", "2019-01-01", "2019/2020")
    def extract_year(v):
        if pd.isna(v):
            return np.nan
        m = re.search(r"(19|20)\d{2}", str(v))
        return float(m.group(0)) if m else np.nan

    return series.apply(extract_year)

def compute_kpis(df: pd.DataFrame, year_col: str, value_col: str, group_col: str):
    kpi_rows = []
    for g, sub in df.dropna(subset=[year_col, value_col]).groupby(group_col):
        sub = sub.sort_values(year_col)
        if sub.empty:
            continue
        latest = sub.iloc[-1]
        latest_year = latest[year_col]
        latest_val = latest[value_col]

        # YoY change
        yoy = np.nan
        if len(sub) >= 2:
            prev = sub.iloc[-2][value_col]
            if pd.notna(prev) and prev != 0:
                yoy = (latest_val - prev) / abs(prev) * 100.0

        # 5-year CAGR (or best available span)
        cagr = np.nan
        if len(sub) >= 2:
            start = sub.iloc[max(0, len(sub) - 6)]  # try ~5-year back
            years = latest_year - start[year_col]
            if pd.notna(years) and years > 0 and pd.notna(start[value_col]) and start[value_col] != 0:
                cagr = ((latest_val / start[value_col]) ** (1.0 / years) - 1.0) * 100.0

        # volatility (std of pct change)
        vol = np.nan
        pct = sub[value_col].pct_change()
        if pct.notna().sum() >= 2:
            vol = pct.std() * 100.0

        kpi_rows.append(
            {
                group_col: g,
                "Latest year": int(latest_year) if pd.notna(latest_year) else latest_year,
                "Latest value": latest_val,
                "YoY %": yoy,
                "CAGR % (≈5y)": cagr,
                "Volatility % (pctchg std)": vol,
            }
        )
    return pd.DataFrame(kpi_rows)

@st.cache_data(ttl=60 * 60)
def fetch_geoareas():
    url = f"{UNSD_V5_BASE}/GeoArea/List"
    j = _safe_get_json(url)
    rows = _rows_from_json(j)
    df = pd.DataFrame(rows)
    return df

@st.cache_data(ttl=60 * 60)
def fetch_indicators():
    url = f"{UNSD_V5_BASE}/Indicator/List"
    j = _safe_get_json(url)
    rows = _rows_from_json(j)
    df = pd.DataFrame(rows)
    return df

@st.cache_data(ttl=60 * 30)
def fetch_indicator_data(indicator_code: str, area_codes: list[str], page_size: int = 10000):
    url = f"{UNSD_V5_BASE}/Indicator/Data"
    all_frames = []
    for ac in area_codes:
        params = {"indicator": indicator_code, "areaCode": ac, "pageSize": page_size}
        j = _safe_get_json(url, params=params)
        rows = _rows_from_json(j)
        if rows:
            df = pd.DataFrame(rows)
            df["__areaCode_req"] = ac
            all_frames.append(df)
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)

def normalize_dataset(df: pd.DataFrame):
    if df.empty:
        return df, None, None, None

    # Guess core columns
    year_col = _pick_col(df, ["timePeriodStart", "timePeriod", "year", "TimePeriodStart", "Year"])
    value_col = _pick_col(df, ["value", "Value", "obsValue", "ObservationValue"])
    area_name_col = _pick_col(df, ["geoAreaName", "GeoAreaName", "areaName", "Country", "country"])
    area_code_col = _pick_col(df, ["geoAreaCode", "GeoAreaCode", "areaCode", "__areaCode_req"])
    series_col = _pick_col(df, ["series", "seriesCode", "SeriesCode", "seriesDescription"])

    # If no area name, use area code
    if area_name_col is None and area_code_col is not None:
        df["geoAreaName"] = df[area_code_col].astype(str)
        area_name_col = "geoAreaName"

    # If no year col, create placeholder
    if year_col is None:
        # Try to find anything resembling time
        maybe_time = [c for c in df.columns if "time" in c.lower() or "period" in c.lower() or "year" in c.lower()]
        year_col = maybe_time[0] if maybe_time else None

    # Coerce year/value
    if year_col is not None:
        df["__year"] = _coerce_year(df[year_col])
    else:
        df["__year"] = np.nan

    if value_col is not None:
        df["__value"] = pd.to_numeric(df[value_col], errors="coerce")
    else:
        df["__value"] = np.nan

    # Group label
    if area_name_col is not None:
        df["__group"] = df[area_name_col].astype(str)
    elif area_code_col is not None:
        df["__group"] = df[area_code_col].astype(str)
    else:
        df["__group"] = "All"

    # Series label (optional)
    if series_col is not None:
        df["__series"] = df[series_col].astype(str)
    else:
        df["__series"] = "Series"

    # Clean for plotting
    cleaned = df.dropna(subset=["__year", "__value"]).copy()
    cleaned["__year"] = cleaned["__year"].astype(int)

    return cleaned, "__year", "__value", "__group"


# -----------------------------
# UI
# -----------------------------
st.title("UN Office for Partnerships — Financial & Economic Dashboard (UNSD SDG API v5)")
st.caption(
    "Use SDG indicators as economic/financial signals (macro context, outcomes, risk drivers). "
    "Upload internal finance files to compare against external SDG trends."
)

source_mode = st.sidebar.radio("Data source", ["UNSD SDG API v5", "Upload CSV/XLSX"], index=0)

df_raw = pd.DataFrame()

if source_mode == "UNSD SDG API v5":
    st.sidebar.subheader("API selectors")

    # Load lists
    with st.spinner("Loading GeoAreas + Indicators..."):
        geo_df = fetch_geoareas()
        ind_df = fetch_indicators()

    # Detect columns for lists (robust)
    geo_code = _pick_col(geo_df, ["geoAreaCode", "GeoAreaCode", "areaCode"])
    geo_name = _pick_col(geo_df, ["geoAreaName", "GeoAreaName", "areaName"])

    ind_code = _pick_col(ind_df, ["code", "indicator", "Indicator", "indicatorCode"])
    ind_desc = _pick_col(ind_df, ["description", "indicatorDescription", "IndicatorDescription", "name", "Title"])

    if geo_code is None or geo_name is None:
        st.error("Could not detect GeoArea code/name columns from API response.")
        st.stop()

    if ind_code is None:
        st.error("Could not detect Indicator code column from API response.")
        st.stop()

    # Build selector display
    geo_df["_label"] = geo_df[geo_name].astype(str) + " (M49 " + geo_df[geo_code].astype(str) + ")"
    geo_options = geo_df.sort_values(geo_name)["_label"].tolist()

    if ind_desc is not None:
        ind_df["_label"] = ind_df[ind_code].astype(str) + " — " + ind_df[ind_desc].astype(str)
    else:
        ind_df["_label"] = ind_df[ind_code].astype(str)
    ind_options = ind_df.sort_values(ind_code)["_label"].tolist()

    indicator_label = st.sidebar.selectbox("Indicator", ind_options, index=0)
    # extract code before " — "
    indicator_code = indicator_label.split(" — ")[0].strip()

    chosen_geo = st.sidebar.multiselect("GeoAreas (countries/regions)", geo_options, default=geo_options[:1])
    area_codes = [re.search(r"\(M49\s+([0-9]+)\)", x).group(1) for x in chosen_geo] if chosen_geo else []

    page_size = st.sidebar.slider("Page size", min_value=1000, max_value=20000, value=10000, step=1000)

    if st.sidebar.button("Fetch data", type="primary") and area_codes:
        with st.spinner("Fetching indicator observations..."):
            df_raw = fetch_indicator_data(indicator_code, area_codes, page_size=page_size)

else:
    st.sidebar.subheader("Upload dataset")
    up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(up)
        else:
            df_raw = pd.read_excel(up)

# -----------------------------
# Data prep + filters
# -----------------------------
if df_raw.empty:
    st.info("Pick an indicator + GeoArea(s) and click **Fetch data**, or upload a file to begin.")
    st.stop()

df, year_col, value_col, group_col = normalize_dataset(df_raw)
if df.empty:
    st.warning("No numeric time/value observations found after cleaning. Try another indicator/GeoArea or dataset.")
    st.dataframe(df_raw.head(50), use_container_width=True)
    st.stop()

# Optional categorical filters
st.sidebar.subheader("Filters (auto-detected)")
cat_cols = []
for c in df.columns:
    if c.startswith("__"):
        continue
    if df[c].dtype == "object":
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 30:
            cat_cols.append(c)

selected_filters = {}
for c in cat_cols[:6]:  # cap UI complexity
    opts = sorted([x for x in df[c].dropna().unique().tolist()])
    chosen = st.sidebar.multiselect(f"{c}", opts, default=[])
    if chosen:
        selected_filters[c] = chosen

df_f = df.copy()
for c, chosen in selected_filters.items():
    df_f = df_f[df_f[c].isin(chosen)]

if df_f.empty:
    st.warning("Filters removed all data. Relax filters.")
    st.stop()

# -----------------------------
# KPI row
# -----------------------------
kpi_df = compute_kpis(df_f, year_col, value_col, group_col)

top = st.container()
with top:
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Key metrics")
        st.dataframe(kpi_df.sort_values("Latest year", ascending=False), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Observations (sample)")
        st.dataframe(df_f[[group_col, year_col, value_col]].sort_values([group_col, year_col]).head(200),
                     use_container_width=True, hide_index=True)

# -----------------------------
# Build 10 charts
# -----------------------------
st.subheader("Dashboard — 10 chart types")

# Aggregations used across charts
latest_year = int(df_f[year_col].max())
latest_df = df_f[df_f[year_col] == latest_year].copy()

# 1) Line
fig1 = px.line(df_f.sort_values(year_col), x=year_col, y=value_col, color=group_col, markers=True,
               title="1) Line — Trend over time")

# 2) Area
fig2 = px.area(df_f.sort_values(year_col), x=year_col, y=value_col, color=group_col,
               title="2) Area — Cumulative movement over time")

# 3) Bar (latest year)
fig3 = px.bar(latest_df, x=group_col, y=value_col,
              title=f"3) Bar — Latest year ({latest_year}) comparison")

# 4) Stacked bar (by group across years, stacked by group)
# Use year as x and group as stack
agg_year_group = df_f.groupby([year_col, group_col], as_index=False)[value_col].mean()
fig4 = px.bar(agg_year_group, x=year_col, y=value_col, color=group_col, barmode="stack",
              title="4) Stacked bar — Composition by GeoArea across years (mean if multiple obs)")

# 5) Scatter
fig5 = px.scatter(df_f, x=year_col, y=value_col, color=group_col,
                  title="5) Scatter — Observations over time")

# 6) Bubble (size=abs(value))
bubble = df_f.copy()
bubble["__size"] = bubble[value_col].abs()
fig6 = px.scatter(bubble, x=year_col, y=value_col, size="__size", color=group_col,
                  title="6) Bubble — Value with magnitude as bubble size")

# 7) Histogram
fig7 = px.histogram(df_f, x=value_col, color=group_col, nbins=30,
                    title="7) Histogram — Distribution of values")

# 8) Box plot
fig8 = px.box(df_f, x=group_col, y=value_col,
              title="8) Box — Dispersion by GeoArea")

# 9) Heatmap (year x group)
pivot = df_f.pivot_table(index=group_col, columns=year_col, values=value_col, aggfunc="mean")
heat = pivot.copy()
fig9 = px.imshow(heat, aspect="auto", title="9) Heatmap — Mean value (GeoArea × Year)")

# 10) Treemap (latest year composition)
tree = latest_df.groupby(group_col, as_index=False)[value_col].mean()
fig10 = px.treemap(tree, path=[group_col], values=value_col,
                   title=f"10) Treemap — Latest year ({latest_year}) share (mean if multiple obs)")

tabs = st.tabs(["1 Line", "2 Area", "3 Bar", "4 Stacked bar", "5 Scatter", "6 Bubble", "7 Histogram", "8 Box", "9 Heatmap", "10 Treemap"])

for t, fig in zip(tabs, [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]):
    with t:
        st.plotly_chart(fig, use_container_width=True)

# Download cleaned dataset
st.download_button(
    "Download cleaned dataset (CSV)",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dashboard_data.csv",
    mime="text/csv",
)

st.caption(
    "Tip: For true financial statements (income statement, balance sheet, cashflow), upload your internal GL/ERP exports, "
    "and use SDG indicators as external macro/risk context."
)
