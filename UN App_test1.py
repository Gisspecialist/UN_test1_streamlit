import re
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from dateutil.parser import parse as dtparse

st.set_page_config(page_title="UN Finance & Econ Dashboard (UNSD SDG API v5)", layout="wide")

UNSD_V5_BASE = "https://unstats.un.org/sdgs/UNSDGAPIV5/v1/sdg"

# -----------------------------
# Robust helpers
# -----------------------------
def _safe_get_json(url: str, params: dict | None = None, timeout: int = 60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _rows_from_json(j):
    # API sometimes returns {"data":[...]} or a direct list
    if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
        return j["data"]
    if isinstance(j, list):
        return j
    for k in ["Data", "result", "Results"]:
        if isinstance(j, dict) and k in j and isinstance(j[k], list):
            return j[k]
    return []

def _pick_col(df: pd.DataFrame, candidates: list[str]):
    """Pick a column name by exact/lower match first, then fuzzy contains."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def make_hashable_for_uniques(s: pd.Series) -> pd.Series:
    """
    Convert unhashable objects (list/dict/set/tuple) into stable string representations
    so nunique/unique won't crash.
    """
    def _fix(v):
        if isinstance(v, (list, dict, set, tuple)):
            return str(v)
        return v
    return s.map(_fix)

def _coerce_year(series: pd.Series) -> pd.Series:
    """
    Extract a year from numeric, or strings containing YYYY, or date-like strings.
    """
    y = pd.to_numeric(series, errors="coerce")
    if y.notna().sum() > 0:
        return y

    def extract_year(v):
        if pd.isna(v):
            return np.nan
        s = str(v)

        # direct YYYY
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            return float(m.group(0))

        # try parse date
        try:
            dt = dtparse(s, fuzzy=True)
            return float(dt.year)
        except Exception:
            return np.nan

    return series.apply(extract_year)

def compute_kpis(df: pd.DataFrame, year_col: str, value_col: str, group_col: str):
    """
    KPI table by group:
    - latest value
    - YoY %
    - approx 5-year CAGR
    - volatility (std of pct changes)
    """
    rows = []
    for g, sub in df.dropna(subset=[year_col, value_col]).groupby(group_col):
        sub = sub.sort_values(year_col)
        if sub.empty:
            continue

        latest = sub.iloc[-1]
        latest_year = latest[year_col]
        latest_val = latest[value_col]

        yoy = np.nan
        if len(sub) >= 2:
            prev = sub.iloc[-2][value_col]
            if pd.notna(prev) and prev != 0:
                yoy = (latest_val - prev) / abs(prev) * 100.0

        cagr = np.nan
        if len(sub) >= 2:
            start = sub.iloc[max(0, len(sub) - 6)]
            years = latest_year - start[year_col]
            if pd.notna(years) and years > 0 and pd.notna(start[value_col]) and start[value_col] != 0:
                cagr = ((latest_val / start[value_col]) ** (1.0 / years) - 1.0) * 100.0

        vol = np.nan
        pct = sub[value_col].pct_change()
        if pct.notna().sum() >= 2:
            vol = pct.std() * 100.0

        rows.append(
            {
                group_col: g,
                "Latest year": int(latest_year) if pd.notna(latest_year) else latest_year,
                "Latest value": latest_val,
                "YoY %": yoy,
                "CAGR % (≈5y)": cagr,
                "Volatility % (pctchg std)": vol,
                "Obs count": int(sub[value_col].notna().sum()),
            }
        )
    return pd.DataFrame(rows)

def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))
    raise ValueError("Unsupported file type. Upload CSV or Excel.")

# -----------------------------
# UNSD SDG API
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_geoareas():
    url = f"{UNSD_V5_BASE}/GeoArea/List"
    j = _safe_get_json(url)
    rows = _rows_from_json(j)
    # flatten in case nested
    return pd.json_normalize(rows, sep=".")

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_indicators():
    url = f"{UNSD_V5_BASE}/Indicator/List"
    j = _safe_get_json(url)
    rows = _rows_from_json(j)
    return pd.json_normalize(rows, sep=".")

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_indicator_data(indicator_code: str, area_codes: list[str], page_size: int = 10000):
    url = f"{UNSD_V5_BASE}/Indicator/Data"
    frames = []
    for ac in area_codes:
        params = {"indicator": indicator_code, "areaCode": ac, "pageSize": page_size}
        j = _safe_get_json(url, params=params)

        rows = _rows_from_json(j)
        if not rows:
            continue

        # FLATTEN nested objects (prevents unhashable dict/list columns)
        df = pd.json_normalize(rows, sep=".")
        df["__areaCode_req"] = ac
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out

# -----------------------------
# Dataset normalization
# -----------------------------
def normalize_dataset(df: pd.DataFrame):
    """
    Produces:
      __year   int
      __value  float
      __group  str (geoAreaName)
    Returns cleaned df + column keys.
    """
    if df.empty:
        return df, None, None, None

    year_col_guess = _pick_col(df, [
        "timePeriodStart", "timePeriod", "year", "TimePeriodStart", "Year",
        "timePeriodStart.value", "timePeriod.value"
    ])
    value_col_guess = _pick_col(df, [
        "value", "Value", "obsValue", "ObservationValue",
        "value.value"
    ])
    area_name_guess = _pick_col(df, [
        "geoAreaName", "GeoAreaName", "areaName", "Country", "country",
        "geoAreaName.value"
    ])
    area_code_guess = _pick_col(df, [
        "geoAreaCode", "GeoAreaCode", "areaCode", "__areaCode_req",
        "geoAreaCode.value"
    ])

    if area_name_guess is None and area_code_guess is not None:
        df["geoAreaName"] = df[area_code_guess].astype(str)
        area_name_guess = "geoAreaName"

    if year_col_guess is None:
        maybe_time = [c for c in df.columns if any(k in c.lower() for k in ["time", "period", "year"])]
        year_col_guess = maybe_time[0] if maybe_time else None

    if value_col_guess is None:
        maybe_val = [c for c in df.columns if "value" in c.lower()]
        value_col_guess = maybe_val[0] if maybe_val else None

    # Build normalized columns
    df2 = df.copy()

    if year_col_guess is not None:
        df2["__year"] = _coerce_year(df2[year_col_guess])
    else:
        df2["__year"] = np.nan

    if value_col_guess is not None:
        df2["__value"] = pd.to_numeric(df2[value_col_guess], errors="coerce")
    else:
        df2["__value"] = np.nan

    if area_name_guess is not None:
        df2["__group"] = df2[area_name_guess].astype(str)
    elif area_code_guess is not None:
        df2["__group"] = df2[area_code_guess].astype(str)
    else:
        df2["__group"] = "All"

    cleaned = df2.dropna(subset=["__year", "__value"]).copy()
    cleaned["__year"] = cleaned["__year"].astype(int)

    return cleaned, "__year", "__value", "__group"

# -----------------------------
# UI
# -----------------------------
st.title("UN Office for Partnerships — Dashboard (UNSD SDG API v5 + Upload)")
st.caption(
    "This dashboard consumes UNSD SDG indicator time series (and/or uploads) and renders 10 chart types safely."
)

source_mode = st.sidebar.radio("Data source", ["UNSD SDG API v5", "Upload CSV/XLSX"], index=0)

df_raw = pd.DataFrame()

if source_mode == "UNSD SDG API v5":
    st.sidebar.subheader("API selectors")

    with st.spinner("Loading GeoAreas + Indicators..."):
        geo_df = fetch_geoareas()
        ind_df = fetch_indicators()

    # geo columns
    geo_code = _pick_col(geo_df, ["geoAreaCode", "GeoAreaCode", "areaCode", "geoAreaCode.value"])
    geo_name = _pick_col(geo_df, ["geoAreaName", "GeoAreaName", "areaName", "geoAreaName.value"])
    if geo_code is None or geo_name is None:
        st.error("Could not detect GeoArea code/name columns from API response.")
        st.dataframe(geo_df.head(50), use_container_width=True)
        st.stop()

    # indicator columns
    ind_code = _pick_col(ind_df, ["code", "indicator", "indicatorCode", "Code"])
    ind_desc = _pick_col(ind_df, ["description", "indicatorDescription", "name", "title", "Title"])
    if ind_code is None:
        st.error("Could not detect Indicator code column from API response.")
        st.dataframe(ind_df.head(50), use_container_width=True)
        st.stop()

    # labels
    geo_df["_label"] = geo_df[geo_name].astype(str) + " (M49 " + geo_df[geo_code].astype(str) + ")"
    geo_options = geo_df.sort_values(geo_name)["_label"].tolist()

    if ind_desc is not None:
        ind_df["_label"] = ind_df[ind_code].astype(str) + " — " + ind_df[ind_desc].astype(str)
    else:
        ind_df["_label"] = ind_df[ind_code].astype(str)
    ind_options = ind_df.sort_values(ind_code)["_label"].tolist()

    indicator_label = st.sidebar.selectbox("Indicator", ind_options, index=0)
    indicator_code = indicator_label.split(" — ")[0].strip()

    chosen_geo = st.sidebar.multiselect(
        "GeoAreas (countries/regions)", geo_options, default=geo_options[:1]
    )
    area_codes = [re.search(r"\(M49\s+([0-9]+)\)", x).group(1) for x in chosen_geo] if chosen_geo else []

    page_size = st.sidebar.slider("Page size", min_value=1000, max_value=20000, value=10000, step=1000)

    fetch = st.sidebar.button("Fetch data", type="primary")
    if fetch and area_codes:
        with st.spinner("Fetching indicator observations..."):
            df_raw = fetch_indicator_data(indicator_code, area_codes, page_size=page_size)

else:
    st.sidebar.subheader("Upload dataset")
    up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if up is not None:
        try:
            df_raw = read_uploaded_table(up)
        except Exception as e:
            st.error(f"Failed to read upload: {e}")
            st.stop()

if df_raw.empty:
    st.info("Select indicator + GeoAreas and click **Fetch data**, or upload a dataset.")
    st.stop()

# Normalize
df, year_col, value_col, group_col = normalize_dataset(df_raw)
if df.empty:
    st.warning("No numeric time/value observations found after cleaning. Try another indicator/GeoArea or dataset.")
    with st.expander("Show raw preview"):
        st.dataframe(df_raw.head(100), use_container_width=True)
    st.stop()

# -----------------------------
# Diagnostics (optional)
# -----------------------------
with st.expander("Diagnostics (optional)"):
    bad_cols = []
    for c in df_raw.columns:
        if df_raw[c].dtype == "object":
            sample = df_raw[c].dropna().head(50).tolist()
            if any(isinstance(v, (list, dict, set, tuple)) for v in sample):
                bad_cols.append(c)
    st.write("Columns in raw data containing unhashable objects (lists/dicts/sets/tuples):")
    st.write(bad_cols if bad_cols else "None detected")
    st.write("Detected plotting columns:", {"year_col": year_col, "value_col": value_col, "group_col": group_col})

# -----------------------------
# SAFE filter discovery (all patches applied)
# -----------------------------
st.sidebar.subheader("Filters (auto-detected)")
cat_cols = []

for c in df.columns:
    if c.startswith("__"):
        continue

    if df[c].dtype == "object":
        s = make_hashable_for_uniques(df[c])

        # Skip columns that still contain nested junk or are too sparse
        # (also helps performance)
        non_null = s.dropna()
        if non_null.empty:
            continue

        # Compute nunique safely
        try:
            nunique = non_null.nunique(dropna=True)
        except TypeError:
            nunique = non_null.astype(str).nunique(dropna=True)

        if 2 <= nunique <= 30:
            cat_cols.append(c)

selected_filters = {}
MAX_OPTIONS = 2000
MAX_FILTER_COLS = 6

for c in cat_cols[:MAX_FILTER_COLS]:
    s = make_hashable_for_uniques(df[c]).dropna().astype(str)

    # Cap options (prevents huge dropdowns)
    opts = sorted(s.unique().tolist())
    if len(opts) > MAX_OPTIONS:
        opts = opts[:MAX_OPTIONS]

    chosen = st.sidebar.multiselect(f"{c}", opts, default=[])
    if chosen:
        selected_filters[c] = set(chosen)

df_f = df.copy()
for c, chosen in selected_filters.items():
    s = make_hashable_for_uniques(df_f[c]).astype(str)
    df_f = df_f[s.isin(chosen)]

if df_f.empty:
    st.warning("Filters removed all data. Relax filters.")
    st.stop()

# -----------------------------
# KPI section
# -----------------------------
kpi_df = compute_kpis(df_f, year_col, value_col, group_col)

k1, k2 = st.columns([2, 3], gap="large")
with k1:
    st.subheader("Key metrics (per GeoArea)")
    st.dataframe(
        kpi_df.sort_values(["Latest year", "Latest value"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True
    )
with k2:
    st.subheader("Observations (sample)")
    st.dataframe(
        df_f[[group_col, year_col, value_col]].sort_values([group_col, year_col]).head(300),
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# 10 charts dashboard
# -----------------------------
st.subheader("Dashboard — 10 chart types")

latest_year = int(df_f[year_col].max())
latest_df = df_f[df_f[year_col] == latest_year].copy()

# Shared aggregates
agg_year_group = df_f.groupby([year_col, group_col], as_index=False)[value_col].mean()

# 1) Line
fig1 = px.line(df_f.sort_values(year_col), x=year_col, y=value_col, color=group_col, markers=True,
               title="1) Line — Trend over time")

# 2) Area
fig2 = px.area(df_f.sort_values(year_col), x=year_col, y=value_col, color=group_col,
               title="2) Area — Cumulative movement over time")

# 3) Bar (latest year)
fig3 = px.bar(latest_df, x=group_col, y=value_col,
              title=f"3) Bar — Latest year ({latest_year}) comparison")

# 4) Stacked bar (year on x, group stacked)
fig4 = px.bar(agg_year_group, x=year_col, y=value_col, color=group_col, barmode="stack",
              title="4) Stacked bar — Composition across years (mean if multiple obs)")

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
fig8 = px.box(df_f, x=group_col, y=value_col, title="8) Box — Dispersion by GeoArea")

# 9) Heatmap (GeoArea × Year)
pivot = df_f.pivot_table(index=group_col, columns=year_col, values=value_col, aggfunc="mean")
fig9 = px.imshow(pivot, aspect="auto", title="9) Heatmap — Mean value (GeoArea × Year)")

# 10) Treemap (latest year)
tree = latest_df.groupby(group_col, as_index=False)[value_col].mean()
fig10 = px.treemap(tree, path=[group_col], values=value_col,
                   title=f"10) Treemap — Latest year ({latest_year}) share (mean if multiple obs)")

tabs = st.tabs([
    "1 Line", "2 Area", "3 Bar", "4 Stacked bar", "5 Scatter",
    "6 Bubble", "7 Histogram", "8 Box", "9 Heatmap", "10 Treemap"
])

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
    "If you upload internal UN finance exports (budget, commitments, obligations, expenditures), "
    "the same 10-chart dashboard will work. SDG indicators can provide macro/impact context."
)
