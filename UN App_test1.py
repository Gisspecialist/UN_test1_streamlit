import re
import io
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from dateutil.parser import parse as dtparse

st.set_page_config(page_title="UN Dashboard (UNSD SDG API v5 + Upload)", layout="wide")

# UNSDGAPIV5 swagger reference: https://unstats.un.org/sdgs/UNSDGAPIV5/swagger/index.html
UNSD_V5_BASE = "https://unstats.un.org/sdgs/UNSDGAPIV5"
UNSD_V5_API = f"{UNSD_V5_BASE}/v1/sdg"

# Legacy SDG API path (still commonly deployed)
UNSD_V1_BASE = "https://unstats.un.org/SDGAPI"
UNSD_V1_API = f"{UNSD_V1_BASE}/v1/sdg"

# -----------------------------
# ✅ STREAMLIT CLOUD SAFE WRITABLE DIRS
# -----------------------------
RUNTIME_DIR = Path(tempfile.gettempdir()) / "un_sdg_dashboard"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = RUNTIME_DIR / ".cache_sdg"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Robust helpers
# -----------------------------
def _safe_get_json(url: str, params: dict | None = None, timeout: int = 60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _rows_from_json(j):
    if isinstance(j, dict) and "data" in j and isinstance(j["data"], list):
        return j["data"]
    if isinstance(j, list):
        return j
    for k in ["Data", "result", "Results"]:
        if isinstance(j, dict) and k in j and isinstance(j[k], list):
            return j[k]
    return []

def _pick_col(df: pd.DataFrame, candidates: list[str]):
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
    def _fix(v):
        if isinstance(v, (list, dict, set, tuple)):
            return str(v)
        return v
    return s.map(_fix)

def _coerce_year(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    if y.notna().sum() > 0:
        return y

    def extract_year(v):
        if pd.isna(v):
            return np.nan
        s = str(v)
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            return float(m.group(0))
        try:
            dt = dtparse(s, fuzzy=True)
            return float(dt.year)
        except Exception:
            return np.nan

    return series.apply(extract_year)

def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))
    raise ValueError("Unsupported file type. Upload CSV or Excel.")

# -----------------------------
# Disk cache helpers
# -----------------------------
def stable_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]

def cache_paths(key: str):
    return (CACHE_DIR / f"{key}.parquet", CACHE_DIR / f"{key}.csv", CACHE_DIR / f"{key}.json")

def cache_read(key: str) -> pd.DataFrame | None:
    p_parq, _, _ = cache_paths(key)
    if p_parq.exists():
        try:
            return pd.read_parquet(p_parq)
        except Exception:
            return None
    return None

def cache_write(key: str, df: pd.DataFrame, meta: dict):
    p_parq, p_csv, p_meta = cache_paths(key)
    try:
        df.to_parquet(p_parq, index=False)
    except Exception:
        pass
    try:
        df.to_csv(p_csv, index=False)
    except Exception:
        pass
    try:
        p_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def list_cache_items(limit: int = 30):
    items = []
    for meta_file in sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            key = meta_file.stem
            p_parq, p_csv, _ = cache_paths(key)
            items.append({
                "key": key,
                "type": meta.get("type"),
                "label": meta.get("label"),
                "created": meta.get("created"),
                "rows": meta.get("rows"),
                "parquet": str(p_parq) if p_parq.exists() else "",
                "csv": str(p_csv) if p_csv.exists() else "",
            })
        except Exception:
            continue
        if len(items) >= limit:
            break
    return items

def clear_cache():
    for p in CACHE_DIR.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass

# -----------------------------
# Endpoint fallback utilities
# -----------------------------
def try_get_first_json(paths: list[tuple[str, dict | None]]):
    last_err = None
    for url, params in paths:
        try:
            return _safe_get_json(url, params=params)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("No endpoints succeeded.")

def json_to_df(rows) -> pd.DataFrame:
    return pd.json_normalize(rows, sep=".")

# -----------------------------
# UNSD SDG API: lists + data
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_geoareas():
    j = try_get_first_json([
        (f"{UNSD_V5_API}/GeoArea/List", None),
        (f"{UNSD_V1_API}/GeoArea/List", None),
    ])
    return json_to_df(_rows_from_json(j))

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_goals():
    j = try_get_first_json([
        (f"{UNSD_V5_API}/Goal/List", None),
        (f"{UNSD_V1_API}/Goal/List", None),
    ])
    return json_to_df(_rows_from_json(j))

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_targets(goal_code: str):
    j = try_get_first_json([
        (f"{UNSD_V5_API}/Goal/{goal_code}/Target/List", None),
        (f"{UNSD_V1_API}/Goal/{goal_code}/Target/List", None),
        (f"{UNSD_V5_API}/Target/List", {"goalCode": goal_code}),
        (f"{UNSD_V1_API}/Target/List", {"goalCode": goal_code}),
    ])
    return json_to_df(_rows_from_json(j))

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_indicators_for_target(target_code: str):
    j = try_get_first_json([
        (f"{UNSD_V5_API}/Target/{target_code}/Indicator/List", None),
        (f"{UNSD_V1_API}/Target/{target_code}/Indicator/List", None),
        (f"{UNSD_V5_API}/Indicator/List", {"targetCode": target_code}),
        (f"{UNSD_V1_API}/Indicator/List", {"targetCode": target_code}),
    ])
    return json_to_df(_rows_from_json(j))

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_all_indicators():
    j = try_get_first_json([
        (f"{UNSD_V5_API}/Indicator/List", None),
        (f"{UNSD_V1_API}/Indicator/List", None),
    ])
    return json_to_df(_rows_from_json(j))

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_series_for_indicator(indicator_code: str):
    j = try_get_first_json([
        (f"{UNSD_V5_API}/Indicator/{indicator_code}/Series/List", None),
        (f"{UNSD_V1_API}/Indicator/{indicator_code}/Series/List", None),
        (f"{UNSD_V5_API}/Series/List", {"indicator": indicator_code}),
        (f"{UNSD_V1_API}/Series/List", {"indicator": indicator_code}),
    ])
    return json_to_df(_rows_from_json(j))

def _fetch_series_data_no_cache(series_code: str, area_codes: list[str], page_size: int = 10000):
    frames = []
    for ac in area_codes:
        j = try_get_first_json([
            (f"{UNSD_V5_API}/Series/Data", {"seriesCode": series_code, "areaCode": ac, "pageSize": page_size}),
            (f"{UNSD_V1_API}/Series/Data", {"seriesCode": series_code, "areaCode": ac, "pageSize": page_size}),
        ])
        rows = _rows_from_json(j)
        if not rows:
            continue
        df = json_to_df(rows)
        df["__areaCode_req"] = ac
        df["__seriesCode_req"] = series_code
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _fetch_indicator_data_no_cache(indicator_code: str, area_codes: list[str], page_size: int = 10000):
    frames = []
    for ac in area_codes:
        j = try_get_first_json([
            (f"{UNSD_V5_API}/Indicator/Data", {"indicator": indicator_code, "areaCode": ac, "pageSize": page_size}),
            (f"{UNSD_V1_API}/Indicator/Data", {"indicator": indicator_code, "areaCode": ac, "pageSize": page_size}),
        ])
        rows = _rows_from_json(j)
        if not rows:
            continue
        df = json_to_df(rows)
        df["__areaCode_req"] = ac
        df["__indicator_req"] = indicator_code
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fetch_series_data(series_code: str, area_codes: list[str], page_size: int, disk_cache: bool, label: str):
    payload = {"type": "series", "seriesCode": series_code, "areaCodes": sorted(area_codes), "pageSize": page_size}
    key = stable_key(payload)
    if disk_cache:
        cached = cache_read(key)
        if cached is not None and not cached.empty:
            return cached, key, True
    df = _fetch_series_data_no_cache(series_code, area_codes, page_size=page_size)
    if disk_cache and not df.empty:
        meta = {"type": "series", "label": label, "created": pd.Timestamp.utcnow().isoformat(), "rows": int(df.shape[0]), **payload}
        cache_write(key, df, meta)
    return df, key, False

def fetch_indicator_data(indicator_code: str, area_codes: list[str], page_size: int, disk_cache: bool, label: str):
    payload = {"type": "indicator", "indicator": indicator_code, "areaCodes": sorted(area_codes), "pageSize": page_size}
    key = stable_key(payload)
    if disk_cache:
        cached = cache_read(key)
        if cached is not None and not cached.empty:
            return cached, key, True
    df = _fetch_indicator_data_no_cache(indicator_code, area_codes, page_size=page_size)
    if disk_cache and not df.empty:
        meta = {"type": "indicator", "label": label, "created": pd.Timestamp.utcnow().isoformat(), "rows": int(df.shape[0]), **payload}
        cache_write(key, df, meta)
    return df, key, False

# -----------------------------
# Dataset normalization for charting
# -----------------------------
def normalize_dataset(df: pd.DataFrame):
    if df.empty:
        return df, None, None, None

    year_col_guess = _pick_col(df, [
        "timePeriodStart", "timePeriod", "year", "Year",
        "timePeriodStart.value", "timePeriod.value"
    ])
    value_col_guess = _pick_col(df, [
        "value", "Value", "obsValue", "ObservationValue",
        "value.value"
    ])
    area_name_guess = _pick_col(df, [
        "geoAreaName", "GeoAreaName", "geoAreaName.value",
        "areaName", "Country", "country"
    ])
    area_code_guess = _pick_col(df, [
        "geoAreaCode", "GeoAreaCode", "geoAreaCode.value",
        "areaCode", "__areaCode_req"
    ])

    df2 = df.copy()

    if area_name_guess is None and area_code_guess is not None:
        df2["geoAreaName"] = df2[area_code_guess].astype(str)
        area_name_guess = "geoAreaName"

    if year_col_guess is None:
        maybe_time = [c for c in df2.columns if any(k in c.lower() for k in ["time", "period", "year"])]
        year_col_guess = maybe_time[0] if maybe_time else None

    if value_col_guess is None:
        maybe_val = [c for c in df2.columns if "value" in c.lower()]
        value_col_guess = maybe_val[0] if maybe_val else None

    df2["__year"] = _coerce_year(df2[year_col_guess]) if year_col_guess else np.nan
    df2["__value"] = pd.to_numeric(df2[value_col_guess], errors="coerce") if value_col_guess else np.nan
    df2["__group"] = df2[area_name_guess].astype(str) if area_name_guess else (
        df2[area_code_guess].astype(str) if area_code_guess else "All"
    )

    cleaned = df2.dropna(subset=["__year", "__value"]).copy()
    cleaned["__year"] = cleaned["__year"].astype(int)

    return cleaned, "__year", "__value", "__group"

# -----------------------------
# KPI
# -----------------------------
def compute_kpis(df: pd.DataFrame, year_col: str, value_col: str, group_col: str):
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

        rows.append({
            group_col: g,
            "Latest year": int(latest_year) if pd.notna(latest_year) else latest_year,
            "Latest value": latest_val,
            "YoY %": yoy,
            "CAGR % (≈5y)": cagr,
            "Volatility %": vol,
            "Obs count": int(sub[value_col].notna().sum()),
        })
    return pd.DataFrame(rows)

# -----------------------------
# UI
# -----------------------------
st.title("UN Office for Partnerships — SDG Data Explorer + Finance Dashboard")
st.caption(
    "Browse SDG catalog (Goal→Target→Indicator→Series), apply disaggregation filters, render 10 chart types, "
    "and cache datasets safely on Streamlit Cloud."
)

mode = st.sidebar.radio("Mode", ["Browse SDG Catalog", "Quick Indicator Fetch", "Upload CSV/XLSX"], index=0, key="mode")

# Disk cache controls
st.sidebar.subheader("Performance")
disk_cache = st.sidebar.toggle("Enable disk cache (Parquet/CSV)", value=True, key="disk_cache")

with st.sidebar.expander("Cache utilities"):
    if st.button("Clear disk cache", key="clear_cache_btn"):
        clear_cache()
        st.success("Cache cleared.")
    items = list_cache_items(limit=15)
    st.caption(f"Cache directory: {str(CACHE_DIR)}")
    if items:
        st.caption("Recent cached datasets")
        st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
    else:
        st.caption("No cached datasets yet.")

df_raw = pd.DataFrame()
meta_note = ""
cache_key = ""
cache_hit = False

if mode == "Browse SDG Catalog":
    st.sidebar.subheader("Catalog Browser")

    with st.spinner("Loading Goals + GeoAreas..."):
        goals = fetch_goals()
        geo = fetch_geoareas()

    goal_code = _pick_col(goals, ["goal", "goalCode", "code", "Goal"])
    goal_title = _pick_col(goals, ["title", "description", "goalTitle", "name"])
    if goal_code is None:
        st.error("Could not detect goal code in Goal/List response.")
        st.dataframe(goals.head(50), use_container_width=True)
        st.stop()

    goals["_label"] = goals[goal_code].astype(str) + (" — " + goals[goal_title].astype(str) if goal_title else "")
    goal_choice = st.sidebar.selectbox("Goal", goals["_label"].tolist(), index=0, key="goal_choice")
    chosen_goal_code = goal_choice.split(" — ")[0].strip()

    with st.spinner("Loading Targets..."):
        targets = fetch_targets(chosen_goal_code)

    target_code = _pick_col(targets, ["target", "targetCode", "code"])
    target_title = _pick_col(targets, ["title", "description", "name"])
    if target_code is None:
        st.error("Could not detect target code in Target/List response.")
        st.dataframe(targets.head(50), use_container_width=True)
        st.stop()

    targets["_label"] = targets[target_code].astype(str) + (" — " + targets[target_title].astype(str) if target_title else "")
    target_choice = st.sidebar.selectbox("Target", targets["_label"].tolist(), index=0, key="target_choice")
    chosen_target_code = target_choice.split(" — ")[0].strip()

    with st.spinner("Loading Indicators..."):
        inds = fetch_indicators_for_target(chosen_target_code)

    ind_code = _pick_col(inds, ["code", "indicator", "indicatorCode"])
    ind_desc = _pick_col(inds, ["description", "indicatorDescription", "name", "title"])
    if ind_code is None:
        st.error("Could not detect indicator code in Indicator list response.")
        st.dataframe(inds.head(50), use_container_width=True)
        st.stop()

    inds["_label"] = inds[ind_code].astype(str) + (" — " + inds[ind_desc].astype(str) if ind_desc else "")
    ind_choice = st.sidebar.selectbox("Indicator", inds["_label"].tolist(), index=0, key="ind_choice")
    chosen_ind_code = ind_choice.split(" — ")[0].strip()

    with st.spinner("Loading Series..."):
        series = fetch_series_for_indicator(chosen_ind_code)

    series_code = _pick_col(series, ["seriesCode", "series", "code"])
    series_desc = _pick_col(series, ["seriesDescription", "description", "name", "title"])

    if series_code is None:
        st.sidebar.warning("Series list not available for this indicator. Will use Indicator/Data.")
        chosen_series_code = None
        series_label = ""
    else:
        series["_label"] = series[series_code].astype(str) + (" — " + series[series_desc].astype(str) if series_desc else "")
        s_choice = st.sidebar.selectbox("Series", series["_label"].tolist(), index=0, key="series_choice")
        chosen_series_code = s_choice.split(" — ")[0].strip()
        series_label = s_choice

    geo_code = _pick_col(geo, ["geoAreaCode", "GeoAreaCode", "areaCode", "geoAreaCode.value"])
    geo_name = _pick_col(geo, ["geoAreaName", "GeoAreaName", "areaName", "geoAreaName.value"])
    if geo_code is None or geo_name is None:
        st.error("Could not detect GeoArea code/name.")
        st.dataframe(geo.head(50), use_container_width=True)
        st.stop()

    geo["_label"] = geo[geo_name].astype(str) + " (M49 " + geo[geo_code].astype(str) + ")"
    geo_options = geo.sort_values(geo_name)["_label"].tolist()

    selected_geo = st.sidebar.multiselect(
        "GeoAreas", geo_options, default=geo_options[:2], key=f"geoareas_{mode}"
    )
    area_codes = [re.search(r"\(M49\s+([0-9]+)\)", x).group(1) for x in selected_geo] if selected_geo else []

    page_size = st.sidebar.slider("Page size", 1000, 20000, 10000, 1000, key="page_size")
    run = st.sidebar.button("Fetch Data", type="primary", key="fetch_data_btn")

    if run and area_codes:
        with st.spinner("Fetching SDG observations..."):
            if chosen_series_code:
                df_raw, cache_key, cache_hit = fetch_series_data(
                    chosen_series_code, area_codes, page_size, disk_cache,
                    label=series_label or f"Series {chosen_series_code}"
                )
                meta_note = f"Source: Series/Data | seriesCode={chosen_series_code} | cache={'HIT' if cache_hit else 'MISS'} | key={cache_key}"
            else:
                df_raw, cache_key, cache_hit = fetch_indicator_data(
                    chosen_ind_code, area_codes, page_size, disk_cache, label=ind_choice
                )
                meta_note = f"Source: Indicator/Data | indicator={chosen_ind_code} | cache={'HIT' if cache_hit else 'MISS'} | key={cache_key}"

elif mode == "Quick Indicator Fetch":
    st.sidebar.subheader("Indicator Fetch")

    with st.spinner("Loading GeoAreas + Indicators..."):
        geo = fetch_geoareas()
        inds = fetch_all_indicators()

    ind_code = _pick_col(inds, ["code", "indicator", "indicatorCode"])
    ind_desc = _pick_col(inds, ["description", "indicatorDescription", "name", "title"])
    if ind_code is None:
        st.error("Could not detect indicator code in Indicator/List response.")
        st.dataframe(inds.head(50), use_container_width=True)
        st.stop()

    inds["_label"] = inds[ind_code].astype(str) + (" — " + inds[ind_desc].astype(str) if ind_desc else "")
    ind_choice = st.sidebar.selectbox("Indicator", inds["_label"].tolist(), index=0, key="quick_ind_choice")
    chosen_ind_code = ind_choice.split(" — ")[0].strip()

    geo_code = _pick_col(geo, ["geoAreaCode", "GeoAreaCode", "areaCode", "geoAreaCode.value"])
    geo_name = _pick_col(geo, ["geoAreaName", "GeoAreaName", "areaName", "geoAreaName.value"])

    geo["_label"] = geo[geo_name].astype(str) + " (M49 " + geo[geo_code].astype(str) + ")"
    geo_options = geo.sort_values(geo_name)["_label"].tolist()

    selected_geo = st.sidebar.multiselect(
        "GeoAreas", geo_options, default=geo_options[:2], key=f"geoareas_{mode}"
    )
    area_codes = [re.search(r"\(M49\s+([0-9]+)\)", x).group(1) for x in selected_geo] if selected_geo else []

    page_size = st.sidebar.slider("Page size", 1000, 20000, 10000, 1000, key="quick_page_size")
    run = st.sidebar.button("Fetch Data", type="primary", key="quick_fetch_btn")

    if run and area_codes:
        with st.spinner("Fetching SDG observations..."):
            df_raw, cache_key, cache_hit = fetch_indicator_data(
                chosen_ind_code, area_codes, page_size, disk_cache, label=ind_choice
            )
            meta_note = f"Source: Indicator/Data | indicator={chosen_ind_code} | cache={'HIT' if cache_hit else 'MISS'} | key={cache_key}"

else:
    st.sidebar.subheader("Upload dataset")
    up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="uploader")
    if up is not None:
        df_raw = read_uploaded_table(up)
        meta_note = f"Source: Upload | file={up.name}"

if df_raw.empty:
    st.info("Select items and click **Fetch Data**, or upload a dataset to begin.")
    st.stop()

# -----------------------------
# Diagnostics / Raw preview
# -----------------------------
with st.expander("Diagnostics / Raw preview (optional)"):
    st.write(meta_note)
    st.dataframe(df_raw.head(200), use_container_width=True)

# -----------------------------
# ✅ Export raw dataset (CSV + Parquet in-memory)
# -----------------------------
st.sidebar.subheader("Export")
st.sidebar.download_button(
    "Download raw as CSV",
    df_raw.to_csv(index=False).encode("utf-8"),
    file_name="sdg_raw_export.csv",
    mime="text/csv",
    key="dl_raw_csv",
)

try:
    buf = io.BytesIO()
    df_raw.to_parquet(buf, index=False)
    st.sidebar.download_button(
        "Download raw as Parquet",
        buf.getvalue(),
        file_name="sdg_raw_export.parquet",
        mime="application/octet-stream",
        key="dl_raw_parquet",
    )
except Exception:
    st.sidebar.caption("Parquet export requires pyarrow (kept in requirements.txt).")

# -----------------------------
# Disaggregation filters from RAW (auto) ✅ UNIQUE KEYS (PATCH)
# -----------------------------
st.sidebar.subheader("Disaggregation filters (auto)")
raw_cat_cols = []
for c in df_raw.columns:
    if str(c).startswith("__"):
        continue
    if df_raw[c].dtype == "object":
        s = make_hashable_for_uniques(df_raw[c]).dropna()
        if s.empty:
            continue
        try:
            nunique = s.nunique()
        except TypeError:
            nunique = s.astype(str).nunique()
        if 2 <= nunique <= 60:
            raw_cat_cols.append(c)

MAX_DIM_FILTERS = 8
MAX_OPTIONS = 2500
chosen_dim_filters = {}

for c in raw_cat_cols[:MAX_DIM_FILTERS]:
    s = make_hashable_for_uniques(df_raw[c]).dropna().astype(str)
    opts = sorted(s.unique().tolist())
    if len(opts) > MAX_OPTIONS:
        opts = opts[:MAX_OPTIONS]

    chosen = st.sidebar.multiselect(
        label=str(c),
        options=opts,
        default=[],
        key=f"raw_dim_{mode}_{str(c)}"
    )
    if chosen:
        chosen_dim_filters[c] = set(chosen)

df_raw_f = df_raw.copy()
for c, chosen in chosen_dim_filters.items():
    s = make_hashable_for_uniques(df_raw_f[c]).astype(str)
    df_raw_f = df_raw_f[s.isin(chosen)]

if df_raw_f.empty:
    st.warning("Disaggregation filters removed all rows. Relax filters.")
    st.stop()

# Normalize for charting
df, year_col, value_col, group_col = normalize_dataset(df_raw_f)
if df.empty:
    st.warning("No numeric time/value observations found after cleaning. Try different series/indicator/filters.")
    st.stop()

# -----------------------------
# Extra filters (cleaned data) ✅ UNIQUE KEYS (PATCH)
# -----------------------------
st.sidebar.subheader("Extra filters (cleaned data)")
cat_cols = []
for c in df.columns:
    if str(c).startswith("__"):
        continue
    if df[c].dtype == "object":
        s = make_hashable_for_uniques(df[c]).dropna()
        if s.empty:
            continue
        try:
            nunique = s.nunique()
        except TypeError:
            nunique = s.astype(str).nunique()
        if 2 <= nunique <= 30:
            cat_cols.append(c)

selected_filters = {}
for c in cat_cols[:6]:
    s = make_hashable_for_uniques(df[c]).dropna().astype(str)
    opts = sorted(s.unique().tolist())
    if len(opts) > 2000:
        opts = opts[:2000]

    chosen = st.sidebar.multiselect(
        label=str(c),
        options=opts,
        default=[],
        key=f"clean_dim_{mode}_{str(c)}"
    )
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
# KPIs
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
agg_year_group = df_f.groupby([year_col, group_col], as_index=False)[value_col].mean()

fig1 = px.line(df_f.sort_values(year_col), x=year_col, y=value_col, color=group_col, markers=True,
               title="1) Line — Trend over time")

fig2 = px.area(df_f.sort_values(year_col), x=year_col, y=value_col, color=group_col,
               title="2) Area — Cumulative movement over time")

fig3 = px.bar(latest_df, x=group_col, y=value_col,
              title=f"3) Bar — Latest year ({latest_year}) comparison")

fig4 = px.bar(agg_year_group, x=year_col, y=value_col, color=group_col, barmode="stack",
              title="4) Stacked bar — Composition across years (mean if multiple obs)")

fig5 = px.scatter(df_f, x=year_col, y=value_col, color=group_col,
                  title="5) Scatter — Observations over time")

bubble = df_f.copy()
bubble["__size"] = bubble[value_col].abs()
fig6 = px.scatter(bubble, x=year_col, y=value_col, size="__size", color=group_col,
                  title="6) Bubble — Value with magnitude as bubble size")

fig7 = px.histogram(df_f, x=value_col, color=group_col, nbins=30,
                    title="7) Histogram — Distribution of values")

fig8 = px.box(df_f, x=group_col, y=value_col, title="8) Box — Dispersion by GeoArea")

pivot = df_f.pivot_table(index=group_col, columns=year_col, values=value_col, aggfunc="mean")
fig9 = px.imshow(pivot, aspect="auto", title="9) Heatmap — Mean value (GeoArea × Year)")

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

st.download_button(
    "Download cleaned dataset (CSV)",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dashboard_data.csv",
    mime="text/csv",
    key="dl_cleaned_csv",
)

st.caption(
    "✅ Streamlit Cloud safe: writes only to /tmp via tempfile.gettempdir(). "
    "✅ Widget IDs stabilized with unique keys for dynamic multiselect filters."
)
