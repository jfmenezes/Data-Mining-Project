"""
ACDC Members - Interactive EDA Dashboard
========================================
Lightweight Streamlit dashboard for stakeholders to explore customer attributes.

Covers the project guidelines:
  * Filtering        -> sidebar filters (plan, country, industry, gender, account type,
                        status, age, start year) that apply to every tab.
  * Drill-down       -> "Drill-down" tab: segment -> sub-segment -> member list.
  * Distributions    -> "Distributions" tab (histogram / box / violin) + overview charts.
  * Self-exploration -> every chart has selectors, so stakeholders can build their own views.

Run with:
    streamlit run app.py

The app looks for `acdc_dataset.csv` next to this file (or in ./Dataset, ../Dataset).
If it is not found, you can upload the CSV from the sidebar.
"""

import io
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------------------
# Page setup and constants (taken from the notebook)
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="ACDC Members - EDA Dashboard", page_icon="📊", layout="wide")

HERE = Path(__file__).resolve().parent
DATA_PATHS = [
    HERE / "acdc_dataset.csv",
    HERE / "Dataset" / "acdc_dataset.csv",
    HERE.parent / "Dataset" / "acdc_dataset.csv",
]

PALETTE = ["#ebc334", "#bd1041", "#76d5de", "#6250d6", "#48e348"]
TIER_ORDER = ["hot desk", "flex", "dedicated desk", "full time", "private office"]
TIER_COLOR = dict(zip(TIER_ORDER, PALETTE))
TIER_COLOR["missing"] = "#b0b0b0"
STATUS_COLOR = {"Active": "#2ca02c", "Churned": "#d62728"}
ACCOUNT_COLOR = {"Individual": "#EA9F09", "Team": "#87cefa"}
AGE_BINS = [0, 18, 25, 35, 45, 55, 65, np.inf]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
SNAPSHOT = pd.Timestamp("2023-12-31")  # data extraction date (same as the notebook)
MAX_HOURS = 3 * 365 * 24               # only 3 years of data -> more hours than this is impossible

CAT_LABELS = {
    "plan_tier": "Plan tier",
    "country": "Country",
    "industry": "Industry",
    "gender": "Gender",
    "account_type": "Account type",
    "status": "Membership status",
    "age_group": "Age group",
}
NUM_LABELS = {
    "age": "Age",
    "lifetime_value": "Lifetime value (€)",
    "value_per_month": "Value per month (€)",
    "avg_weekly_visits": "Average weekly visits",
    "total_visits": "Total visits",
    "n_renewals": "Number of renewals",
    "plan_days_per_week": "Plan days per week",
    "days_since_last_visit": "Days since last visit",
    "entitled_visits_last_period": "Entitled visits (last period)",
    "actual_visits_last_period": "Actual visits (last period)",
    "meeting_room_hours": "Meeting room hours",
    "event_space_hours": "Event space hours",
    "guest_passes_issued": "Guest passes issued",
    "n_colleagues_on_account": "Colleagues on account",
    "usage_rate": "Usage rate (weekly visits / plan days)",
    "usage_rate_last_period": "Plan utilisation, last period",
    "visit_trend": "Visit trend (recent vs historical)",
    "recency_vs_usual_gap": "Recency vs usual gap",
}
# Aggregations offered in the drill-down tab
AGG_METRICS = {
    "Median lifetime value (€)": "lifetime_value",
    "Median value per month (€)": "value_per_month",
    "Median weekly visits": "avg_weekly_visits",
    "Median plan utilisation": "usage_rate_last_period",
    "Median usage rate": "usage_rate",
    "Median visit trend": "visit_trend",
    "Median age": "age",
}

_chart_ids = itertools.count()


# --------------------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------------------
def show(fig, height=None):
    """Render a Plotly figure full-width (works across Streamlit versions)."""
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    if height:
        fig.update_layout(height=height)
    key = f"chart_{next(_chart_ids)}"
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def fmt(x, pattern="{:,.0f}"):
    return "n/a" if pd.isna(x) else pattern.format(x)


def cat_order(col, data):
    """Sensible category order for each categorical dimension."""
    present = list(data[col].dropna().unique())
    if col == "plan_tier":
        return [t for t in TIER_ORDER + ["missing"] if t in present]
    if col == "age_group":
        return [a for a in AGE_LABELS + ["missing"] if a in present]
    if col == "status":
        return [s for s in ["Active", "Churned"] if s in present]
    return list(data[col].value_counts().index)


def color_map(col):
    return {
        "plan_tier": TIER_COLOR,
        "status": STATUS_COLOR,
        "account_type": ACCOUNT_COLOR,
    }.get(col)


# --------------------------------------------------------------------------------------
# Data loading and cleaning (same logic as the notebook)
# --------------------------------------------------------------------------------------
def parse_mixed_dates(series):
    iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    european = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(european)


def read_csv_any_sep(source):
    """The project file uses ';' as separator; fall back to ',' just in case."""
    df = pd.read_csv(source, sep=";")
    if df.shape[1] == 1:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, sep=",")
    return df


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(source):
    """`source` is either a file path (str) or the raw bytes of an uploaded CSV."""
    raw = read_csv_any_sep(io.BytesIO(source) if isinstance(source, bytes) else source)
    df = raw.copy()

    # --- duplicates: exact duplicates + same member registered under different ids
    df = df.drop_duplicates()
    df = df[~df.drop(columns="member_id").duplicated(keep="first")].reset_index(drop=True)

    # --- categorical labels
    df["gender"] = (
        df["gender"].astype("string").str.strip().str.lower()
        .replace({"f": "female", "m": "male", "o": "other"})
    )
    industry_dict = {"realestate": "real estate", "it": "information technology", "tech": "information technology"}
    df["industry"] = (
        df["industry"].astype("string").str.strip().str.lower()
        .str.replace("/", " & ", regex=False)
        .str.replace(r"\band\b", "&", regex=True)
        .str.replace(r"[-_]", " ", regex=True)
        .str.replace(r"\s*&\s*", " & ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace(industry_dict)
    )
    country_dict = {
        "portucal": "portugal", "pt": "portugal", "de": "germany", "deutschland": "germany",
        "uk": "united kingdom", "u.k.": "united kingdom", "gb": "united kingdom", "ie": "ireland",
        "es": "spain", "españa": "spain", "espana": "spain", "fr": "france", "nl": "netherlands",
        "the netherlands": "netherlands", "holland": "netherlands",
    }
    df["country"] = df["country"].astype("string").str.strip().str.lower().replace(country_dict)
    plan_dict = {"fulltime": "full time", "hotdesk": "hot desk",
                 "privateoffice": "private office", "dedicateddesk": "dedicated desk"}
    df["plan_tier"] = (
        df["plan_tier"].astype("string").str.strip().str.lower()
        .str.replace("_", " ", regex=False).replace(plan_dict)
    )

    # --- dates and membership status (1970-01-01 = placeholder for "still active")
    for col in ["membership_start", "membership_end", "last_badge_swipe"]:
        df[col] = parse_mixed_dates(df[col])
    active_placeholder = df["membership_end"] == pd.Timestamp("1970-01-01")
    df.loc[active_placeholder, "membership_end"] = pd.NaT
    df["is_active"] = active_placeholder.astype(bool)

    # --- types
    df["uses_print"] = df["uses_print"].astype(bool)
    df["lifetime_value"] = pd.to_numeric(
        df["lifetime_value"].astype("string")
        .str.replace("€", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )

    # --- impossible values -> missing (flagged in the notebook, handled here for plotting)
    df.loc[~df["age"].between(16, 100), "age"] = np.nan
    df.loc[~df["plan_days_per_week"].between(1, 7), "plan_days_per_week"] = np.nan
    for col in ["n_colleagues_on_account", "days_since_last_visit", "total_visits", "event_space_hours",
                "guest_passes_issued", "n_renewals", "meeting_room_hours"]:
        df.loc[df[col] < 0, col] = np.nan
    df.loc[df["meeting_room_hours"] > MAX_HOURS, "meeting_room_hours"] = np.nan

    # --- engineered features (same definitions as the notebook)
    weekly = df["avg_weekly_visits"].replace(0, np.nan)
    df["recency_vs_usual_gap"] = df["days_since_last_visit"] / (7 / weekly)

    entitled = df["entitled_visits_last_period"].replace(0, np.nan)
    df["usage_rate_last_period"] = df["actual_visits_last_period"] / entitled

    plan_days = df["plan_days_per_week"].replace(0, np.nan)
    df["usage_rate"] = df["avg_weekly_visits"] / plan_days

    weeks_in_period = df["entitled_visits_last_period"] / plan_days
    expected = (df["avg_weekly_visits"] * weeks_in_period).replace(0, np.nan)
    df["visit_trend"] = df["actual_visits_last_period"] / expected

    end = df["membership_end"].fillna(SNAPSHOT)
    tenure_months = (end - df["membership_start"]).dt.days / 30.44
    df["value_per_month"] = df["lifetime_value"] / tenure_months.replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # --- convenience columns for the dashboard
    df["status"] = np.where(df["is_active"], "Active", "Churned")
    df["account_type"] = np.where(df["is_team_account"], "Team", "Individual")
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype("object")
    df["start_year"] = df["membership_start"].dt.year
    for col in ["plan_tier", "country", "industry", "gender", "age_group"]:
        df[col] = df[col].astype("object").where(df[col].notna(), "missing")
    return df, len(raw)


def get_data():
    """Find the CSV automatically or let the user upload it."""
    with st.sidebar.expander("Data source", expanded=False):
        uploaded = st.file_uploader("Upload acdc_dataset.csv", type="csv")
    if uploaded is not None:
        return load_data(uploaded.getvalue())
    for path in DATA_PATHS:
        if path.exists():
            return load_data(str(path))
    st.title("📊 ACDC Members - EDA Dashboard")
    st.info("Dataset not found. Upload `acdc_dataset.csv` in the sidebar "
            "(or place it next to `app.py`).")
    st.stop()


# --------------------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------------------
FILTER_KEYS = ["f_plan", "f_country", "f_industry", "f_gender", "f_account",
               "f_status", "f_age", "f_age_missing", "f_years"]


def reset_filters():
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


def apply_filters(df):
    st.sidebar.header("🔎 Filters")
    st.sidebar.caption("Leave a filter empty to include everything.")
    st.sidebar.button("Reset all filters", on_click=reset_filters)

    plan_opts = [t for t in TIER_ORDER + ["missing"] if t in df["plan_tier"].unique()]
    plans = st.sidebar.multiselect("Plan tier", plan_opts, key="f_plan", placeholder="All plans")
    countries = st.sidebar.multiselect("Country", list(df["country"].value_counts().index),
                                       key="f_country", placeholder="All countries")
    industries = st.sidebar.multiselect("Industry", list(df["industry"].value_counts().index),
                                        key="f_industry", placeholder="All industries")
    genders = st.sidebar.multiselect("Gender", list(df["gender"].value_counts().index),
                                     key="f_gender", placeholder="All genders")
    accounts = st.sidebar.multiselect("Account type", ["Individual", "Team"],
                                      key="f_account", placeholder="All account types")
    statuses = st.sidebar.multiselect("Membership status", ["Active", "Churned"],
                                      key="f_status", placeholder="All statuses")

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max), key="f_age")
    age_missing = st.sidebar.checkbox("Include members with unknown age", value=True, key="f_age_missing")

    years = sorted(df["start_year"].dropna().astype(int).unique())
    year_range = (years[0], years[-1])
    if len(years) > 1:
        year_range = st.sidebar.select_slider("Membership start year", options=years,
                                              value=(years[0], years[-1]), key="f_years")

    mask = pd.Series(True, index=df.index)
    for col, sel in [("plan_tier", plans), ("country", countries), ("industry", industries),
                     ("gender", genders), ("account_type", accounts), ("status", statuses)]:
        if sel:
            mask &= df[col].isin(sel)
    age_ok = df["age"].between(*age_range)
    mask &= age_ok | (df["age"].isna() & age_missing)
    mask &= df["start_year"].between(*year_range) | df["start_year"].isna()
    return df[mask]


# --------------------------------------------------------------------------------------
# KPI helpers
# --------------------------------------------------------------------------------------
def kpis(d):
    return {
        "members": len(d),
        "active": d["is_active"].mean() * 100 if len(d) else np.nan,
        "ltv": d["lifetime_value"].median(),
        "visits": d["avg_weekly_visits"].median(),
        "util": d["usage_rate_last_period"].median(),
        "vpm": d["value_per_month"].median(),
    }


def kpi_row(d, ref=None, share_label=None):
    """Show headline metrics; if `ref` is given, deltas compare `d` against it."""
    k = kpis(d)
    r = kpis(ref) if ref is not None else None

    def delta(key, pattern):
        if r is None or pd.isna(k[key]) or pd.isna(r[key]):
            return None
        return pattern.format(k[key] - r[key])

    cols = st.columns(6)
    cols[0].metric("Members", f"{k['members']:,}", share_label)
    cols[1].metric("Active members", fmt(k["active"], "{:.1f}%"), delta("active", "{:+.1f} pp"))
    cols[2].metric("Median lifetime value", fmt(k["ltv"], "€{:,.0f}"), delta("ltv", "{:+,.0f} €"))
    cols[3].metric("Median weekly visits", fmt(k["visits"], "{:.2f}"), delta("visits", "{:+.2f}"))
    cols[4].metric("Median plan utilisation", fmt(k["util"], "{:.0%}"), delta("util", "{:+.0%}"))
    cols[5].metric("Median value / month", fmt(k["vpm"], "€{:,.0f}"), delta("vpm", "{:+,.0f} €"))


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
def tab_overview(f):
    st.subheader("Who are our members?")
    c1, c2, c3 = st.columns(3)

    plan = f["plan_tier"].value_counts(normalize=True).mul(100).rename("share").reset_index()
    fig = px.bar(plan, x="plan_tier", y="share", color="plan_tier", color_discrete_map=TIER_COLOR,
                 category_orders={"plan_tier": cat_order("plan_tier", f)}, text_auto=".0f",
                 title="Plan tier mix (% of members)", labels={"share": "% of members", "plan_tier": ""})
    fig.update_layout(showlegend=False)
    with c1:
        show(fig, 340)

    age = f["age_group"].value_counts(normalize=True).mul(100).rename("share").reset_index()
    fig = px.bar(age, x="age_group", y="share", text_auto=".0f", color_discrete_sequence=["#87cefa"],
                 category_orders={"age_group": cat_order("age_group", f)},
                 title="Age groups (% of members)", labels={"share": "% of members", "age_group": ""})
    with c2:
        show(fig, 340)

    status = f["status"].value_counts().rename("members").reset_index()
    fig = px.pie(status, names="status", values="members", hole=0.5, color="status",
                 color_discrete_map=STATUS_COLOR, title="Active vs churned")
    with c3:
        show(fig, 340)

    # new members over time
    f_time = f.dropna(subset=["membership_start"]).copy()
    f_time["quarter"] = f_time["membership_start"].dt.to_period("Q").dt.to_timestamp()
    t = f_time.groupby(["quarter", "status"]).size().rename("members").reset_index()
    fig = px.bar(t, x="quarter", y="members", color="status", color_discrete_map=STATUS_COLOR,
                 title="New memberships per quarter (by current status)", labels={"quarter": ""})
    show(fig, 320)

    st.markdown("---")
    st.subheader("Explore any categorical attribute")
    c1, c2, c3 = st.columns([2, 2, 2])
    dim = c1.selectbox("Attribute", list(CAT_LABELS), format_func=CAT_LABELS.get, key="ov_dim")
    split_opts = ["(none)"] + [c for c in CAT_LABELS if c != dim]
    split = c2.selectbox("Split by", split_opts, format_func=lambda c: c if c == "(none)" else CAT_LABELS[c],
                         key="ov_split")
    measure = c3.radio("Show", ["Share of members (%)", "Members (count)"], horizontal=True, key="ov_measure")

    group = [dim] + ([] if split == "(none)" else [split])
    counts = f.groupby(group).size().rename("members").reset_index()
    # shares are computed within each bar when splitting, otherwise over all members
    denom = counts.groupby(dim)["members"].transform("sum") if split != "(none)" else counts["members"].sum()
    counts["share"] = counts["members"] / denom * 100
    ycol = "share" if measure.startswith("Share") else "members"
    top_n = 15
    keep = list(f[dim].value_counts().index[:top_n])
    counts = counts[counts[dim].isin(keep)]
    fig = px.bar(counts, x=dim, y=ycol, color=None if split == "(none)" else split,
                 color_discrete_map=color_map(split) if split != "(none)" else None,
                 category_orders={dim: [c for c in cat_order(dim, f) if c in keep]},
                 text_auto=".0f", barmode="stack",
                 labels={dim: CAT_LABELS[dim], "share": "% of members", "members": "Members"},
                 title=f"{CAT_LABELS[dim]}" + ("" if split == "(none)" else f" by {CAT_LABELS[split]}"))
    show(fig, 420)
    if split != "(none)" and ycol == "share":
        st.caption("When splitting, each bar shows the percentage within that category (adds up to 100%).")
    if f[dim].nunique() > top_n:
        st.caption(f"Showing the {top_n} most frequent values.")


def tab_distributions(f):
    st.subheader("Distribution of a numeric variable")
    c1, c2, c3, c4 = st.columns(4)
    var = c1.selectbox("Variable", list(NUM_LABELS), format_func=NUM_LABELS.get, index=3, key="d_var")
    group = c2.selectbox("Compare by", ["(none)", "plan_tier", "status", "account_type", "gender"],
                         format_func=lambda c: c if c == "(none)" else CAT_LABELS[c], key="d_group")
    kind = c3.radio("Chart", ["Histogram", "Box", "Violin"], horizontal=True, key="d_kind")
    clip = c4.checkbox("Hide top 1% outliers", value=True, key="d_clip")

    d = f[[var] + ([] if group == "(none)" else [group])].dropna(subset=[var])
    if clip and len(d):
        d = d[d[var] <= d[var].quantile(0.99)]
    if d.empty:
        st.warning("No data for this selection.")
        return
    color = None if group == "(none)" else group
    cmap = color_map(group) if color else None
    orders = {group: cat_order(group, d)} if color else {}
    title = f"{NUM_LABELS[var]}" + (f" by {CAT_LABELS[group]}" if color else "")

    if kind == "Histogram":
        o1, o2 = st.columns(2)
        bins = o1.slider("Number of bins", 5, 100, 30, key="d_bins")
        density = o2.checkbox("Normalise (compare shapes)", value=color is not None, key="d_density")
        fig = px.histogram(d, x=var, color=color, nbins=bins, color_discrete_map=cmap,
                           category_orders=orders, barmode="overlay", opacity=0.7,
                           histnorm="probability density" if density else None,
                           labels={var: NUM_LABELS[var]}, title=title)
    elif kind == "Box":
        fig = px.box(d, x=color, y=var, color=color, color_discrete_map=cmap, category_orders=orders,
                     points=False, labels={var: NUM_LABELS[var]}, title=title)
    else:
        fig = px.violin(d, x=color, y=var, color=color, color_discrete_map=cmap, category_orders=orders,
                        box=True, points=False, labels={var: NUM_LABELS[var]}, title=title)
    show(fig, 460)

    stats = (d.groupby(color)[var] if color else d[var]).describe().round(2)
    if color:
        stats = stats.reindex(orders[group])
    else:
        stats = stats.to_frame(NUM_LABELS[var]).T
    st.caption(f"Summary statistics (n = {len(d):,}; missing values excluded)")
    show_table(stats)


def tab_relationships(f):
    st.subheader("Scatter plot: how do two variables relate?")
    c1, c2, c3, c4 = st.columns(4)
    nums = list(NUM_LABELS)
    x = c1.selectbox("X axis", nums, format_func=NUM_LABELS.get, index=nums.index("avg_weekly_visits"), key="s_x")
    y = c2.selectbox("Y axis", nums, format_func=NUM_LABELS.get, index=nums.index("lifetime_value"), key="s_y")
    color = c3.selectbox("Colour by", ["(none)", "plan_tier", "status", "account_type", "gender"],
                         format_func=lambda c: c if c == "(none)" else CAT_LABELS[c], key="s_color")
    clip = c4.checkbox("Hide top 1% outliers", value=True, key="s_clip")

    cols = [x, y] + ([] if color == "(none)" else [color])
    d = f[cols].dropna(subset=[x, y])
    if clip and len(d):
        d = d[(d[x] <= d[x].quantile(0.99)) & (d[y] <= d[y].quantile(0.99))]
    if d.empty:
        st.warning("No data for this selection.")
        return
    n_total = len(d)
    d = d.sample(min(n_total, 6000), random_state=0)
    col = None if color == "(none)" else color
    fig = px.scatter(d, x=x, y=y, color=col, opacity=0.45,
                     color_discrete_map=color_map(col) if col else None,
                     category_orders={col: cat_order(col, d)} if col else None,
                     labels={x: NUM_LABELS[x], y: NUM_LABELS[y]},
                     title=f"{NUM_LABELS[y]} vs {NUM_LABELS[x]}")
    fig.update_traces(marker=dict(size=5))
    show(fig, 500)
    rho = d[[x, y]].corr(method="spearman").iloc[0, 1]
    st.caption(f"Spearman correlation: **{rho:.2f}** · showing {len(d):,} of {n_total:,} members.")

    st.markdown("---")
    st.subheader("Correlation matrix (Spearman)")
    default_vars = ["age", "plan_days_per_week", "days_since_last_visit", "total_visits",
                    "avg_weekly_visits", "entitled_visits_last_period", "actual_visits_last_period",
                    "meeting_room_hours", "event_space_hours", "guest_passes_issued", "n_renewals",
                    "lifetime_value", "usage_rate_last_period"]
    chosen = st.multiselect("Variables", nums, default=default_vars, format_func=NUM_LABELS.get, key="corr_vars")
    if len(chosen) >= 2:
        corr = f[chosen].corr(method="spearman")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="PiYG", zmin=-1, zmax=1, aspect="auto")
        show(fig, 560)
    else:
        st.info("Select at least two variables.")


def tab_plans(f):
    st.subheader("Plan tiers")
    c1, c2 = st.columns([1, 3])
    metric = c1.selectbox("Metric", ["lifetime_value", "value_per_month", "avg_weekly_visits",
                                     "usage_rate", "usage_rate_last_period", "visit_trend", "age"],
                          format_func=NUM_LABELS.get, key="p_metric")
    d = f[["plan_tier", metric]].dropna()
    if metric.startswith("usage_rate"):
        d = d[d[metric].between(0, 1)]
    elif len(d):
        d = d[d[metric] <= d[metric].quantile(0.99)]
    fig = px.box(d, x="plan_tier", y=metric, color="plan_tier", color_discrete_map=TIER_COLOR,
                 category_orders={"plan_tier": cat_order("plan_tier", d)}, points=False,
                 labels={metric: NUM_LABELS[metric], "plan_tier": ""},
                 title=f"{NUM_LABELS[metric]} by plan tier")
    fig.update_layout(showlegend=False)
    with c2:
        show(fig, 420)

    summary = f.groupby("plan_tier").agg(
        members=("member_id", "size"),
        active_pct=("is_active", lambda s: s.mean() * 100),
        median_ltv=("lifetime_value", "median"),
        median_value_per_month=("value_per_month", "median"),
        median_weekly_visits=("avg_weekly_visits", "median"),
        median_utilisation=("usage_rate_last_period", "median"),
    ).reindex(cat_order("plan_tier", f)).round(2)
    st.caption("Summary by plan tier")
    show_table(summary)

    st.markdown("---")
    st.subheader("Add-on services")
    addons = [c for c in f.columns if c.startswith("uses_")]
    labels = {a: a.replace("uses_", "").replace("_", " ") for a in addons}
    by = st.radio("Adoption by", ["Plan tier", "Account type"], horizontal=True, key="a_by")
    dim = "plan_tier" if by == "Plan tier" else "account_type"
    heat = f.groupby(dim)[addons].mean().mul(100).rename(columns=labels)
    heat = heat.reindex(cat_order(dim, f)) if dim == "plan_tier" else heat
    fig = px.imshow(heat, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                    labels=dict(color="% using"), title=f"Add-on adoption (%) by {by.lower()}")
    show(fig, 360 if dim == "plan_tier" else 260)


def tab_drilldown(f):
    st.subheader("Drill-down: from a segment to individual members")
    st.caption("Pick a segment, optionally narrow it down to a single value, then break it down further.")

    # ---- Level 1: choose segment dimension
    c1, c2 = st.columns(2)
    dim = c1.selectbox("1. Segment by", list(CAT_LABELS), format_func=CAT_LABELS.get, key="dd_dim")
    metric_label = c2.selectbox("Metric to compare", list(AGG_METRICS), key="dd_metric")
    metric = AGG_METRICS[metric_label]

    top_n = 15
    order = [c for c in cat_order(dim, f)][:top_n]
    g = f[f[dim].isin(order)].groupby(dim).agg(members=("member_id", "size"),
                                                value=(metric, "median")).reindex(order).reset_index()
    ca, cb = st.columns(2)
    fig = px.bar(g, x=dim, y="members", color=dim if color_map(dim) else None, color_discrete_map=color_map(dim),
                 category_orders={dim: order}, text_auto=",.0f", title=f"Members by {CAT_LABELS[dim].lower()}",
                 labels={dim: ""})
    fig.update_layout(showlegend=False)
    with ca:
        show(fig, 360)
    fig = px.bar(g, x=dim, y="value", color=dim if color_map(dim) else None, color_discrete_map=color_map(dim),
                 category_orders={dim: order}, text_auto=".2f", title=metric_label, labels={dim: "", "value": ""})
    fig.update_layout(showlegend=False)
    with cb:
        show(fig, 360)

    # ---- Level 2: choose one value
    st.markdown("---")
    value = st.selectbox(f"2. Drill into a {CAT_LABELS[dim].lower()}", ["(all)"] + order, key="dd_value")
    seg = f if value == "(all)" else f[f[dim] == value]
    share = f"{len(seg) / len(f) * 100:.1f}% of filtered members" if len(f) else None
    st.markdown(f"**Segment:** {CAT_LABELS[dim]} = `{value}`")
    kpi_row(seg, ref=f if value != "(all)" else None, share_label=share)
    if value != "(all)":
        st.caption("Deltas compare this segment against all currently filtered members.")

    if seg.empty:
        st.warning("No members in this segment.")
        return

    # ---- Level 3: break the segment down by a second attribute
    other = [c for c in CAT_LABELS if c != dim]
    sub = st.selectbox("3. Break this segment down by", other, format_func=CAT_LABELS.get,
                       index=0, key="dd_sub")
    sub_order = cat_order(sub, seg)[:top_n]
    sg = (seg[seg[sub].isin(sub_order)].groupby(sub)
          .agg(members=("member_id", "size"), value=(metric, "median"), n_valid=(metric, "count"))
          .reindex(sub_order).reset_index())
    cc, cd = st.columns(2)
    fig = px.bar(sg, x=sub, y="members", color=sub if color_map(sub) else None, color_discrete_map=color_map(sub),
                 category_orders={sub: sub_order}, text_auto=",.0f",
                 title=f"Members by {CAT_LABELS[sub].lower()}", labels={sub: ""})
    fig.update_layout(showlegend=False)
    with cc:
        show(fig, 340)
    sg_plot = sg[sg["n_valid"] >= 10]
    fig = px.bar(sg_plot, x=sub, y="value", color=sub if color_map(sub) else None, color_discrete_map=color_map(sub),
                 category_orders={sub: sub_order}, text_auto=".2f", title=metric_label,
                 labels={sub: "", "value": ""})
    fig.update_layout(showlegend=False)
    with cd:
        show(fig, 340)
    st.caption("Groups with fewer than 10 valid values are hidden in the right-hand chart.")

    # ---- Level 4: member list
    st.markdown("---")
    st.markdown("**4. Members in this segment**")
    cols = ["member_id", "plan_tier", "country", "industry", "gender", "age", "status", "account_type",
            "membership_start", "lifetime_value", "value_per_month", "avg_weekly_visits",
            "usage_rate", "usage_rate_last_period", "visit_trend"]
    table = seg[cols].sort_values("lifetime_value", ascending=False)
    config = {
        "membership_start": st.column_config.DateColumn("membership_start", format="YYYY-MM-DD"),
        "lifetime_value": st.column_config.NumberColumn("lifetime_value", format="€%.0f"),
        "value_per_month": st.column_config.NumberColumn("value_per_month", format="€%.0f"),
        "avg_weekly_visits": st.column_config.NumberColumn(format="%.2f"),
        "usage_rate": st.column_config.NumberColumn(format="%.2f"),
        "usage_rate_last_period": st.column_config.NumberColumn(format="%.2f"),
        "visit_trend": st.column_config.NumberColumn(format="%.2f"),
    }
    show_table(table.head(1000), hide_index=True, column_config=config)
    st.caption(f"Showing the top {min(len(table), 1000):,} of {len(table):,} members, sorted by lifetime value.")
    st.download_button("⬇️ Download this segment (CSV)",
                       table.to_csv(index=False, sep=";").encode("utf-8"),
                       file_name="segment_members.csv", mime="text/csv")


def tab_data(f, n_raw, n_clean, n_total_filtered):
    st.subheader("Data and quality notes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in raw file", f"{n_raw:,}")
    c2.metric("After removing duplicates", f"{n_clean:,}")
    c3.metric("Rows in current filter", f"{n_total_filtered:,}")

    with st.expander("How the data was prepared", expanded=True):
        st.markdown(
            """
- **Duplicates** removed: exact duplicates and members registered twice under different ids.
- **Labels** standardised for gender, country, industry and plan tier (e.g. `PT` / `portucal` → `portugal`).
- **Dates** parsed from mixed formats; `membership_end = 1970-01-01` means the member is **still active**.
- **Impossible values** (e.g. age outside 16-100, negative visit counts, more than 3 years of meeting-room
  hours, plan days outside 1-7) are treated as missing, so they do not distort charts.
- **Missing categories** appear as `missing`.
- **Engineered features** (same definitions as in the analysis notebook):
  `usage_rate`, `usage_rate_last_period`, `visit_trend`, `recency_vs_usual_gap`, `value_per_month`.
- ⚠️ `usage_rate_last_period` is biased by status: for active members the last period is still in progress,
  so their visits are incomplete.
            """
        )

    st.markdown("**Filtered data**")
    show_table(f.drop(columns=["start_year"]).head(2000), hide_index=True)
    st.caption(f"Showing the first {min(len(f), 2000):,} of {len(f):,} rows.")
    st.download_button("⬇️ Download filtered data (CSV)",
                       f.to_csv(index=False, sep=";").encode("utf-8"),
                       file_name="acdc_filtered.csv", mime="text/csv")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    df, n_raw = get_data()
    f = apply_filters(df)

    st.title("📊 ACDC Members - Interactive EDA Dashboard")
    st.caption("Explore customer attributes: use the sidebar filters, then switch tabs to see distributions, "
               "relationships, plan usage and segment drill-downs.")

    if f.empty:
        st.warning("No members match the current filters. Try removing some of them.")
        st.stop()

    kpi_row(f, share_label=f"{len(f) / len(df) * 100:.0f}% of all members" if len(f) < len(df) else None)
    st.markdown("")

    tabs = st.tabs(["Overview", "Distributions", "Relationships", "Plans & services", "Drill-down", "Data"])
    with tabs[0]:
        tab_overview(f)
    with tabs[1]:
        tab_distributions(f)
    with tabs[2]:
        tab_relationships(f)
    with tabs[3]:
        tab_plans(f)
    with tabs[4]:
        tab_drilldown(f)
    with tabs[5]:
        tab_data(f, n_raw, len(df), len(f))


main()
