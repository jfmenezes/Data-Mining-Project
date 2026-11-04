import io
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------------------
# Page setup and constants
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="ACDC Members - EDA Dashboard",
                   page_icon="📊", layout="wide")

HERE = Path(__file__).resolve().parent
DATA_PATHS = [
    HERE / "acdc_dataset.csv",
    HERE / "Dataset" / "acdc_dataset.csv",
    HERE.parent / "Dataset" / "acdc_dataset.csv",
]

PALETTE = ["#ebc334", "#bd1041", "#76d5de", "#6250d6", "#48e348"]
TIER_ORDER = ["hot desk", "flex", "dedicated desk",
              "full time", "private office"]
TIER_COLOR = dict(zip(TIER_ORDER, PALETTE))
TIER_COLOR["missing"] = "#b0b0b0"
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER, 1)}
STATUS_COLOR = {"Active": "#2ca02c", "Churned": "#d62728"}
ACCOUNT_COLOR = {"Individual": "#EA9F09", "Team": "#87cefa"}

AGE_BINS = [0, 25, 35, 45, 55, np.inf]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]
TENURE_BINS = [-0.001, 3, 12, 24, np.inf]
TENURE_LABELS = ["≤3 months", "3-12 months", "12-24 months", "24+ months"]
RENEWAL_LABELS = ["Renewed", "Never renewed"]

# extraction date used for tenure of active members
REFERENCE_DATE = pd.Timestamp("2024-01-01")
# only 3 years of data -> more hours is impossible
MAX_HOURS = 3 * 365 * 24
EARLY_CHURN_MONTHS = 3

CAT_LABELS = {
    "plan_tier": "Plan tier",
    "country": "Country",
    "industry": "Industry",
    "gender": "Gender",
    "account_type": "Account type",
    "status": "Membership status",
    "age_group": "Age group",
    "tenure_band": "Tenure",
    "renewal_status": "Renewal status",
}
STAR = " ★"  # marks engineered features
NUM_LABELS = {
    "age": "Age",
    "lifetime_value": "Lifetime value (€)",
    "monthly_value": "Monthly value (€)" + STAR,
    "value_per_visit": "Value per visit (€)" + STAR,
    "tenure_months": "Tenure (months)" + STAR,
    "usage_rate": "Usage rate (weekly visits / plan days)" + STAR,
    "visit_trend": "Visit trend (recent vs historical)" + STAR,
    "recency_vs_usual_gap": "Recency vs usual gap" + STAR,
    "tier_rank": "Tier rank (1-5)" + STAR,
    "avg_weekly_visits": "Average weekly visits",
    "total_visits": "Total visits",
    "days_since_last_visit": "Days since last visit (corrected)",
    "visit_utilization_rate": "Plan utilisation, last period",
    "entitled_visits_last_period": "Entitled visits (last period)",
    "actual_visits_last_period": "Actual visits (last period)",
    "n_renewals": "Number of renewals",
    "plan_days_per_week": "Plan days per week",
    "meeting_room_hours": "Meeting room hours",
    "event_space_hours": "Event space hours",
    "guest_passes_issued": "Guest passes issued",
    "n_colleagues_on_account": "Colleagues on account",
}
# Aggregations offered in the drill-down tab: label -> (column, function, scale)
AGG_METRICS = {
    "Median monthly value (€)": ("monthly_value", "median", 1),
    "Median lifetime value (€)": ("lifetime_value", "median", 1),
    "Median value per visit (€)": ("value_per_visit", "median", 1),
    "Median usage rate": ("usage_rate", "median", 1),
    "Median visit trend": ("visit_trend", "median", 1),
    "Median weekly visits": ("avg_weekly_visits", "median", 1),
    "Median tenure (months)": ("tenure_months", "median", 1),
    "Median age": ("age", "median", 1),
    "Renewal rate (%)": ("has_renewed", "mean", 100),
    "Early-churn rate (%)": ("early_churn", "mean", 100),
    "Active share (%)": ("is_active", "mean", 100),
}

# Redundancy check
NEW_COLS = ["tenure_months", "monthly_value", "value_per_visit", "tier_rank", "has_renewed",
            "early_churn", "recency_vs_usual_gap", "usage_rate", "visit_trend"]
ORIGINAL_COLS = ["age", "plan_days_per_week", "days_since_last_visit", "total_visits",
                 "avg_weekly_visits", "entitled_visits_last_period", "actual_visits_last_period",
                 "meeting_room_hours", "event_space_hours", "guest_passes_issued",
                 "n_renewals", "lifetime_value", "visit_utilization_rate"]

# Clustering perspectives
PERSPECTIVES = {
    "Value": ["tier_rank", "monthly_value", "value_per_visit"],
    "Behavioural": ["usage_rate", "tenure_months", "recency_vs_usual_gap", "visit_trend"],
    "Profiling only": ["age_group", "gender", "country", "industry", "is_team_account",
                       "is_active", "has_renewed", "early_churn"],
}

FEATURE_DOC = pd.DataFrame(
    [
        ("tenure_months", "Behavioural", "Months from membership start to end (to 2024-01-01 for active members). "
         "Missing if the end is before the start.", "Active members are still accumulating months (right-censored)."),
        ("monthly_value", "Value", "Lifetime value / tenure (tenure floored at 1 month; negative lifetime values ignored).",
         "Follows plan price closely: about €128 (hot desk) to €697 (private office)."),
        ("value_per_visit", "Value", "Lifetime value / total visits (only members with more than 0 visits).",
         "High = pays a lot but rarely comes (private office)."),
        ("tier_rank", "Value", "Plan tier as a number: hot desk 1, flex 2, dedicated desk 3, full time 4, private office 5.",
         "Ordinal version of the plan tier."),
        ("usage_rate", "Behavioural", "Average weekly visits / plan days per week (plan days must be 1-7).",
         "Near 0: pays for days never used. Near 1: uses the full plan."),
        ("recency_vs_usual_gap", "Behavioural", "Corrected days since last visit / usual gap between visits (7 / weekly visits).",
         "About 1: on schedule. Above 1: overdue (disengagement risk)."),
        ("visit_trend", "Behavioural", "Visits in the last period / visits expected from the member's historical average.",
         "Below 1: fading (0.7 or lower is the clearest warning). Weak signal on its own."),
        ("has_renewed", "Profiling only", "1 if the member renewed at least once, 0 if never, missing if renewals are invalid.",
         "Outcome: used to describe segments, not to build them."),
        ("early_churn", "Profiling only", f"1 if the member left within the first {EARLY_CHURN_MONTHS} months.",
         "Outcome: used to describe segments, not to build them."),
        ("age_group", "Profiling only", "<25, 25-34, 35-44, 45-54, 55+ (ages outside 16-90 are missing).",
         "Used to describe segments."),
        ("days_since_last_visit", "Corrected input", "The 999 placeholder is replaced by the real gap between the last badge "
         "swipe and the end date (or extraction date); negative values are missing.",
         "Used to build recency_vs_usual_gap."),
        ("visit_utilization_rate", "Dropped (redundant)", "Actual / entitled visits in the last period.",
         "Biased for active members (period still running); correlation 0.95 with usage_rate."),
    ],
    columns=["Feature", "Role", "Definition", "How to read it"],
)

_chart_ids = itertools.count()


# --------------------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------------------
def show(fig, height=None):
    """Render a Plotly figure full-width (works across Streamlit versions)."""
    fig.update_layout(template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10))
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
    fixed = {
        "plan_tier": TIER_ORDER,
        "age_group": AGE_LABELS,
        "tenure_band": TENURE_LABELS,
        "renewal_status": RENEWAL_LABELS,
        "status": ["Active", "Churned"],
    }
    if col in fixed:
        return [v for v in fixed[col] + ["missing"] if v in present]
    return list(data[col].value_counts().index)


def color_map(col):
    return {
        "plan_tier": TIER_COLOR,
        "status": STATUS_COLOR,
        "account_type": ACCOUNT_COLOR,
    }.get(col)


def q99_clip(d, col):
    return d[d[col] <= d[col].quantile(0.99)] if len(d) else d


def rate(series):
    """Mean of a 0/1 series expressed in percent (NaN-safe)."""
    return series.mean() * 100


# --------------------------------------------------------------------------------------
# Data loading, cleaning and feature engineering
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


def quality_report(df):
    """Count the data-quality problems BEFORE they are fixed (mirrors the notebook's risk table)."""
    ref_point = df["membership_end"].fillna(REFERENCE_DATE)
    tenure_m = (ref_point - df["membership_start"]).dt.days / 30.44
    age_bad = df["age"].notna() & ~df["age"].between(16, 90)
    rows = [
        ("lifetime_value missing",
         df["lifetime_value"].isna().sum(), "Left as missing"),
        ("lifetime_value zero", (df["lifetime_value"]
         == 0).sum(), "Kept (check separately)"),
        ("lifetime_value negative",
         (df["lifetime_value"] < 0).sum(), "Set to missing"),
        ("days_since_last_visit = 999 (placeholder)", (df["days_since_last_visit"] == 999).sum(),
         "Replaced by the real gap since the last badge swipe"),
        ("days_since_last_visit negative",
         (df["days_since_last_visit"] < 0).sum(), "Set to missing"),
        ("Membership end before start", (df["membership_end"] < df["membership_start"]).sum(),
         "Tenure set to missing"),
        ("Renewals with 1 month of tenure or less", ((tenure_m <= 1) & (df["n_renewals"] > 0)).sum(),
         "Renewal rates are an upper bound (profiling only)"),
        ("n_renewals negative",
         (df["n_renewals"] < 0).sum(), "Set to missing"),
        ("n_colleagues_on_account negative",
         (df["n_colleagues_on_account"] < 0).sum(), "Set to missing"),
        ("total_visits negative",
         (df["total_visits"] < 0).sum(), "Set to missing"),
        ("plan_days_per_week outside 1-7",
         (~df["plan_days_per_week"].between(1, 7)).sum(), "Set to missing"),
        ("age outside 16-90", age_bad.sum(), "Set to missing"),
        ("age missing", df["age"].isna().sum(), "Left as missing"),
        ("gender missing", df["gender"].isna().sum(), "Shown as 'missing'"),
        ("country missing", df["country"].isna().sum(), "Shown as 'missing'"),
        ("plan_tier missing",
         df["plan_tier"].isna().sum(), "Shown as 'missing'"),
    ]
    out = pd.DataFrame(
        rows, columns=["Issue", "Members", "Handling in the dashboard"])
    out["% of members"] = (out["Members"] / len(df) * 100).round(1)
    return out[["Issue", "Members", "% of members", "Handling in the dashboard"]]


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(source):
    """`source` is either a file path (str) or the raw bytes of an uploaded CSV."""
    raw = read_csv_any_sep(io.BytesIO(
        source) if isinstance(source, bytes) else source)
    df = raw.copy()

    # --- duplicates: exact duplicates + same member registered under different ids
    df = df.drop_duplicates()
    df = df[~df.drop(columns="member_id").duplicated(
        keep="first")].reset_index(drop=True)

    # --- categorical labels
    df["gender"] = (
        df["gender"].astype("string").str.strip().str.lower()
        .replace({"f": "female", "m": "male", "o": "other"})
    )
    industry_dict = {"realestate": "real estate",
                     "it": "information technology", "tech": "information technology"}
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
    df["country"] = df["country"].astype(
        "string").str.strip().str.lower().replace(country_dict)
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

    quality = quality_report(df)

    # --- invalid values -> missing (the notebook does this for the feature calculations)
    df.loc[~df["age"].between(16, 90), "age"] = np.nan
    df.loc[~df["plan_days_per_week"].between(
        1, 7), "plan_days_per_week"] = np.nan
    for col in ["n_colleagues_on_account", "total_visits", "event_space_hours", "guest_passes_issued",
                "meeting_room_hours", "n_renewals", "lifetime_value"]:
        df.loc[df[col] < 0, col] = np.nan
    df.loc[df["meeting_room_hours"] > MAX_HOURS, "meeting_room_hours"] = np.nan

    # ================= engineered features =================
    # Renewals
    df["has_renewed"] = (df["n_renewals"] > 0).astype(
        float).where(df["n_renewals"].notna())

    # Tenure: active members run until the extraction date; end before start -> missing
    reference_point = df["membership_end"].fillna(REFERENCE_DATE)
    tenure_days = (reference_point - df["membership_start"]).dt.days
    df["tenure_months"] = (tenure_days / 30.44).where(tenure_days >= 0)

    # Value features (tenure floored at 1 month, visits must be > 0)
    df["monthly_value"] = df["lifetime_value"] / \
        df["tenure_months"].clip(lower=1)
    df["value_per_visit"] = df["lifetime_value"] / \
        df["total_visits"].where(df["total_visits"] > 0)

    # Early churn (an outcome, used for profiling)
    df["early_churn"] = (
        ((~df["is_active"]) & (df["tenure_months"]
         <= EARLY_CHURN_MONTHS)).astype(float)
        .where(df["tenure_months"].notna())
    )

    # Tier rank
    df["tier_rank"] = df["plan_tier"].map(TIER_RANK)

    # Age group (new 5-band version)
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS,
                             labels=AGE_LABELS, right=False).astype("object")

    # Days since last visit (corrected): replace the 999 placeholder with the real gap
    dslv = df["days_since_last_visit"]
    real_gap = (reference_point - df["last_badge_swipe"]).dt.days
    df["days_since_last_visit"] = dslv.where(dslv != 999, real_gap)
    df["days_since_last_visit"] = df["days_since_last_visit"].where(
        df["days_since_last_visit"] >= 0)

    # Recency vs usual gap
    weekly = df["avg_weekly_visits"].where(df["avg_weekly_visits"] > 0)
    df["recency_vs_usual_gap"] = df["days_since_last_visit"] / (7 / weekly)

    # Usage rate and visit trend
    plan_days = df["plan_days_per_week"]
    df["usage_rate"] = df["avg_weekly_visits"] / plan_days
    weeks_in_period = df["entitled_visits_last_period"] / plan_days
    expected_visits = (df["avg_weekly_visits"] *
                       weeks_in_period).replace(0, np.nan)
    df["visit_trend"] = df["actual_visits_last_period"] / expected_visits

    # Last-period utilisation (kept from the EDA; biased for active members)
    entitled = df["entitled_visits_last_period"].where(
        df["entitled_visits_last_period"] > 0)
    df["visit_utilization_rate"] = df["actual_visits_last_period"] / entitled

    df = df.replace([np.inf, -np.inf], np.nan)

    quality.loc[len(quality)] = ["usage_rate above 1", int((df["usage_rate"] > 1).sum()),
                                 round((df["usage_rate"] > 1).mean() * 100, 1), "Kept (flagged as a risk)"]

    # ================= convenience columns for the dashboard =================
    df["status"] = np.where(df["is_active"], "Active", "Churned")
    df["account_type"] = np.where(df["is_team_account"], "Team", "Individual")
    df["start_year"] = df["membership_start"].dt.year
    df["tenure_band"] = pd.cut(
        df["tenure_months"], bins=TENURE_BINS, labels=TENURE_LABELS).astype("object")
    df["renewal_status"] = df["has_renewed"].map(
        {1.0: "Renewed", 0.0: "Never renewed"})
    for col in ["plan_tier", "country", "industry", "gender", "age_group", "tenure_band", "renewal_status"]:
        df[col] = df[col].astype("object").where(df[col].notna(), "missing")
    return df, len(raw), quality


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
FILTER_KEYS = ["f_plan", "f_country", "f_industry", "f_gender", "f_account", "f_status",
               "f_age", "f_age_missing", "f_years", "f_renewal", "f_tenure", "f_tenure_missing"]


def reset_filters():
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


def apply_filters(df):
    st.sidebar.header("🔎 Filters")
    st.sidebar.caption("Leave a filter empty to include everything.")
    st.sidebar.button("Reset all filters", on_click=reset_filters)

    plan_opts = [t for t in TIER_ORDER + ["missing"]
                 if t in df["plan_tier"].unique()]
    plans = st.sidebar.multiselect(
        "Plan tier", plan_opts, key="f_plan", placeholder="All plans")
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
    renewals = st.sidebar.multiselect("Renewal status", RENEWAL_LABELS + ["missing"],
                                      key="f_renewal", placeholder="All members")

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.sidebar.slider(
        "Age", age_min, age_max, (age_min, age_max), key="f_age")
    age_missing = st.sidebar.checkbox(
        "Include members with unknown age", value=True, key="f_age_missing")

    ten_max = int(np.ceil(df["tenure_months"].max())
                  ) if df["tenure_months"].notna().any() else 0
    ten_range, ten_missing = (0, ten_max), True
    if ten_max >= 1:
        ten_range = st.sidebar.slider(
            "Tenure (months)", 0, ten_max, (0, ten_max), key="f_tenure")
        ten_missing = st.sidebar.checkbox("Include members with unknown tenure", value=True,
                                          key="f_tenure_missing")

    years = sorted(df["start_year"].dropna().astype(int).unique())
    year_range = (years[0], years[-1])
    if len(years) > 1:
        year_range = st.sidebar.select_slider("Membership start year", options=years,
                                              value=(years[0], years[-1]), key="f_years")

    mask = pd.Series(True, index=df.index)
    for col, sel in [("plan_tier", plans), ("country", countries), ("industry", industries),
                     ("gender", genders), ("account_type",
                                           accounts), ("status", statuses),
                     ("renewal_status", renewals)]:
        if sel:
            mask &= df[col].isin(sel)
    mask &= df["age"].between(*age_range) | (df["age"].isna() & age_missing)
    mask &= df["tenure_months"].between(
        *ten_range) | (df["tenure_months"].isna() & ten_missing)
    mask &= df["start_year"].between(*year_range) | df["start_year"].isna()
    return df[mask]


# --------------------------------------------------------------------------------------
# KPI helpers
# --------------------------------------------------------------------------------------
def kpis(d):
    return {
        "members": len(d),
        "active": rate(d["is_active"]) if len(d) else np.nan,
        "renewal": rate(d["has_renewed"]),
        "early": rate(d["early_churn"]),
        "mv": d["monthly_value"].median(),
        "usage": d["usage_rate"].median(),
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
    cols[1].metric("Active members", fmt(
        k["active"], "{:.1f}%"), delta("active", "{:+.1f} pp"))
    cols[2].metric("Renewal rate", fmt(k["renewal"], "{:.1f}%"), delta("renewal", "{:+.1f} pp"),
                   help="Share of members who renewed at least once. Likely an upper bound for premium tiers.")
    cols[3].metric("Early churn", fmt(k["early"], "{:.1f}%"), delta("early", "{:+.1f} pp"), delta_color="inverse",
                   help=f"Share of members who left within their first {EARLY_CHURN_MONTHS} months.")
    cols[4].metric("Median monthly value", fmt(k["mv"], "€{:,.0f}"), delta("mv", "{:+,.0f} €"),
                   help="Lifetime value / tenure in months. Follows plan price closely.")
    cols[5].metric("Median usage rate", fmt(k["usage"], "{:.2f}"), delta("usage", "{:+.2f}"),
                   help="Average weekly visits / plan days per week. 1 = uses the full plan.")


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
def tab_overview(f):
    st.subheader("Who are our members?")
    c1, c2, c3 = st.columns(3)

    plan = f["plan_tier"].value_counts(normalize=True).mul(
        100).rename("share").reset_index()
    fig = px.bar(plan, x="plan_tier", y="share", color="plan_tier", color_discrete_map=TIER_COLOR,
                 category_orders={"plan_tier": cat_order("plan_tier", f)}, text_auto=".0f",
                 title="Plan tier mix (% of members)", labels={"share": "% of members", "plan_tier": ""})
    fig.update_layout(showlegend=False)
    with c1:
        show(fig, 340)

    age = f["age_group"].value_counts(normalize=True).mul(
        100).rename("share").reset_index()
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

    f_time = f.dropna(subset=["membership_start"]).copy()
    f_time["quarter"] = f_time["membership_start"].dt.to_period(
        "Q").dt.to_timestamp()
    t = f_time.groupby(["quarter", "status"]).size().rename(
        "members").reset_index()
    fig = px.bar(t, x="quarter", y="members", color="status", color_discrete_map=STATUS_COLOR,
                 title="New memberships per quarter (by current status)", labels={"quarter": ""})
    show(fig, 320)

    st.markdown("---")
    st.subheader("Explore any categorical attribute")
    c1, c2, c3 = st.columns([2, 2, 2])
    dim = c1.selectbox("Attribute", list(CAT_LABELS),
                       format_func=CAT_LABELS.get, key="ov_dim")
    split_opts = ["(none)"] + [c for c in CAT_LABELS if c != dim]
    split = c2.selectbox("Split by", split_opts, format_func=lambda c: c if c == "(none)" else CAT_LABELS[c],
                         key="ov_split")
    measure = c3.radio("Show", [
                       "Share of members (%)", "Members (count)"], horizontal=True, key="ov_measure")

    group = [dim] + ([] if split == "(none)" else [split])
    counts = f.groupby(group).size().rename("members").reset_index()
    denom = counts.groupby(dim)["members"].transform(
        "sum") if split != "(none)" else counts["members"].sum()
    counts["share"] = counts["members"] / denom * 100
    ycol = "share" if measure.startswith("Share") else "members"
    top_n = 15
    keep = list(f[dim].value_counts().index[:top_n])
    counts = counts[counts[dim].isin(keep)]
    fig = px.bar(counts, x=dim, y=ycol, color=None if split == "(none)" else split,
                 color_discrete_map=color_map(
                     split) if split != "(none)" else None,
                 category_orders={
                     dim: [c for c in cat_order(dim, f) if c in keep]},
                 text_auto=".0f", barmode="stack",
                 labels={
                     dim: CAT_LABELS[dim], "share": "% of members", "members": "Members"},
                 title=f"{CAT_LABELS[dim]}" + ("" if split == "(none)" else f" by {CAT_LABELS[split]}"))
    show(fig, 420)
    if split != "(none)" and ycol == "share":
        st.caption(
            "When splitting, each bar shows the percentage within that category (adds up to 100%).")
    if f[dim].nunique() > top_n:
        st.caption(f"Showing the {top_n} most frequent values.")


def tab_distributions(f):
    st.subheader("Distribution of a numeric variable")
    st.caption("★ = engineered feature.")
    c1, c2, c3, c4 = st.columns(4)
    nums = list(NUM_LABELS)
    var = c1.selectbox("Variable", nums, format_func=NUM_LABELS.get,
                       index=nums.index("monthly_value"), key="d_var")
    group = c2.selectbox("Compare by", ["(none)", "plan_tier", "status", "account_type", "gender", "age_group"],
                         format_func=lambda c: c if c == "(none)" else CAT_LABELS[c], key="d_group")
    kind = c3.radio("Chart", ["Histogram", "Box",
                    "Violin"], horizontal=True, key="d_kind")
    clip = c4.checkbox("Hide top 1% outliers", value=True, key="d_clip")

    d = f[[var] + ([] if group == "(none)" else [group])].dropna(subset=[var])
    if clip:
        d = q99_clip(d, var)
    if d.empty:
        st.warning("No data for this selection.")
        return
    color = None if group == "(none)" else group
    cmap = color_map(group) if color else None
    orders = {group: cat_order(group, d)} if color else {}
    title = f"{NUM_LABELS[var]}" + \
        (f" by {CAT_LABELS[group]}" if color else "")

    if kind == "Histogram":
        o1, o2 = st.columns(2)
        bins = o1.slider("Number of bins", 5, 100, 30, key="d_bins")
        density = o2.checkbox("Normalise (compare shapes)",
                              value=color is not None, key="d_density")
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
    if var == "visit_trend":
        fig.add_hline(y=1, line_dash="dash", line_color="gray") if kind != "Histogram" else \
            fig.add_vline(x=1, line_dash="dash", line_color="gray")
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
    x = c1.selectbox("X axis", nums, format_func=NUM_LABELS.get,
                     index=nums.index("usage_rate"), key="s_x")
    y = c2.selectbox("Y axis", nums, format_func=NUM_LABELS.get,
                     index=nums.index("monthly_value"), key="s_y")
    color = c3.selectbox("Colour by", ["(none)", "status", "plan_tier", "account_type", "gender"],
                         format_func=lambda c: c if c == "(none)" else CAT_LABELS[c], key="s_color")
    clip = c4.checkbox("Hide top 1% outliers", value=True, key="s_clip")

    cols = list(dict.fromkeys([x, y] + ([] if color == "(none)" else [color])))
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
    rho = d[[x, y]].corr(method="spearman").iloc[0, 1] if x != y else 1.0
    st.caption(
        f"Spearman correlation: **{rho:.2f}** · showing {len(d):,} of {n_total:,} members.")

    st.markdown("---")
    st.subheader("Correlation matrix (Spearman)")
    default_vars = [c for c in ORIGINAL_COLS if c in NUM_LABELS]
    chosen = st.multiselect("Variables", nums, default=default_vars,
                            format_func=NUM_LABELS.get, key="corr_vars")
    if len(chosen) >= 2:
        corr = f[chosen].corr(method="spearman")
        corr.index = [NUM_LABELS[c] for c in corr.index]
        corr.columns = [NUM_LABELS[c] for c in corr.columns]
        fig = px.imshow(corr, text_auto=".2f",
                        color_continuous_scale="PiYG", zmin=-1, zmax=1, aspect="auto")
        show(fig, 600)
    else:
        st.info("Select at least two variables.")

    st.markdown("---")
    st.subheader("Redundancy check: engineered vs original features")
    st.caption("Pairs with a high absolute Spearman correlation carry almost the same information, "
               "so only one of them should be kept for clustering.")
    corr_all = f[NEW_COLS + ORIGINAL_COLS].corr(method="spearman")
    cross = corr_all.loc[NEW_COLS, ORIGINAL_COLS]
    fig = px.imshow(cross, text_auto=".2f", color_continuous_scale="PiYG", zmin=-1, zmax=1, aspect="auto",
                    labels=dict(color="Spearman ρ"), title="Engineered features (rows) vs original features (columns)")
    show(fig, 480)

    thr = st.slider("Show pairs with |ρ| above", 0.5,
                    0.95, 0.8, 0.05, key="red_thr")
    new_corr = corr_all.loc[NEW_COLS, NEW_COLS]
    lower = ~np.triu(np.ones(new_corr.shape, dtype=bool))
    pairs = pd.concat(
        [cross.stack(), new_corr.where(lower).stack()]).rename("rho")
    pairs.index.names = ["Feature", "Compared with"]
    pairs = pairs[pairs.abs() > thr].sort_values(
        key=abs, ascending=False).round(2).reset_index()
    if pairs.empty:
        st.info("No pairs above this threshold in the current selection.")
    else:
        show_table(pairs, hide_index=True)


def tab_plans(f):
    st.subheader("Plan tiers: value and engagement")
    st.caption("★ = engineered feature.")
    metric_opts = ["monthly_value", "lifetime_value", "value_per_visit", "usage_rate", "avg_weekly_visits",
                   "visit_utilization_rate", "visit_trend", "recency_vs_usual_gap", "tenure_months", "age"]
    c1, c2 = st.columns([1, 3])
    metric = c1.selectbox("Metric", metric_opts,
                          format_func=NUM_LABELS.get, key="p_metric")
    d = f[["plan_tier", metric]].dropna()
    if metric in ("usage_rate", "visit_utilization_rate"):
        d = d[d[metric].between(0, 1)]
        c1.caption("Showing values between 0 and 1.")
    else:
        d = q99_clip(d, metric)
        c1.caption("Top 1% outliers hidden.")
    fig = px.box(d, x="plan_tier", y=metric, color="plan_tier", color_discrete_map=TIER_COLOR,
                 category_orders={"plan_tier": cat_order("plan_tier", d)}, points=False,
                 labels={metric: NUM_LABELS[metric], "plan_tier": ""},
                 title=f"{NUM_LABELS[metric]} by plan tier")
    if metric == "visit_trend":
        fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(showlegend=False)
    with c2:
        show(fig, 420)

    order = cat_order("plan_tier", f)
    summary = f.groupby("plan_tier").agg(
        members=("member_id", "size"),
        active_pct=("is_active", rate),
        renewal_rate_pct=("has_renewed", rate),
        early_churn_pct=("early_churn", rate),
        median_monthly_value=("monthly_value", "median"),
        median_value_per_visit=("value_per_visit", "median"),
        median_usage_rate=("usage_rate", "median"),
        median_weekly_visits=("avg_weekly_visits", "median"),
        median_lifetime_value=("lifetime_value", "median"),
    ).reindex(order).round(2)
    st.caption("Summary by plan tier")
    show_table(summary)

    st.markdown("---")
    st.subheader("Share of members vs share of lifetime value")
    cap = st.checkbox("Cap lifetime value at the 99th percentile (reduces the weight of a few extreme members)",
                      value=False, key="p_cap")
    ltv = f["lifetime_value"]
    if cap and ltv.notna().any():
        ltv = ltv.clip(upper=ltv.quantile(0.99))
    share = pd.DataFrame({"plan_tier": f["plan_tier"], "ltv": ltv}).groupby("plan_tier").agg(
        members=("ltv", "size"), ltv=("ltv", "sum"))
    share["% of members"] = share["members"] / share["members"].sum() * 100
    share["% of lifetime value"] = share["ltv"] / share["ltv"].sum() * 100
    long = share.reindex(order)[["% of members", "% of lifetime value"]].reset_index().melt(
        id_vars="plan_tier", var_name="measure", value_name="pct")
    fig = px.bar(long, x="plan_tier", y="pct", color="measure", barmode="group", text_auto=".0f",
                 category_orders={"plan_tier": order},
                 color_discrete_sequence=["#87cefa", "#EA9F09"],
                 labels={"pct": "%", "plan_tier": "", "measure": ""})
    show(fig, 360)
    st.caption("A tier whose share of value is much larger than its share of members is disproportionately "
               "important to revenue (the notebook finds this for private office).")

    st.markdown("---")
    st.subheader("Add-on services")
    addons = [c for c in f.columns if c.startswith("uses_")]
    labels = {a: a.replace("uses_", "").replace("_", " ") for a in addons}
    by = st.radio("Adoption by", ["Plan tier",
                  "Account type"], horizontal=True, key="a_by")
    dim = "plan_tier" if by == "Plan tier" else "account_type"
    heat = f.groupby(dim)[addons].mean().mul(100).rename(columns=labels)
    heat = heat.reindex(cat_order(dim, f)) if dim == "plan_tier" else heat
    fig = px.imshow(heat, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                    labels=dict(color="% using"), title=f"Add-on adoption (%) by {by.lower()}")
    show(fig, 360 if dim == "plan_tier" else 260)


def tab_behaviour(f):
    st.subheader("Behavioural signals")
    st.caption(
        "How much, how regularly and how recently members use the space. Green = active, red = churned.")

    # ---- the three scatter plots from the notebook
    views = {
        "Long-run attendance vs recent trend": ("usage_rate", "visit_trend"),
        "Usage rate vs monthly value": ("usage_rate", "monthly_value"),
        "Absence vs recent trend": ("recency_vs_usual_gap", "visit_trend"),
    }
    view = st.radio("View", list(views), horizontal=True, key="b_view")
    x, y = views[view]
    d = f[[x, y, "status"]].dropna()
    if x == "usage_rate":
        d = d[d[x].between(0, 1)]
    elif len(d):
        d = d[d[x].between(0, d[x].quantile(0.99))]
    if len(d):
        d = d[d[y].between(0, d[y].quantile(0.99))]
    if d.empty:
        st.warning("No data for this selection.")
    else:
        n_shown = len(d)
        d = d.sample(min(n_shown, 6000), random_state=0)
        fig = px.scatter(d, x=x, y=y, color="status", color_discrete_map=STATUS_COLOR, opacity=0.35,
                         category_orders={"status": ["Active", "Churned"]},
                         labels={x: NUM_LABELS[x], y: NUM_LABELS[y]},
                         title=f"{view} (n = {n_shown:,} of {len(f):,} members shown)")
        fig.update_traces(marker=dict(size=5))
        if y == "visit_trend":
            fig.add_hline(y=1, line_dash="dash", line_color="gray")
        show(fig, 480)
        if y == "visit_trend":
            st.caption(
                "Dashed line: trend = 1 (stable). A row of points at 0 are members who stopped visiting.")

    # ---- feature by tier and by status
    st.markdown("---")
    st.subheader("Engineered features by plan tier and by membership status")
    bounds = {"usage_rate": (0, 1), "visit_trend": (0, None), "recency_vs_usual_gap": (0, None),
              "monthly_value": (0, None), "value_per_visit": (0, None), "tenure_months": (0, None)}
    feat = st.selectbox("Feature", list(bounds),
                        format_func=NUM_LABELS.get, key="b_feat")
    lo, hi = bounds[feat]
    d = f[[feat, "plan_tier", "status"]].dropna()
    if len(d):
        upper = hi if hi is not None else d[feat].quantile(0.99)
        d = d[d[feat].between(lo, upper)]
    if d.empty:
        st.warning("No data for this selection.")
    else:
        ca, cb = st.columns(2)
        fig = px.box(d, x="plan_tier", y=feat, color="plan_tier", color_discrete_map=TIER_COLOR,
                     category_orders={"plan_tier": cat_order("plan_tier", d)}, points=False,
                     labels={feat: NUM_LABELS[feat], "plan_tier": ""}, title="By plan tier")
        fig.update_layout(showlegend=False)
        if feat == "visit_trend":
            fig.add_hline(y=1, line_dash="dash", line_color="gray")
        with ca:
            show(fig, 400)
        fig = px.box(d, x="status", y=feat, color="status", color_discrete_map=STATUS_COLOR,
                     category_orders={"status": ["Active", "Churned"]}, points=False,
                     labels={feat: NUM_LABELS[feat], "status": ""}, title="By membership status")
        fig.update_layout(showlegend=False)
        if feat == "visit_trend":
            fig.add_hline(y=1, line_dash="dash", line_color="gray")
        with cb:
            show(fig, 400)
        if feat == "usage_rate":
            st.caption("⚠️ Churned members show a higher usage rate than active ones, the opposite of what was "
                       "expected. The notebook flags this as a data-quality risk to investigate.")

    # ---- paying but absent
    st.markdown("---")
    st.subheader("Paying members who stopped coming")
    c1, c2 = st.columns(2)
    days = c1.slider(
        "Dormant = active member not seen for more than (days)", 14, 365, 60, 7, key="b_days")
    low = c2.slider("Low usage = usage rate below",
                    0.05, 0.75, 0.25, 0.05, key="b_low")

    active = f[f["is_active"]]
    dormant = active[active["days_since_last_visit"] > days]
    stopped = int((active["visit_trend"] == 0).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dormant active members", f"{len(dormant):,}")
    k2.metric("Share of active members", fmt(len(dormant) /
              len(active) * 100 if len(active) else np.nan, "{:.1f}%"))
    k3.metric("Monthly value of dormant members", fmt(dormant["monthly_value"].sum(), "€{:,.0f}"),
              help="Sum of monthly value: revenue potentially at risk if they leave.")
    k4.metric("Active with no visit last period",
              f"{stopped:,}", help="Active members with visit_trend = 0.")

    ud = f[f["usage_rate"].notna()].copy()
    ud["low"] = ud["usage_rate"] < low
    lowshare = ud.groupby("plan_tier")["low"].mean().mul(
        100).rename("pct").reset_index()
    fig = px.bar(lowshare, x="plan_tier", y="pct", color="plan_tier", color_discrete_map=TIER_COLOR,
                 category_orders={"plan_tier": cat_order("plan_tier", lowshare)}, text_auto=".0f",
                 labels={"pct": "% of members", "plan_tier": ""},
                 title=f"% of members using less than {low:.0%} of their plan, by tier")
    fig.update_layout(showlegend=False)
    show(fig, 340)

    st.markdown("**Dormant active members, highest monthly value first**")
    table = dormant[["member_id", "plan_tier", "country", "days_since_last_visit", "usage_rate", "visit_trend",
                     "monthly_value", "tenure_months"]].sort_values("monthly_value", ascending=False)
    config = {
        "days_since_last_visit": st.column_config.NumberColumn("days_since_last_visit", format="%d"),
        "usage_rate": st.column_config.NumberColumn(format="%.2f"),
        "visit_trend": st.column_config.NumberColumn(format="%.2f"),
        "monthly_value": st.column_config.NumberColumn(format="€%.0f"),
        "tenure_months": st.column_config.NumberColumn(format="%.1f"),
    }
    show_table(table.head(500), hide_index=True, column_config=config)
    st.caption(
        f"Showing the top {min(len(table), 500):,} of {len(table):,} dormant members.")
    st.download_button("⬇️ Download dormant members (CSV)", table.to_csv(index=False, sep=";").encode("utf-8"),
                       file_name="dormant_members.csv", mime="text/csv")


def tab_retention(f):
    st.subheader("Retention and churn")
    leavers = f[~f["is_active"]]
    c = st.columns(5)
    c[0].metric("Renewal rate", fmt(rate(f["has_renewed"]), "{:.1f}%"),
                help="Members who renewed at least once. Premium tiers are likely overstated.")
    c[1].metric(f"Early churn (≤{EARLY_CHURN_MONTHS} months)", fmt(rate(f["early_churn"]), "{:.1f}%"),
                help="Share of all members who left within the first months.")
    c[2].metric("Early leavers among churned", fmt(
        rate(leavers["early_churn"]), "{:.1f}%"))
    c[3].metric("Median tenure, active", fmt(
        f.loc[f["is_active"], "tenure_months"].median(), "{:.1f} mo"))
    c[4].metric("Median tenure, churned", fmt(
        leavers["tenure_months"].median(), "{:.1f} mo"))

    order = cat_order("plan_tier", f)
    st.markdown("---")
    st.subheader("Renewal rate by plan tier")
    r = f.groupby("plan_tier").agg(members=("member_id", "size"), renewal_rate=("has_renewed", rate),
                                   median_renewals=("n_renewals", "median")).reindex(order).reset_index()
    ca, cb = st.columns([3, 2])
    fig = px.bar(r, x="plan_tier", y="renewal_rate", color="plan_tier", color_discrete_map=TIER_COLOR,
                 category_orders={"plan_tier": order}, text_auto=".0f",
                 labels={"renewal_rate": "% of members who renewed",
                         "plan_tier": ""},
                 title="Share of members who renewed at least once")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(showlegend=False)
    with ca:
        show(fig, 380)
    with cb:
        st.caption("Renewal detail by tier")
        show_table(r.round(1).set_index("plan_tier"))
    st.caption("Commitment drives retention: dedicated desk and private office renew most, and dedicated desk "
               "renews better than full time. Treat premium rates as an upper bound, because some members with "
               "one month of tenure or less already show renewals (see the Data tab).")

    st.markdown("---")
    st.subheader("Early churn")
    months = st.slider("Early-churn window (months)", 1, 12, EARLY_CHURN_MONTHS, key="r_months",
                       help=f"The dashboard-wide early-churn KPI uses {EARLY_CHURN_MONTHS} months.")
    d = f.copy()
    d["early"] = ((~d["is_active"]) & (d["tenure_months"] <= months)).astype(
        float).where(d["tenure_months"].notna())
    ea, eb = st.columns(2)
    by_tier = d.groupby("plan_tier")["early"].apply(
        rate).rename("pct").reindex(order).reset_index()
    fig = px.bar(by_tier, x="plan_tier", y="pct", color="plan_tier", color_discrete_map=TIER_COLOR,
                 category_orders={"plan_tier": order}, text_auto=".0f",
                 labels={"pct": "% of members", "plan_tier": ""},
                 title=f"% of members who left within {months} months")
    fig.update_layout(showlegend=False)
    with ea:
        show(fig, 360)
    early_only = d[d["early"] == 1]
    if early_only.empty:
        eb.info("No early leavers in the current selection.")
    else:
        comp = early_only["plan_tier"].value_counts(
            normalize=True).mul(100).rename("pct").reset_index()
        fig = px.bar(comp, x="plan_tier", y="pct", color="plan_tier", color_discrete_map=TIER_COLOR,
                     category_orders={"plan_tier": order}, text_auto=".0f",
                     labels={"pct": "% of early leavers", "plan_tier": ""},
                     title="Who are the early leavers? (plan tier mix)")
        fig.update_layout(showlegend=False)
        with eb:
            show(fig, 360)
    st.caption(
        "Early leavers are concentrated in hot desk and flex plans, a natural target for onboarding actions.")

    st.markdown("---")
    st.subheader("Tenure")
    t = f.dropna(subset=["tenure_months"])
    fig = px.histogram(t, x="tenure_months", color="status", color_discrete_map=STATUS_COLOR, nbins=36,
                       barmode="overlay", opacity=0.7, category_orders={"status": ["Active", "Churned"]},
                       labels={"tenure_months": "Tenure (months)"}, title="Tenure by membership status")
    show(fig, 360)
    st.caption("Active members' tenure is right-censored: they are still accumulating months. "
               "The data covers roughly three years of memberships.")

    st.markdown("---")
    st.subheader("Retention by any attribute")
    c1, c2 = st.columns(2)
    dim = c1.selectbox("Attribute", list(CAT_LABELS), format_func=CAT_LABELS.get, key="r_dim",
                       index=list(CAT_LABELS).index("industry"))
    measure = c2.radio("Measure", ["Renewal rate (%)", "Early-churn rate (%)", "Active share (%)"],
                       horizontal=True, key="r_measure")
    col, _, _ = AGG_METRICS[measure]
    g = f.groupby(dim).agg(members=("member_id", "size"), value=(
        col, rate), n_valid=(col, "count")).reset_index()
    g = g[g["n_valid"] >= 20]
    if g.empty:
        st.info("Not enough members per group (minimum 20).")
    else:
        keep = [v for v in cat_order(dim, f) if v in set(g[dim])][:20]
        g = g[g[dim].isin(keep)]
        fig = px.bar(g, x=dim, y="value", color=dim if color_map(dim) else None, color_discrete_map=color_map(dim),
                     category_orders={dim: keep}, text_auto=".0f", hover_data={"members": True},
                     labels={dim: "", "value": measure}, title=f"{measure} by {CAT_LABELS[dim].lower()}")
        fig.update_layout(showlegend=False)
        show(fig, 400)
        st.caption("Groups with fewer than 20 valid values are hidden.")


def tab_drilldown(f):
    st.subheader("Drill-down: from a segment to individual members")
    st.caption(
        "Pick a segment, optionally narrow it down to a single value, then break it down further.")

    # ---- Level 1: choose segment dimension
    c1, c2 = st.columns(2)
    dim = c1.selectbox("1. Segment by", list(CAT_LABELS),
                       format_func=CAT_LABELS.get, key="dd_dim")
    metric_label = c2.selectbox(
        "Metric to compare", list(AGG_METRICS), key="dd_metric")
    mcol, mfunc, mscale = AGG_METRICS[metric_label]

    def aggregate(d, by, order):
        g = d[d[by].isin(order)].groupby(by).agg(members=("member_id", "size"),
                                                 value=(mcol, mfunc), n_valid=(mcol, "count"))
        g["value"] = g["value"] * mscale
        return g.reindex(order).reset_index()

    top_n = 15
    order = cat_order(dim, f)[:top_n]
    g = aggregate(f, dim, order)
    ca, cb = st.columns(2)
    fig = px.bar(g, x=dim, y="members", color=dim if color_map(dim) else None, color_discrete_map=color_map(dim),
                 category_orders={dim: order}, text_auto=",.0f", title=f"Members by {CAT_LABELS[dim].lower()}",
                 labels={dim: ""})
    fig.update_layout(showlegend=False)
    with ca:
        show(fig, 360)
    gp = g[g["n_valid"] >= 10]
    fig = px.bar(gp, x=dim, y="value", color=dim if color_map(dim) else None, color_discrete_map=color_map(dim),
                 category_orders={dim: order}, text_auto=".2f", title=metric_label, labels={dim: "", "value": ""})
    fig.update_layout(showlegend=False)
    with cb:
        show(fig, 360)

    # ---- Level 2: choose one value
    st.markdown("---")
    value = st.selectbox(f"2. Drill into a {CAT_LABELS[dim].lower()}", [
                         "(all)"] + order, key="dd_value")
    seg = f if value == "(all)" else f[f[dim] == value]
    share = f"{len(seg) / len(f) * 100:.1f}% of filtered members" if len(f) else None
    st.markdown(f"**Segment:** {CAT_LABELS[dim]} = `{value}`")
    kpi_row(seg, ref=f if value != "(all)" else None, share_label=share)
    if value != "(all)":
        st.caption(
            "Deltas compare this segment against all currently filtered members.")

    if seg.empty:
        st.warning("No members in this segment.")
        return

    # ---- Level 3: break the segment down by a second attribute
    other = [c for c in CAT_LABELS if c != dim]
    sub = st.selectbox("3. Break this segment down by", other,
                       format_func=CAT_LABELS.get, index=0, key="dd_sub")
    sub_order = cat_order(sub, seg)[:top_n]
    sg = aggregate(seg, sub, sub_order)
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
    st.caption(
        "Groups with fewer than 10 valid values are hidden in the right-hand chart.")

    # ---- Level 4: member list
    st.markdown("---")
    st.markdown("**4. Members in this segment**")
    cols = ["member_id", "plan_tier", "country", "industry", "gender", "age", "status", "account_type",
            "membership_start", "tenure_months", "lifetime_value", "monthly_value", "value_per_visit",
            "usage_rate", "visit_trend", "recency_vs_usual_gap", "has_renewed", "early_churn"]
    table = seg[cols].sort_values("monthly_value", ascending=False)
    config = {
        "membership_start": st.column_config.DateColumn("membership_start", format="YYYY-MM-DD"),
        "tenure_months": st.column_config.NumberColumn(format="%.1f"),
        "lifetime_value": st.column_config.NumberColumn(format="€%.0f"),
        "monthly_value": st.column_config.NumberColumn(format="€%.0f"),
        "value_per_visit": st.column_config.NumberColumn(format="€%.1f"),
        "usage_rate": st.column_config.NumberColumn(format="%.2f"),
        "visit_trend": st.column_config.NumberColumn(format="%.2f"),
        "recency_vs_usual_gap": st.column_config.NumberColumn(format="%.2f"),
        "has_renewed": st.column_config.NumberColumn(format="%d"),
        "early_churn": st.column_config.NumberColumn(format="%d"),
    }
    show_table(table.head(1000), hide_index=True, column_config=config)
    st.caption(
        f"Showing the top {min(len(table), 1000):,} of {len(table):,} members, sorted by monthly value.")
    st.download_button("⬇️ Download this segment (CSV)",
                       table.to_csv(index=False, sep=";").encode("utf-8"),
                       file_name="segment_members.csv", mime="text/csv")


def tab_data(f, n_raw, n_clean, quality):
    st.subheader("Data, features and quality notes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in raw file", f"{n_raw:,}")
    c2.metric("After removing duplicates", f"{n_clean:,}")
    c3.metric("Rows in current filter", f"{len(f):,}")

    with st.expander("How the data was prepared", expanded=False):
        st.markdown(
            f"""
- **Duplicates** removed: exact duplicates and members registered twice under different ids.
- **Labels** standardised for gender, country, industry and plan tier (e.g. `PT` / `portucal` → `portugal`).
- **Dates** parsed from mixed formats; `membership_end = 1970-01-01` means the member is **still active**.
  Tenure of active members runs until the extraction date ({REFERENCE_DATE.date()}).
- **Invalid values** (age outside 16-90, negative counts or lifetime values, more than 3 years of meeting-room
  hours, plan days outside 1-7) are treated as missing so they do not distort the charts.
- **`days_since_last_visit`**: the 999 placeholder is replaced by the real gap since the last badge swipe.
- **Missing categories** appear as `missing`.
- ⚠️ `visit_utilization_rate` and `usage_rate` are biased by status (active members look less engaged than churned
  ones). The notebook flags this as a data-quality risk.
            """
        )

    st.markdown("**Data-quality issues found in the raw data**")
    show_table(quality, hide_index=True)

    st.markdown("---")
    st.markdown("**Engineered features (★)**")
    show_table(FEATURE_DOC, hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Feature sets for the segmentation (clustering perspectives)**")
    cols = st.columns(3)
    for col, (name, feats) in zip(cols, PERSPECTIVES.items()):
        with col:
            st.markdown(f"**{name}**")
            st.markdown("\n".join(f"- `{x}`" for x in feats))
    st.caption("Profiling-only variables describe the segments once they are built; outcomes such as `has_renewed` "
               "and `early_churn` are not used to form them. Dropped as redundant: `plan_days_per_week`, "
               "`entitled_visits_last_period`, `visit_utilization_rate`, `avg_weekly_visits`, `n_renewals`, "
               "`lifetime_value`.")
    clustering_cols = PERSPECTIVES["Value"] + PERSPECTIVES["Behavioural"]
    miss = f[clustering_cols].isna().sum().rename("missing").to_frame()
    miss["% missing"] = (miss["missing"] / max(len(f), 1) * 100).round(1)
    st.caption("Missing values in the clustering features (current filter)")
    show_table(miss)

    st.markdown("---")
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
    df, n_raw, quality = get_data()
    f = apply_filters(df)

    st.title("📊 ACDC Members - Interactive EDA Dashboard")
    st.caption("Explore customer attributes: use the sidebar filters, then switch tabs to see distributions, "
               "plan value, behaviour, retention and segment drill-downs. ★ marks engineered features.")

    if f.empty:
        st.warning(
            "No members match the current filters. Try removing some of them.")
        st.stop()

    kpi_row(
        f, share_label=f"{len(f) / len(df) * 100:.0f}% of all members" if len(f) < len(df) else None)
    st.markdown("")

    tabs = st.tabs(["Overview", "Distributions", "Relationships", "Plans & value",
                    "Behaviour", "Retention", "Drill-down", "Data & features"])
    with tabs[0]:
        tab_overview(f)
    with tabs[1]:
        tab_distributions(f)
    with tabs[2]:
        tab_relationships(f)
    with tabs[3]:
        tab_plans(f)
    with tabs[4]:
        tab_behaviour(f)
    with tabs[5]:
        tab_retention(f)
    with tabs[6]:
        tab_drilldown(f)
    with tabs[7]:
        tab_data(f, n_raw, len(df), quality)


main()
