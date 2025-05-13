import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Meta Campaign Marketing Insights", layout="wide")

######################## CSS Tweaks and Card Format ########################
st.markdown("""
    <style>
        div[data-testid="metric-container"] {background: #F8F9FB; border-radius: 7px; padding: 15px}
        .element-container:has(.css-1y9yaex) {margin-bottom: 16px;}
        .my-card {background:#fff; border:1px solid #efefef; border-radius:8px; padding:14px 24px; box-shadow:1px 2px 8px 0 #eee;}
        .zone-green {color:#1aaa55;} .zone-red {color:#e6524e;} .zone-yellow {color:#eec95e;}
        .legend-dot {display:inline-block; width:14px; height:14px; border-radius:7px; margin-right:6px;}
        .green-dot {background:#ccffcc;} .red-dot{background:#ffcccc;} .yellow-dot{background:#fff7cc;}
    </style>
""", unsafe_allow_html=True)

def date_range_selector(df, col="date_start", default_days=60):
    minv = pd.to_datetime(df[col], errors="coerce").min() if col in df.columns else pd.to_datetime('today') - pd.Timedelta(days=default_days)
    maxv = pd.to_datetime(df[col], errors="coerce").max() if col in df.columns else pd.to_datetime('today')
    drange = st.sidebar.date_input("Filter by date range", [minv, maxv])
    if isinstance(drange, tuple): start, end = drange
    else: start, end = minv, maxv
    mask = (pd.to_datetime(df[col], errors="coerce") >= pd.to_datetime(start)) & (pd.to_datetime(df[col], errors="coerce") <= pd.to_datetime(end))
    return mask, start, end

################################# Data Loading and Type Safety ####################################

def safe_read_csv(fname):
    try:
        df = pd.read_csv(fname)
        return df
    except Exception as e:
        st.error(f"Cannot read {fname}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_data():
    campaigns = safe_read_csv("60 days insights - Campaign Metrics.csv")
    adsets = safe_read_csv("60 days insights - Adset Metrics.csv")
    ads = safe_read_csv("60 days insights - Ad Metrics (1).csv")
    for df in [campaigns, adsets, ads]:
        for col in ["dt", "date_start", "date_stop", "updated_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for id_col in ["buid", "campaign_id", "adset_id"]:
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str).replace("nan", "")
    return campaigns, adsets, ads

def robust_floatify(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_meta_hierarchy(campaigns, adsets, ads):
    adsets_campaign = pd.merge(adsets, campaigns, on="campaign_id", how="left", suffixes=('_adset', '_campaign')) \
        if "campaign_id" in adsets.columns and "campaign_id" in campaigns.columns else pd.DataFrame()
    ads_adsets = pd.merge(ads, adsets[["adset_id","campaign_id"]].drop_duplicates(), on='adset_id', how='left') \
        if "adset_id" in ads.columns and "adset_id" in adsets.columns else ads.copy()
    ads_full = pd.merge(ads_adsets, campaigns[["campaign_id", "buid", "objective", "status"]], on='campaign_id', how="left") \
        if "campaign_id" in ads_adsets.columns and "campaign_id" in campaigns.columns else ads_adsets.copy()
    return adsets_campaign, ads_full

############################### Metrics, Trend and Targeting Analysis ################################

def classify_zone(row):
    roas = float(row["roas_mean"]) if not pd.isnull(row.get("roas_mean", np.nan)) else 0
    ctr  = float(row["ctr_mean"]) if not pd.isnull(row.get("ctr_mean", np.nan)) else 0
    cpm  = float(row["cpm_mean"]) if not pd.isnull(row.get("cpm_mean", np.nan)) else 1000
    if (roas < 1) or (ctr < 0.005) or (cpm > 10):
        return "Red"
    elif ((1 <= roas < 1.5) or (0.005 <= ctr < 0.01) or (5 < cpm <= 10)):
        return "Yellow"
    elif (roas >= 1.5) and (ctr >= 0.01) and (cpm <= 5):
        return "Green"
    else:
        return "Yellow"

def style_zone(val):
    color = {"Red": "#fff3f3", "Yellow":"#fdf7da", "Green":"#eaf9ed"}.get(val, "#f8f8f8")
    return f"background-color: {color}"

def summarize_metrics(df, groupby_col=None):
    numeric_cols = ['purchase_roas', 'roas', 'ctr', 'cpm', 'impressions', 'spend', 'results', 'cpa', 'cost_per_result']
    robust_floatify(df, numeric_cols)
    agg_roas = next((col for col in ["purchase_roas", "roas"] if col in df.columns), None)
    agg_cpa = next((col for col in ["cpa", "cost_per_result"] if col in df.columns), None)
    if groupby_col:
        g = df.groupby(groupby_col).agg(
            roas_mean = (agg_roas, 'mean') if agg_roas else ('campaign_id', 'size'),
            ctr_mean = ('ctr', 'mean') if 'ctr' in df.columns else ('campaign_id','size'),
            cpm_mean = ('cpm', 'mean') if 'cpm' in df.columns else ('campaign_id','size'),
            impressions=('impressions','sum') if 'impressions' in df.columns else ('campaign_id','size'),
            spend=('spend','sum') if 'spend' in df.columns else ('campaign_id','size'),
            results = ('results', 'sum') if 'results' in df.columns else ('campaign_id','size'),
            cpa_mean = (agg_cpa, 'mean') if agg_cpa else ('campaign_id', 'size')
        ).reset_index()
    else:
        g = pd.DataFrame([{
            "roas_mean": df[agg_roas].mean() if agg_roas else np.nan,
            "ctr_mean": df["ctr"].mean() if "ctr" in df.columns else np.nan,
            "cpm_mean": df["cpm"].mean() if "cpm" in df.columns else np.nan,
            "impressions": df["impressions"].sum() if "impressions" in df.columns else np.nan,
            "spend": df["spend"].sum() if "spend" in df.columns else np.nan,
            "results": df["results"].sum() if "results" in df.columns else np.nan,
            "cpa_mean": df[agg_cpa].mean() if agg_cpa else np.nan,
        }])
    if "roas_mean" in g.columns and "ctr_mean" in g.columns and "cpm_mean" in g.columns:
        g["Zone"] = g.apply(classify_zone, axis=1)
    return g

def aggregate_metric_trend(user_ads, metric, groupby="date_start"):
    if groupby not in user_ads.columns or metric not in user_ads.columns:
        return pd.DataFrame()
    df = user_ads.groupby(groupby).agg({metric:'mean'}).reset_index()
    return df

TARGET_COLS_CANDIDATES = [
    "age", "gender", "genders", "geo", "country", "countries", "location",
    "region", "city", "interest", "interests", "detailed_targeting", "custom_audience",
    "platform", "platforms", "publisher_platform", "placement", "device", "device_platform",
    "user_os", "user_device", "mobile_device_type", "ad_placement"
]

def get_targeting_columns(df):
    tcols = []
    for c in TARGET_COLS_CANDIDATES:
        for col in df.columns:
            if c in col.lower():
                tcols.append(col)
    return list(sorted(set(tcols)))

def segment_performance_table(df, segment_col, perf_col="roas_mean", min_group=2, as_percent=False, show_chart=False):
    if segment_col not in df or perf_col not in df:
        return
    g = df.groupby(segment_col)[perf_col].mean().dropna().sort_values(ascending=False)
    if as_percent:
        g = g*100
    g = g[g.index.notnull()]
    st.write(f"**{perf_col.replace('_',' ').capitalize()} by {segment_col}:**")
    st.dataframe(g.head(10).to_frame(), use_container_width=True)
    if show_chart:
        fig = px.bar(g.head(10), x=g.head(10), y=g.head(10).index, orientation='h', text_auto='.2f')
        st.plotly_chart(fig, use_container_width=True)

def show_targeting_breakdown(adset_or_ads_df, perf_df=None, where="adset"):
    st.subheader(f"🎯 Targeting Segment Insights ({where.title()})")
    tcols = get_targeting_columns(adset_or_ads_df)
    if not tcols:
        st.info("No targeting columns found in this data.")
        return
    with st.expander("Show targeting columns, value examples", expanded=False):
        st.write(tcols)
        st.dataframe(adset_or_ads_df[tcols].drop_duplicates().head(20), use_container_width=True)
    # Most frequent values, and (if possible) performance overlays
    for col in tcols:
        col_data = adset_or_ads_df[col].dropna()
        st.markdown(f"**Top Segments: `{col}`** ({col_data.nunique()} unique)")
        val_ct = col_data.value_counts().head(7)
        st.write(val_ct)
        if len(val_ct) > 1:
            fig = px.pie(values=val_ct.values, names=val_ct.index, title=f"{col}: Share of {where}s", hole=.4)
            st.plotly_chart(fig, use_container_width=True)
        if perf_df is not None and col in adset_or_ads_df.columns and "adset_id" in adset_or_ads_df.columns:
            joined = adset_or_ads_df[[col,"adset_id"]].drop_duplicates().merge(perf_df, left_on="adset_id", right_on="adset_id", how="left")
            if "roas_mean" in joined.columns:
                segment_performance_table(joined, col, perf_col="roas_mean", show_chart=True)
            if "ctr_mean" in joined.columns:
                segment_performance_table(joined, col, perf_col="ctr_mean")
        else:
            st.markdown("<span style='color:gray; font-size:12px'>No performance stats available for this segment.</span>", unsafe_allow_html=True)

########################## UI Layer: Sidebar + NAV #############################

campaigns, adsets, ads = load_data()
adsets_campaign, ads_full = build_meta_hierarchy(campaigns, adsets, ads)
st.sidebar.title("📊 Sections")
page = st.sidebar.radio("Select:", ["Overall","Per Account (BUID)"])
ad_date_col = "date_start" if "date_start" in ads_full.columns else ("dt" if "dt" in ads_full.columns else None)

############################# OVERALL PAGE #############################
if page == "Overall":
    st.title("🌍 All Accounts Marketing Dashboard")
    st.markdown("Get a cross-account view of aggregate spend, results, and targeting effectiveness.")

    mask, start, end = date_range_selector(ads_full, ad_date_col)
    all_ads = ads_full.loc[mask]
    all_adsets = adsets_campaign.loc[mask] if "date_start" in adsets_campaign.columns else adsets_campaign

    # 1. Metric Cards
    stats = summarize_metrics(all_ads)
    stats_dict = stats.iloc[0].to_dict()
    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric("Total Spend", f"${stats_dict.get('spend',0):,.0f}")
    m2.metric("Impressions", int(stats_dict.get("impressions",0)))
    m3.metric("Results", int(stats_dict.get("results",0)) if not pd.isnull(stats_dict.get("results",0)) else "-")
    m4.metric("Avg ROAS", f"{stats_dict.get('roas_mean',0):.2f}")
    m5.metric("Avg CPM", f"${stats_dict.get('cpm_mean',0):.2f}")
    m6.metric("Avg CTR", f"{100*stats_dict.get('ctr_mean',0):.2f}%")
    m7.metric("Avg CPA", f"${stats_dict.get('cpa_mean',0):.2f}" if not pd.isnull(stats_dict.get("cpa_mean",None)) else "-")

    # 2. Spend/Result/ROAS Trends Line Chart
    with st.expander("📈 Performance Trends (Time Series)", expanded=True):
        kpis = [("spend","Spend ($)"),("impressions","Impressions"),
                ("roas","ROAS"),("results","Results")]
        tabts = st.tabs([k for (k,_) in kpis])
        for i, (mc, label) in enumerate(kpis):
            mcol = "purchase_roas" if mc=="roas" and "purchase_roas" in all_ads.columns else mc
            ts = aggregate_metric_trend(all_ads, mcol)
            with tabts[i]:
                if not ts.empty:
                    avg = ts[mcol].mean()
                    fig = px.line(ts, x="date_start", y=mcol, title=label,
                                  markers=True, labels={mcol:label}, template="plotly_white")
                    fig.add_hline(y=avg, line_dash="dash", line_color="green", annotation_text="Period Avg", annotation_position="top left")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for this trend.")

    # 3. Top/Bottom Campaigns (sortable table with color)
    st.markdown("#### 🏆 Top Campaigns (by ROAS / Zone)")
    camp_stats = summarize_metrics(all_ads, "campaign_id")
    if not camp_stats.empty:
        ct1, ct2 = st.columns(2)
        top_by_roas = camp_stats.sort_values("roas_mean", ascending=False).head(10)
        bot_by_roas = camp_stats.sort_values("roas_mean", ascending=True).head(10)
        with ct1:
            st.dataframe(top_by_roas.style.applymap(style_zone, subset=["Zone"]), use_container_width=True)
        with ct2:
            st.dataframe(bot_by_roas.style.applymap(style_zone, subset=["Zone"]), use_container_width=True)
    st.caption(
        "🟢 <span class='legend-dot green-dot'></span>Green=Best (ROAS≥1.5, CPM≤$5, CTR≥1%)<br>"
        "🟡 <span class='legend-dot yellow-dot'></span>Yellow=Middle<br>"
        "🔴 <span class='legend-dot red-dot'></span>Red=Underperformer"
        , unsafe_allow_html=True
    )

    # 4. Targeting analysis
    st.header("Targeting: Audience & Placement Insights (All Accounts)")
    show_targeting_breakdown(all_adsets, perf_df=None, where="adset")
    show_targeting_breakdown(all_ads, perf_df=None, where="ad")
    st.markdown("- See which targeting options are used most often and their relative share.<br>- Where mergeable, ROAS by targeting segment appears.", unsafe_allow_html=True)

    # 5. Share/Pie
    zone_counts = camp_stats["Zone"].value_counts() if "Zone" in camp_stats else pd.Series(dtype=int)
    if not zone_counts.empty:
        st.write("#### Zone Health")
        figz = px.pie(zone_counts, values=zone_counts.values, names=zone_counts.index,
            color=zone_counts.index, title="Campaign Health Zones",
            color_discrete_map={"Red":"#e6524e","Yellow":"#eec95e","Green":"#1aaa55"})
        st.plotly_chart(figz, use_container_width=True)

    # 6. Download options
    st.download_button("📥 Download campaign stats CSV", camp_stats.to_csv(index=False), "campaign_stats.csv", "text/csv")
    st.download_button("📥 Download ads data CSV", all_ads.to_csv(index=False), "adstats.csv", "text/csv")

    st.markdown("---")
    with st.expander("Show all raw ad data", expanded=False):
        st.dataframe(all_ads.head(1000), use_container_width=True)

################################# ACCOUNT/BUID PAGE ##############################
if page == "Per Account (BUID)":
    buids = campaigns['buid'].dropna().unique()
    st.title("👤 Per Account Analysis")
    selected_buid = st.sidebar.selectbox("Select BUID", buids)
    user_campaigns = campaigns[campaigns['buid']==selected_buid].copy()
    user_adsets = adsets[adsets['buid']==selected_buid].copy() if "buid" in adsets.columns else pd.DataFrame()
    user_ads = ads_full[ads_full['buid']==selected_buid].copy() if "buid" in ads_full.columns else pd.DataFrame()
    mask, start, end = date_range_selector(user_ads, ad_date_col)
    user_ads = user_ads.loc[mask]
    st.header(f"Profile: {selected_buid}")

    stats = summarize_metrics(user_ads)
    stats_dict = stats.iloc[0].to_dict()
    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric("Spend", f"${stats_dict.get('spend',0):,.0f}")
    m2.metric("Impressions", int(stats_dict.get("impressions",0)))
    m3.metric("Results", int(stats_dict.get("results",0)) if not pd.isnull(stats_dict.get("results",0)) else "-")
    m4.metric("Avg ROAS", f"{stats_dict.get('roas_mean',0):.2f}")
    m5.metric("Avg CPM", f"${stats_dict.get('cpm_mean',0):.2f}")
    m6.metric("Avg CTR", f"{100*stats_dict.get('ctr_mean',0):.2f}%")
    m7.metric("Avg CPA", f"${stats_dict.get('cpa_mean',0):.2f}" if not pd.isnull(stats_dict.get("cpa_mean",None)) else "-")

    camp_stats = summarize_metrics(user_ads, "campaign_id")
    if not camp_stats.empty:
        st.markdown("#### 👀 Campaign Breakdown")
        st.dataframe(camp_stats.style.applymap(style_zone, subset=["Zone"]), use_container_width=True)
    # Trend chart
    with st.expander("📈 Time Trends", expanded=False):
        kpis = [("spend","Spend ($)"),("impressions","Impressions"),
                ("roas","ROAS"),("results","Results")]
        tabts = st.tabs([k for (k,_) in kpis])
        for i, (mc, label) in enumerate(kpis):
            mcol = "purchase_roas" if mc=="roas" and "purchase_roas" in user_ads.columns else mc
            ts = aggregate_metric_trend(user_ads, mcol)
            with tabts[i]:
                if not ts.empty:
                    avg = ts[mcol].mean()
                    fig = px.line(ts, x="date_start", y=mcol, title=label, markers=True, labels={mcol:label}, template="plotly_white")
                    fig.add_hline(y=avg, line_dash="dash", line_color="green", annotation_text="Avg", annotation_position="top left")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for this trend.")

    # Targeting by-account
    st.header("Targeting Segments: What are you using?")
    show_targeting_breakdown(user_adsets, perf_df=camp_stats, where="adset")
    show_targeting_breakdown(user_ads, perf_df=camp_stats, where="ad")

    # Top/Bottom Ads/Campaigns
    st.markdown("#### 🥇 Top & Bottom Campaigns (by ROAS)")
    if not camp_stats.empty and "roas_mean" in camp_stats.columns:
        st.dataframe(camp_stats.sort_values("roas_mean", ascending=False).head(6), use_container_width=True)
        st.dataframe(camp_stats.sort_values("roas_mean", ascending=True).head(6), use_container_width=True)
    # Modification/creative
    if "updated_at" in user_ads.columns and "dt" in user_ads.columns:
        changed = user_ads[user_ads["updated_at"] != user_ads["dt"]]
        if len(changed)>0:
            st.info(f"Found {len(changed)} ads modified after launch.")
            st.dataframe(changed[["campaign_id","ad_account_id","adset_id","updated_at"]], hide_index=True)
    if "creative_link" in user_ads.columns:
        st.header("Ad Creatives")
        st.dataframe(user_ads[["campaign_id","creative_link"]].drop_duplicates(), hide_index=True)
    with st.expander("Show All Ad Data", expanded=False):
        st.dataframe(user_ads.head(1000), use_container_width=True)

st.markdown("---")
st.caption(
    "Built for marketers: actionable, conversion-focused, targeting-centric. | "
    "Campaign → Adset → Ad → Targeting | v3"
)
