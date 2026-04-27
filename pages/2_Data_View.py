import html

import pandas as pd
import streamlit as st

from backend.utils import render_sidebar_config, load_df, enrich_province_column

st.set_page_config(page_title="Data View", page_icon="📋", layout="wide")
render_sidebar_config()

st.title("📋 Data View")
st.caption("Review the crawled raw comment data")

# ── 数据检查 ──
data_file = st.session_state.get("data_file")
if not data_file:
    st.warning("⚠️ No data has been crawled yet. Please go to **📥 Data Crawling** first.")
    st.stop()

df = load_df(data_file)
if df is None or df.empty:
    st.error("The data file could not be read or is empty. Please crawl again.")
    st.session_state["data_file"] = None
    st.stop()

df = enrich_province_column(df)

# ── 顶部统计卡片 ──
st.divider()
st.markdown(
    """
    <style>
    .metric-tooltip {
        min-width: 0;
        position: relative;
    }
    .metric-tooltip__label {
        color: rgba(49, 51, 63, 0.72);
        font-size: 0.875rem;
        line-height: 1.25;
        margin-bottom: 0.25rem;
    }
    .metric-tooltip__value {
        color: rgb(49, 51, 63);
        font-size: 2.25rem;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        cursor: default;
    }
    .metric-tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 0;
        top: calc(100% + 0.5rem);
        z-index: 9999;
        width: max-content;
        max-width: min(36rem, 70vw);
        padding: 0.55rem 0.7rem;
        border-radius: 6px;
        background: rgba(17, 24, 39, 0.96);
        color: #fff;
        font-size: 0.9rem;
        line-height: 1.45;
        white-space: normal;
        overflow-wrap: anywhere;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_metric(label: str, value: object, tooltip: object | None = None) -> None:
    value_text = str(value)
    tooltip_text = str(tooltip if tooltip is not None else value_text)
    st.markdown(
        f"""
        <div class="metric-tooltip" data-tooltip="{html.escape(tooltip_text, quote=True)}">
            <div class="metric-tooltip__label">{html.escape(label)}</div>
            <div class="metric-tooltip__value">{html.escape(value_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


video_title = df["video_title"].iloc[0] if "video_title" in df.columns else "—"
bvid = df["bvid"].iloc[0] if "bvid" in df.columns else "—"

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    render_metric("Total Comments", f"{len(df):,}")
with m2:
    render_metric("Video Title", video_title, video_title)
with m3:
    render_metric("BV ID", bvid, bvid)
male = (df["gender"] == "男").sum() if "gender" in df.columns else 0
female = (df["gender"] == "女").sum() if "gender" in df.columns else 0
with m4:
    render_metric("Male / Female", f"{male} / {female}")
provinces = df["ip_province"].nunique() if "ip_province" in df.columns else 0
with m5:
    render_metric("Regions Covered", provinces)

st.divider()

# ── 筛选控制栏 ──
filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])

with filter_col1:
    search_kw = st.text_input("🔍 Keyword Search (comment content)", placeholder="Enter keywords to filter...")

with filter_col2:
    gender_options = ["All"] + sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else ["All"]
    gender_filter = st.selectbox("Gender Filter", gender_options)

with filter_col3:
    sort_by = st.selectbox(
        "Sort By",
        ["Newest comments first", "Oldest comments first", "Most likes first", "Fewest likes first"],
    )

# ── 应用筛选 ──
filtered_df = df.copy()

if search_kw.strip():
    filtered_df = filtered_df[
        filtered_df["content"].astype(str).str.contains(search_kw.strip(), case=False, na=False)
    ]

if gender_filter != "All" and "gender" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["gender"] == gender_filter]

if sort_by in ["Newest comments first", "Oldest comments first"] and "comment_time" in filtered_df.columns:
    filtered_df = filtered_df.assign(
        _comment_time_sort=pd.to_datetime(filtered_df["comment_time"], errors="coerce")
    ).sort_values(
        "_comment_time_sort",
        ascending=(sort_by == "Oldest comments first"),
        na_position="last",
    ).drop(columns=["_comment_time_sort"])
elif sort_by == "Most likes first" and "like_count" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("like_count", ascending=False)
elif sort_by == "Fewest likes first" and "like_count" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("like_count", ascending=True)

st.caption(f"Showing {len(filtered_df)} records (total {len(df)})")

# ── 数据表格 ──
display_cols_map = {
    "username": "Username",
    "content": "Comment",
    "like_count": "Likes",
    "gender": "Gender",
    "ip_province": "IP Location",
    "comment_time": "Comment Time",
}
available_display_cols = [c for c in display_cols_map if c in filtered_df.columns]
show_df = filtered_df[available_display_cols].rename(columns=display_cols_map)

st.dataframe(
    show_df,
    use_container_width=True,
    height=500,
    column_config={
        "Comment": st.column_config.TextColumn(width="large"),
        "Likes": st.column_config.NumberColumn(format="%d"),
    },
)

# ── 下载按钮 ──
csv_bytes = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    label="⬇️ Download Filtered CSV",
    data=csv_bytes,
    file_name=f"comments_{st.session_state.get('current_bvid', 'export')}.csv",
    mime="text/csv",
)
