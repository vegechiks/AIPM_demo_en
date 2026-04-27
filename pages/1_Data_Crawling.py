import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.bilibili_crawler import extract_bvid, validate_bilibili_input, crawl_bilibili
from backend.config import DATA_DIR
from backend.utils import render_sidebar_config, save_df

st.set_page_config(page_title="Data Crawling", page_icon="📥", layout="wide")
render_sidebar_config()

st.title("📥 Data Crawling")
st.caption("Current version supports: Bilibili")

st.divider()

# ── Platform selection ──
col_platform, col_blank = st.columns([1, 3])
with col_platform:
    platform = st.selectbox(
        "Select Platform",
        options=["Bilibili", "Douyin (in development)", "TikTok (in development)", "YouTube (in development)"],
    )

disabled = platform != "Bilibili"
if disabled:
    st.warning("Only Bilibili is supported in the current version. Other platforms are in development.")

# ── Video URL input ──
st.subheader("Video URL")
url_col, hint_col = st.columns([3, 2])
with url_col:
    video_url = st.text_input(
        "Enter a Bilibili video URL or BV ID",
        placeholder="https://www.bilibili.com/video/BV1xx411c7mD  or  BV1xx411c7mD",
        disabled=disabled,
    )

# Live format validation
with hint_col:
    st.write("")  # placeholder
    if video_url.strip():
        valid, err_msg = validate_bilibili_input(video_url)
        if valid:
            bvid = extract_bvid(video_url)
            st.success(f"✅ Recognized BV ID: `{bvid}`")
        else:
            st.error(f"❌ {err_msg}")
    else:
        st.caption("Please enter a video URL")

# ── Crawl Settings ──
st.subheader("Crawl Settings")
param_col1, param_col2 = st.columns(2)
with param_col1:
    max_pages = st.slider(
        "Maximum Pages",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="Bilibili returns about 20 comments per page; 30 pages is about 600 comments",
        disabled=disabled,
    )
with param_col2:
    st.metric("Estimated Maximum Comments", f"About {max_pages * 20}")
    st.caption("Only top-level comments are counted. Nested replies are excluded. Comments are fetched by popularity first.")

# ── Cookie reminder ──
cookie = st.session_state.get("bili_cookie", "").strip()
if not cookie:
    st.warning("⚠️ Bilibili Cookie is not configured. Fill it in under **⚙️ System Settings** in the sidebar; otherwise crawling may fail or return incomplete data.")

st.divider()

# ── Start crawling button ──
can_crawl = (
    not disabled
    and video_url.strip()
    and validate_bilibili_input(video_url)[0]
)

if st.button("🚀 Start Crawling", type="primary", disabled=not can_crawl):
    bvid = extract_bvid(video_url)
    cookie = st.session_state.get("bili_cookie", "").strip()

    st.divider()
    progress_bar = st.progress(0.0, text="Preparing...")
    status_text = st.empty()
    result_placeholder = st.empty()

    all_comments = []
    final_message = ""

    try:
        for progress, message, comments in crawl_bilibili(bvid, cookie, max_pages=max_pages):
            progress_bar.progress(min(progress, 1.0), text=message)
            status_text.caption(message)
            if comments:
                all_comments = comments
            if progress >= 1.0:
                final_message = message

        if all_comments:
            df = pd.DataFrame(all_comments)

            # Save CSV
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comments_{bvid}_{ts}.csv"
            filepath = DATA_DIR / filename
            save_df(df, filepath)

            # Update session_state
            st.session_state["data_file"] = str(filepath)
            st.session_state["current_bvid"] = bvid
            st.session_state["video_title"] = df["video_title"].iloc[0] if "video_title" in df.columns else bvid
            meta_cols = [
                "video_url", "bvid", "aid", "video_title", "video_desc", "video_pubdate",
                "video_duration", "video_tname", "up_name", "up_mid", "view_count",
                "like_count_video", "coin_count", "favorite_count", "share_count", "reply_count",
            ]
            st.session_state["video_meta"] = {
                col: df[col].iloc[0]
                for col in meta_cols
                if col in df.columns
            }
            # Clear previous analysis results for a new crawl
            st.session_state["sentiment_file"] = None
            st.session_state["topic_result"] = None

            progress_bar.progress(1.0, text="Crawl complete!")
            status_text.empty()

            result_placeholder.success(
                f"✅ Crawl succeeded! Retrieved **{len(df)}** comments and saved to `{filename}`"
            )

            # Show data preview
            with st.expander("📋 Data Preview (first 10 rows)", expanded=True):
                preview_cols = ["username", "content", "like_count", "gender", "ip_location", "comment_time"]
                show_cols = [c for c in preview_cols if c in df.columns]
                st.dataframe(df[show_cols].head(10), use_container_width=True)

            st.info("👉 Open **Data View** in the sidebar to review the full dataset, or continue to **Sentiment Analysis**")

        else:
            error_msg = final_message or "No comments were retrieved"
            result_placeholder.error(error_msg)

    except Exception as e:
        st.error(f"❌ An error occurred during crawling: {e}")
