from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html

from backend.config import DATA_DIR
from backend.sentiment import PROMPT_TEMPLATES, generate_event_prompt, run_sentiment_analysis
from backend.sentiment_insights import (
    build_china_map_rows,
    generate_ai_sentiment_report,
    report_cache_key,
    render_china_sentiment_map,
    top_words_html,
    with_china_province,
)
from backend.stopwords import parse_stopwords
from backend.utils import render_sidebar_config, load_df, save_df, enrich_province_column

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
render_sidebar_config()

st.title("💬 Sentiment Analysis")
st.caption("Use DeepSeek to classify comment sentiment")

# ─────────────────────────────────────────
# Chart rendering helpers
# ─────────────────────────────────────────

def render_charts(df: pd.DataFrame):
    if "sentiment" not in df.columns:
        st.error("The data is missing the sentiment column")
        return

    LABEL_MAP = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    COLOR_MAP = {"Positive": "#4CAF50", "Neutral": "#9E9E9E", "Negative": "#F44336"}

    df = df.copy()
    df["Sentiment"] = df["sentiment"].map(LABEL_MAP).fillna(df["sentiment"])
    stopwords = parse_stopwords(st.session_state.get("custom_stopwords_text", ""))

    st.divider()
    st.subheader("📊 Sentiment Analysis Results")

    tab1, tab2, tab3, tab4 = st.tabs(["🥧 Overall Distribution", "👥 Gender Differences", "🗺️ Regional Distribution", "🤖 AI Analysis"])

    # ── Tab1: Overall Distribution ──
    with tab1:
        counts = df["Sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        counts["top_words"] = [
            top_words_html(df[df["Sentiment"] == label], stopwords=stopwords)
            for label in counts["Sentiment"].tolist()
        ]

        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = px.pie(
                counts, names="Sentiment", values="Count",
                color="Sentiment", color_discrete_map=COLOR_MAP,
                title="Sentiment Distribution",
                hole=0.4,
                custom_data=["top_words"],
            )
            fig_pie.update_traces(
                hovertemplate="Sentiment=%{label}<br>Count=%{value}<br>Share=%{percent}<br><br>Top 10 Keywords:<br>%{customdata[0]}<extra></extra>",
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_bar = px.bar(
                counts, x="Sentiment", y="Count",
                color="Sentiment", color_discrete_map=COLOR_MAP,
                title="Sentiment Counts",
                text="Count",
                custom_data=["top_words"],
            )
            fig_bar.update_traces(
                hovertemplate="Sentiment=%{x}<br>Count=%{y}<br><br>Top 10 Keywords:<br>%{customdata[0]}<extra></extra>",
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(showlegend=False, height=380, xaxis_title="", yaxis_title="Comments")
            st.plotly_chart(fig_bar, use_container_width=True)

        total = len(df)
        c1, c2, c3 = st.columns(3)
        pos = (df["Sentiment"] == "Positive").sum()
        neu = (df["Sentiment"] == "Neutral").sum()
        neg = (df["Sentiment"] == "Negative").sum()
        c1.metric("Positive Comments", f"{pos}", f"{pos/total*100:.1f}%")
        c2.metric("Neutral Comments", f"{neu}", f"{neu/total*100:.1f}%")
        c3.metric("Negative Comments", f"{neg}", f"{neg/total*100:.1f}%")

    # ── Tab2: Gender Differences ──
    with tab2:
        if "gender" not in df.columns:
            st.info("The data is missing the gender field")
        else:
            gender_map = {"男": "Male", "女": "Female", "保密": "Unknown"}
            df_g = df.copy()
            df_g["Gender"] = df_g["gender"].map(gender_map).fillna("Unknown")

            gender_sentiment = (
                df_g.groupby(["Gender", "Sentiment"])
                .size()
                .reset_index(name="Count")
            )
            valid_genders = df_g["Gender"].value_counts()
            valid_genders = valid_genders[valid_genders >= 3].index.tolist()
            gender_sentiment = gender_sentiment[gender_sentiment["Gender"].isin(valid_genders)]
            gender_sentiment["top_words"] = [
                    top_words_html(
                        df_g[(df_g["Gender"] == row["Gender"]) & (df_g["Sentiment"] == row["Sentiment"])],
                        stopwords=stopwords,
                    )
                for _, row in gender_sentiment.iterrows()
            ]

            if gender_sentiment.empty:
                st.info("Gender samples are insufficient for difference analysis (each group needs at least 3 comments).")
            else:
                fig_g = px.bar(
                    gender_sentiment,
                    x="Gender", y="Count", color="Sentiment",
                    barmode="group",
                    color_discrete_map=COLOR_MAP,
                    title="Sentiment Distribution by Gender (Grouped Bar Chart)",
                    text="Count",
                    custom_data=["top_words"],
                )
                fig_g.update_traces(
                    hovertemplate="Gender=%{x}<br>Sentiment=%{fullData.name}<br>Count=%{y}<br><br>Top 10 Keywords:<br>%{customdata[0]}<extra></extra>",
                )
                fig_g.update_traces(textposition="outside")
                fig_g.update_layout(height=420, xaxis_title="", yaxis_title="Comments")
                st.plotly_chart(fig_g, use_container_width=True)

                pct = gender_sentiment.copy()
                total_by_gender = pct.groupby("Gender")["Count"].transform("sum")
                pct["Share"] = (pct["Count"] / total_by_gender * 100).round(1)
                fig_pct = px.bar(
                    pct,
                    x="Gender", y="Share", color="Sentiment",
                    barmode="stack",
                    color_discrete_map=COLOR_MAP,
                    title="Sentiment Share by Gender (Stacked Chart)",
                    text="Share",
                    custom_data=["top_words"],
                )
                fig_pct.update_traces(
                    hovertemplate="Gender=%{x}<br>Sentiment=%{fullData.name}<br>Share=%{y:.1f}%<br><br>Top 10 Keywords:<br>%{customdata[0]}<extra></extra>",
                )
                fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
                fig_pct.update_layout(height=380, xaxis_title="", yaxis_title="Share (%)")
                st.plotly_chart(fig_pct, use_container_width=True)

    # ── Tab3: Regional Distribution ──
    with tab3:
        if "ip_province" not in df.columns:
            st.info("The data is missing the region field")
        else:
            china_df = with_china_province(df)
            china_valid = china_df[china_df["china_province"].notna()].copy()
            map_rows = build_china_map_rows(df, stopwords=stopwords)
            if map_rows:
                st.markdown("#### Positive Sentiment Heatmap by China Region")
                html(render_china_sentiment_map(map_rows), height=660)
            else:
                st.info("No Chinese province-level positive/negative comment data is available for the map.")

            province_counts = (
                china_valid["china_province"]
                .value_counts()
                .reset_index()
            )
            province_counts.columns = ["Province/Region", "Comments"]
            province_counts = province_counts.head(20)
            province_counts["top_words"] = [
                top_words_html(china_valid[china_valid["china_province"] == region], stopwords=stopwords)
                for region in province_counts["Province/Region"].tolist()
            ]

            if province_counts.empty:
                st.info("No valid region data is available")
            else:
                fig_map = px.bar(
                    province_counts,
                    x="Comments", y="Province/Region",
                    orientation="h",
                    title=f"Comment Region Distribution Top {len(province_counts)}",
                    color="Comments",
                    color_continuous_scale="Blues",
                    text="Comments",
                    custom_data=["top_words"],
                )
                fig_map.update_traces(
                    hovertemplate="Province/Region=%{y}<br>Comments=%{x}<br><br>Top 10 Keywords:<br>%{customdata[0]}<extra></extra>",
                )
                fig_map.update_traces(textposition="outside")
                fig_map.update_layout(
                    height=max(400, len(province_counts) * 28),
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    xaxis_title="Comments",
                    yaxis_title="",
                )
                st.plotly_chart(fig_map, use_container_width=True)

                top8 = province_counts.head(8)["Province/Region"].tolist()
                df_top8 = china_valid[china_valid["china_province"].isin(top8)]
                region_sent = (
                    df_top8.groupby(["china_province", "Sentiment"])
                    .size()
                    .reset_index(name="Count")
                    .rename(columns={"china_province": "Region"})
                )
                region_sent["top_words"] = [
                    top_words_html(
                        df_top8[(df_top8["china_province"] == row["Region"]) & (df_top8["Sentiment"] == row["Sentiment"])],
                        stopwords=stopwords,
                    )
                    for _, row in region_sent.iterrows()
                ]
                if not region_sent.empty:
                    fig_rs = px.bar(
                        region_sent,
                        x="Region", y="Count", color="Sentiment",
                        barmode="stack",
                        color_discrete_map=COLOR_MAP,
                        title="Sentiment Composition of Top 8 Provinces",
                        custom_data=["top_words"],
                    )
                    fig_rs.update_traces(
                        hovertemplate="Region=%{x}<br>Sentiment=%{fullData.name}<br>Count=%{y}<br><br>Top 10 Keywords:<br>%{customdata[0]}<extra></extra>",
                    )
                    fig_rs.update_layout(height=400, xaxis_title="", yaxis_title="Comments")
                    st.plotly_chart(fig_rs, use_container_width=True)

    with tab4:
        st.markdown("#### AI Sentiment Difference Analysis")
        api_key = st.session_state.get("deepseek_key", "").strip()
        stopwords_text = st.session_state.get("custom_stopwords_text", "")
        cache_key = report_cache_key(df, stopwords_text=stopwords_text)
        cached_key = st.session_state.get("sentiment_ai_report_key")
        cached_report = st.session_state.get("sentiment_ai_report", "")
        if cached_report and cached_key == cache_key:
            st.markdown(cached_report)
        if not api_key:
            st.warning("Please fill in the DeepSeek API Key in the sidebar system settings first.")
        if st.button("Generate AI Analysis Report", disabled=not api_key):
            with st.spinner("Generating overall, gender, and regional difference analysis..."):
                try:
                    report = generate_ai_sentiment_report(df, api_key, stopwords=stopwords)
                except Exception as e:
                    st.error(f"AI analysis generation failed: {e}")
                else:
                    st.session_state["sentiment_ai_report_key"] = cache_key
                    st.session_state["sentiment_ai_report"] = report
                    st.markdown(report)

    # ── Download ──
    st.divider()
    sentiment_file = st.session_state.get("sentiment_file")
    if sentiment_file:
        result_csv = load_df(sentiment_file)
        if result_csv is not None:
            st.download_button(
                "⬇️ Download Sentiment Analysis CSV",
                data=result_csv.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"sentiment_{st.session_state.get('current_bvid', 'result')}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────
# Page body
# ─────────────────────────────────────────

data_file = st.session_state.get("data_file")
if not data_file:
    st.warning("⚠️ No data has been crawled yet. Please go to **📥 Data Crawling** first.")
    st.stop()

raw_df = load_df(data_file)
if raw_df is None or raw_df.empty:
    st.error("The data file could not be read. Please crawl again.")
    st.stop()

vtitle = raw_df["video_title"].iloc[0][:30] if "video_title" in raw_df.columns else ""
st.info(f"Current data: **{len(raw_df)}** comments | Video: {vtitle}")
st.divider()

video_meta_cols = [
    "video_url", "bvid", "aid", "video_title", "video_desc", "video_pubdate",
    "video_duration", "video_tname", "up_name", "up_mid", "view_count",
    "like_count_video", "coin_count", "favorite_count", "share_count", "reply_count",
]
video_meta = {
    col: raw_df[col].iloc[0]
    for col in video_meta_cols
    if col in raw_df.columns
}
if video_meta:
    st.session_state["video_meta"] = video_meta

# ── Prompt Settings ──
st.subheader("📝 Prompt Settings")

template_name = st.selectbox(
    "Select Built-in Prompt Template",
    options=list(PROMPT_TEMPLATES.keys()),
    help=(
        "Event-level sentiment analysis judges comment stance around the specific event, disputed object, or public issue discussed in the video. "
        "It suits trending events, product launches, and social controversy. "
        "General sentiment analysis judges positive, neutral, or negative sentiment from the comment text itself. "
        "It suits everyday topics, entertainment content, or comments without a clear event object."
    ),
)

if (
    st.session_state.get("sentiment_prompt_template") != template_name
    or "sentiment_custom_prompt" not in st.session_state
):
    st.session_state["sentiment_prompt_template"] = template_name
    st.session_state["sentiment_custom_prompt"] = PROMPT_TEMPLATES[template_name]

api_key = st.session_state.get("deepseek_key", "").strip()
is_event_template = template_name.startswith("Event-Level Sentiment Analysis")
if is_event_template:
    video_summary = st.session_state.get("subtitle_summary", "").strip()
    if video_summary:
        st.success("A video summary was detected and will be used to generate a prompt better aligned with this video event.")
    else:
        st.info("It is recommended to generate a video summary on the **Video Summary** page first. The AI can use it to identify the core event and produce a more accurate event-level prompt.")

    sample_comments = (
        raw_df["content"]
        .dropna()
        .astype(str)
        .head(40)
        .tolist()
        if "content" in raw_df.columns
        else []
    )
    gen_disabled = not api_key
    if st.button("Generate Prompt with AI", disabled=gen_disabled):
        with st.spinner("Generating an event-level prompt from video context..."):
            try:
                generated_prompt = generate_event_prompt(
                    video_meta=video_meta or st.session_state.get("video_meta", {}),
                    video_summary=video_summary,
                    sample_comments=sample_comments,
                    api_key=api_key,
                    base_template=PROMPT_TEMPLATES[template_name],
                )
            except Exception as e:
                st.error(f"Prompt generation failed: {e}")
            else:
                if generated_prompt:
                    st.session_state["sentiment_custom_prompt"] = generated_prompt
                    st.success("The prompt has been generated and filled into the text box below. You can edit it before starting analysis.")
                    st.rerun()
                else:
                    st.error("No valid prompt was generated. Please check the DeepSeek API Key or try again later.")
    if gen_disabled:
        st.caption("Fill in the DeepSeek API Key to generate prompts with AI.")

custom_prompt = st.text_area(
    "Prompt Content (editable)",
    key="sentiment_custom_prompt",
    height=250,
)

# ── Analysis Settings ──
st.subheader("⚙️ Analysis Settings")
p1, p2, p3 = st.columns(3)
with p1:
    total_comments = len(raw_df)
    min_comments = 1 if total_comments < 20 else 20
    step_comments = 1 if total_comments < 20 else 20
    default_comments = min(200, total_comments)
    default_comments = max(min_comments, default_comments)
    max_comments = st.slider(
        "Maximum Comments to Analyze",
        min_value=min_comments,
        max_value=total_comments,
        value=default_comments,
        step=step_comments,
        help="The upper limit is the current number of crawled comments. More comments mean more API calls and longer processing time.",
    )
with p2:
    st.metric("Estimated DeepSeek API Calls", max_comments)
with p3:
    max_workers = st.select_slider(
        "Concurrency",
        options=[1, 2, 3, 4, 6, 8],
        value=4,
        help="Higher concurrency is faster but more likely to trigger API rate limits. Reduce it if failures or rate limits occur.",
    )

if not api_key:
    st.warning("⚠️ DeepSeek API Key is not configured. Please fill it in under **⚙️ System Settings** in the sidebar.")

st.divider()

# ── 开始分析 ──
can_analyze = bool(api_key) and bool(custom_prompt.strip())

if st.button("🚀 Start Sentiment Analysis", type="primary", disabled=not can_analyze):
    progress_bar = st.progress(0.0, text="Preparing...")
    status_text = st.empty()
    labels = []

    try:
        for progress, message, current_labels in run_sentiment_analysis(
            raw_df,
            custom_prompt.strip(),
            api_key,
            max_comments=max_comments,
            max_workers=max_workers,
        ):
            progress_bar.progress(min(progress, 1.0), text=message)
            status_text.caption(message)
            if current_labels:
                labels = current_labels

        if labels:
            result_df = raw_df.head(len(labels)).copy()
            result_df["sentiment"] = labels
            result_df = enrich_province_column(result_df)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bvid = st.session_state.get("current_bvid", "unknown")
            filepath = DATA_DIR / f"sentiment_{bvid}_{ts}.csv"
            save_df(result_df, filepath)
            st.session_state["sentiment_file"] = str(filepath)

            progress_bar.progress(1.0, text="Analysis complete!")
            status_text.empty()
            st.success(f"✅ Sentiment analysis complete! Analyzed **{len(labels)}** comments")
            render_charts(result_df)
        else:
            st.error("No analysis result was returned. Please check whether the API Key is valid.")

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {e}")

else:
    # Show existing results
    sentiment_file = st.session_state.get("sentiment_file")
    if sentiment_file:
        sent_df = load_df(sentiment_file)
        if sent_df is not None and not sent_df.empty and "sentiment" in sent_df.columns:
            sent_df = enrich_province_column(sent_df)
            st.success(f"Showing existing analysis results ({len(sent_df)} rows). Click the button above to rerun analysis.")
            render_charts(sent_df)
