import pandas as pd
import plotly.express as px
import streamlit as st

from backend.topic_insights import generate_topic_ai_report, topic_report_cache_key
from backend.topic_model import run_topic_analysis_sync
from backend.utils import (
    render_sidebar_config,
    load_df,
    save_df,
    enrich_province_column,
    generate_wordcloud_image,
    ai_name_topics,
)
from backend.config import DATA_DIR

st.set_page_config(page_title="Topic Analysis", page_icon="🔬", layout="wide")
render_sidebar_config()

st.title("🔬 Topic Analysis")
st.caption("Use BTM (Biterm Topic Model) to discover hot topics in comments")

# ─────────────────────────────────────────
# Chart/result rendering helpers
# ─────────────────────────────────────────

def render_topic_results(result: dict):
    topic_words_df: pd.DataFrame = result["topic_words_df"]
    doc_topic_df: pd.DataFrame = result["doc_topic_df"]
    word_freq: dict = result["word_freq"]
    n_docs: int = result["n_docs"]

    st.divider()
    st.subheader("📊 Topic Analysis Results")

    tab1, tab2, tab3, tab4 = st.tabs(["☁️ Word Cloud", "📋 Topic Keywords", "📈 Topic Distribution", "🤖 AI Analysis"])

    # ── Tab1: Word Cloud ──
    with tab1:
        st.markdown("**Full Word-Frequency Cloud** (based on tokenized results from all comments)")
        if word_freq:
            img_bytes = generate_wordcloud_image(word_freq)
            if img_bytes:
                st.image(img_bytes, use_container_width=True)
            else:
                st.warning(
                    "⚠️ No Chinese font was detected, so the word cloud cannot be generated.\n\n"
                    "For local runs, make sure SimHei or Microsoft YaHei is installed."
                    "For cloud deployment, add `fonts-wqy-zenhei` to packages.txt."
                )
                # 降级展示Top 30 Words by Frequency
                freq_df = (
                    pd.DataFrame(list(word_freq.items()), columns=["Word", "Frequency"])
                    .sort_values("Frequency", ascending=False)
                    .head(30)
                )
                fig = px.bar(
                    freq_df, x="Frequency", y="Word", orientation="h",
                    title="Top 30 Words by Frequency",
                    color="Frequency", color_continuous_scale="Blues",
                )
                fig.update_layout(
                    height=600,
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No word-frequency data is available")

    # ── Tab2: Topic Keywords ──
    with tab2:
        st.markdown(f"**BTM identified {len(topic_words_df)} topics. Each topic shows the top 10 keywords.**")

        # 判断是否已经过 AI 命名
        has_ai_name = (
            "topic_description" in topic_words_df.columns
            and topic_words_df["topic_description"].notna().any()
            and (topic_words_df["topic_description"] != "").any()
        )

        # AI naming button
        api_key = st.session_state.get("deepseek_key", "").strip()
        if not has_ai_name:
            if api_key:
                if st.button("✨ AI Analysis: Auto-name Each Topic", type="secondary"):
                    with st.spinner("DeepSeek is analyzing topic keywords. Please wait..."):
                        try:
                            named_df = ai_name_topics(topic_words_df, api_key)
                            # 更新 session_state 中的结果
                            result["topic_words_df"] = named_df
                            st.session_state["topic_result"] = result
                            topic_words_df = named_df
                            has_ai_name = True
                            st.success("✅ AI naming complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI naming failed: {e}")
            else:
                st.info("Configure the DeepSeek API Key to automatically name topics with AI.")

        # Show each topic
        num_topics = len(topic_words_df)
        cols_per_row = min(3, num_topics)
        rows = (num_topics + cols_per_row - 1) // cols_per_row

        for r in range(rows):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r * cols_per_row + c
                if idx >= num_topics:
                    break
                row = topic_words_df.iloc[idx]
                topic_name = row.get("topic_name") or f"Topic {int(row['topic_id']) + 1}"
                topic_desc = row.get("topic_description") or ""
                keywords = row.get("keywords") or ""

                with cols[c]:
                    with st.container(border=True):
                        st.markdown(f"**{topic_name}**")
                        if topic_desc:
                            st.caption(topic_desc)
                        st.markdown(
                            " ".join(
                                f"`{w}`"
                                for w in keywords.split("、")
                                if w.strip()
                            )
                        )

        # Full DataFrame
        with st.expander("View Full Topic Keyword Table"):
            display_cols = [c for c in ["topic_id", "topic_name", "topic_description", "keywords"] if c in topic_words_df.columns]
            st.dataframe(topic_words_df[display_cols], use_container_width=True)

    # ── Tab3: Topic Distribution ──
    with tab3:
        if "dominant_topic" in doc_topic_df.columns:
            # Merge topic names
            name_map = {
                int(row["topic_id"]): (row.get("topic_name") or f"Topic {int(row['topic_id'])+1}")
                for _, row in topic_words_df.iterrows()
            }
            doc_topic_df = doc_topic_df.copy()
            doc_topic_df["Topic Name"] = doc_topic_df["dominant_topic"].map(name_map)

            dist = doc_topic_df["Topic Name"].value_counts().reset_index()
            dist.columns = ["Topic", "Comments"]
            dist["Share"] = (dist["Comments"] / dist["Comments"].sum() * 100).round(1)

            col_a, col_b = st.columns(2)
            with col_a:
                fig_pie = px.pie(
                    dist, names="Topic", values="Comments",
                    title="Comment Share by Topic",
                    hole=0.4,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(height=380)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                fig_bar = px.bar(
                    dist, x="Topic", y="Comments",
                    title="Comment Count by Topic",
                    text="Comments",
                    color="Comments",
                    color_continuous_scale="Blues",
                )
                fig_bar.update_traces(textposition="outside")
                fig_bar.update_layout(
                    height=380,
                    xaxis_title="",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Topic x sentiment cross analysis (if sentiment data exists)
            if "sentiment" in doc_topic_df.columns:
                st.markdown("**Topic × Sentiment Cross Analysis**")
                LABEL_MAP = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
                COLOR_MAP = {"Positive": "#4CAF50", "Neutral": "#9E9E9E", "Negative": "#F44336"}
                doc_topic_df["Sentiment"] = doc_topic_df["sentiment"].map(LABEL_MAP).fillna(doc_topic_df["sentiment"])
                cross = (
                    doc_topic_df.groupby(["Topic Name", "Sentiment"])
                    .size()
                    .reset_index(name="Count")
                )
                fig_cross = px.bar(
                    cross, x="Topic Name", y="Count", color="Sentiment",
                    barmode="stack",
                    color_discrete_map=COLOR_MAP,
                    title="Sentiment Composition by Topic",
                )
                fig_cross.update_layout(height=420, xaxis_title="")
                st.plotly_chart(fig_cross, use_container_width=True)

        else:
            st.info("No document-topic assignment data is available")

    # ── Tab4: AI Analysis ──
    with tab4:
        st.markdown("#### AI Topic Analysis Report")
        st.caption(
            "AI will generate an explanatory report using topic keywords, topic shares, representative comments, and optional sentiment labels. "
            "If sentiment analysis has been completed, the report will include a Topic × Sentiment interpretation."
        )
        api_key = st.session_state.get("deepseek_key", "").strip()
        video_meta = st.session_state.get("video_meta", {})
        try:
            cache_key = topic_report_cache_key(result, video_meta=video_meta)
        except Exception as e:
            cache_key = ""
            st.warning(f"AI analysis cache generation failed. Switched to non-cached mode: {e}")
        cached_key = st.session_state.get("topic_ai_report_key")
        cached_report = st.session_state.get("topic_ai_report", "")

        if cached_report and cached_key == cache_key:
            st.markdown(cached_report)

        if not api_key:
            st.warning("Please fill in the DeepSeek API Key in the sidebar system settings first.")
        if "sentiment" not in doc_topic_df.columns:
            st.info("No sentiment analysis result was detected. The report can still explain topics, but it will not include topic-level sentiment differences.")

        if st.button("Generate AI Topic Analysis Report", disabled=not api_key):
            with st.spinner("DeepSeek is generating the topic analysis report..."):
                try:
                    report = generate_topic_ai_report(result, api_key, video_meta=video_meta)
                except Exception as e:
                    st.error(f"AI topic analysis report generation failed: {e}")
                else:
                    st.session_state["topic_ai_report_key"] = cache_key
                    st.session_state["topic_ai_report"] = report
                    st.markdown(report)

    # ── Download ──
    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download Topic Keywords CSV",
            data=topic_words_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"topic_words_{st.session_state.get('current_bvid', 'result')}.csv",
            mime="text/csv",
        )
    with dl2:
        if "dominant_topic" in doc_topic_df.columns:
            st.download_button(
                "⬇️ Download Comment-Topic Assignment CSV",
                data=doc_topic_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"topics_{st.session_state.get('current_bvid', 'result')}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────
# Page body
# ─────────────────────────────────────────

data_file = st.session_state.get("data_file")
if not data_file:
    st.warning("⚠️ No data has been crawled yet. Please go to **📥 Data Crawling** first.")
    st.stop()

# Prefer sentiment results when available; otherwise read raw data
sentiment_file = st.session_state.get("sentiment_file")
if sentiment_file:
    source_df = load_df(sentiment_file)
    data_source_label = "Sentiment Analysis Result (with labels)"
else:
    source_df = load_df(data_file)
    data_source_label = "Raw Crawled Data"

if source_df is None or source_df.empty:
    st.error("The data file could not be read. Please crawl again.")
    st.stop()

source_df = enrich_province_column(source_df)
vtitle = source_df["video_title"].iloc[0][:30] if "video_title" in source_df.columns else ""
st.info(f"Current data source: **{data_source_label}** | {len(source_df)} rows | Video: {vtitle}")

if not sentiment_file:
    st.warning("💡 Complete **💬 Sentiment Analysis** before topic analysis to get the Topic × Sentiment cross view.")

st.divider()

# ── Topic Settings ──
st.subheader("⚙️ Topic Analysis Settings")
st.info(
    "Stopwords are common words filtered during tokenization and topic modeling. They directly affect the word cloud, topic keywords, and topic distribution. "
    "Maintain global stopwords in **System Settings** in the sidebar. Adding meaningless filler words, meme words, or platform-specific noise for the current video usually improves topic quality."
)

p1, p2 = st.columns(2)
with p1:
    num_topics = st.slider(
        "Number of Topics (K)",
        min_value=2,
        max_value=10,
        value=5,
        help="Recommended by comment count: < 100 -> K=3, 100-500 -> K=5, > 500 -> K=6-8",
    )
with p2:
    iterations = st.slider(
        "Iterations",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="More iterations usually produce more stable results but take longer.",
    )

st.divider()

if st.button("🚀 Start Topic Analysis", type="primary"):
    with st.status("Running topic analysis...", expanded=True) as status:
        st.write("Preprocessing comment text...")
        try:
            result = run_topic_analysis_sync(
                source_df,
                num_topics=num_topics,
                extra_stopwords_str=st.session_state.get("custom_stopwords_text", ""),
                iterations=iterations,
            )
            st.write(f"✅ BTM model training complete, covering {result['n_docs']} valid documents")
            status.update(label="Topic analysis complete!", state="complete")
        except Exception as e:
            status.update(label="Analysis failed", state="error")
            st.error(f"❌ {e}")
            st.stop()

    st.session_state["topic_result"] = result
    st.session_state["topic_ai_report"] = ""
    st.session_state["topic_ai_report_key"] = ""
    st.success(f"✅ Topic analysis complete! Found **{num_topics}** topics, covering **{result['n_docs']}** comments")
    render_topic_results(result)

else:
    topic_result = st.session_state.get("topic_result")
    if topic_result:
        st.success(f"Showing existing topic analysis results. Click the button above to rerun analysis.")
        render_topic_results(topic_result)
