from pathlib import Path

import streamlit as st

from backend.stopwords import default_stopwords_text

st.set_page_config(
    page_title="Short-Video Comment AI Analysis Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化 session_state
st.session_state.setdefault("bili_cookie", "")
st.session_state.setdefault("deepseek_key", "")
st.session_state.setdefault("openai_asr_key", "")
st.session_state.setdefault("data_file", None)
st.session_state.setdefault("sentiment_file", None)
st.session_state.setdefault("topic_result", None)
st.session_state.setdefault("current_bvid", "")
st.session_state.setdefault("video_title", "")
st.session_state.setdefault("video_meta", {})
st.session_state.setdefault("sentiment_prompt_template", "")
st.session_state.setdefault("sentiment_custom_prompt", "")
st.session_state.setdefault("custom_stopwords_text", default_stopwords_text())

# 侧边栏
from backend.utils import render_sidebar_config

render_sidebar_config()

# 主页内容
st.title("🔍 Short-Video Comment AI Analysis Platform")
st.markdown(
    """
    > This product demo extends the experimental findings from my graduation thesis.
    > The thesis compared multiple sentiment analysis and topic modeling methods for short-video comments.
    > The results showed that **LLMs perform best for event-level sentiment classification**, while **BTM (Biterm Topic Model) is better suited to short-text comment topic modeling**.
    """
)

st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown("### 📥 Step 1\n**Data Crawling**\n\nEnter a Bilibili video URL to crawl top-level comments")
with col2:
    st.markdown("### 📋 Step 2\n**Data View**\n\nReview comments, likes, timestamps, IP location, and other metadata")
with col3:
    st.markdown("### 💬 Step 3\n**Sentiment Analysis**\n\nUse DeepSeek to classify event-level comment sentiment")
with col4:
    st.markdown("### 🔬 Step 4\n**Topic Analysis**\n\nUse BTM to discover high-frequency topics in short comments")
with col5:
    st.markdown("### 📝 Extension\n**Video Summary**\n\nGenerate a content summary and timeline from video subtitles")

st.divider()

st.info(
    "👈 Select a feature page from the left navigation to begin  \n"
    "For first-time use, fill in the Bilibili Cookie and DeepSeek API Key under **⚙️ System Settings** in the sidebar"
)

with st.expander("ℹ️ About This Project"):
    st.markdown(
        """
        **Project Overview**
        This project does not directly reproduce the specific research topic from the thesis. Instead, it combines the validated methods
        into an interactive demo for short-video comment analysis.
        Users can enter a Bilibili video URL and complete comment crawling, data review, sentiment analysis, topic modeling, and subtitle-based video summarization.

        **Method Background**
        The thesis compared multiple sentiment analysis and topic modeling methods on short-video comments.
        Based on the experimental results, this demo uses the following technical route:

        - Sentiment analysis: LLMs performed best on event-level sentiment classification, so this demo uses DeepSeek for comment sentiment judgment.
        - Topic modeling: BTM (Biterm Topic Model) is better suited to short-text comments, so this demo uses BTM to discover comment topics.

        **Data Flow**
        `Video URL` -> `Top-level comment crawl (CSV)` -> `Data view` -> `Sentiment analysis (CSV)` -> `Topic modeling (CSV)` -> `Visual analysis`

        `Video URL` -> `Subtitle retrieval` -> `AI video summary`

        **Version Notes**
        The current version supports Bilibili comment crawling. It crawls top-level video comments only and does not include nested replies.
        Sentiment and topic analysis are mainly designed for Chinese short-text comments. Video summaries depend on Bilibili subtitles; if no subtitle is available, summary generation is unavailable unless ASR is used.
        """
    )

readme_path = Path(__file__).resolve().parent / "README.md"
with st.expander("📘 Product Guide"):
    if readme_path.exists():
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.warning("README.md was not found, so the product guide cannot be loaded.")
