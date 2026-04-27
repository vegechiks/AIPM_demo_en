"""
共享工具函数：字体检测、IP省份解析、词云生成、AI主题命名、侧边栏配置渲染
"""
from __future__ import annotations

import io
import html
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from backend.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from backend.stopwords import default_stopwords_text


# ─────────────────────────────────────────
# 中文字体检测（词云需要）
# ─────────────────────────────────────────

_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/SimHei.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux / Streamlit Cloud
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def get_chinese_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


# ─────────────────────────────────────────
# IP 省份解析
# ─────────────────────────────────────────

def parse_ip_province(ip_location: str) -> str:
    if not ip_location:
        return "未知"
    s = str(ip_location).strip()
    # 去除 "IP属地：" 前缀
    if "：" in s:
        return s.split("：")[-1].strip()
    if ":" in s:
        return s.split(":")[-1].strip()
    return s or "未知"


def enrich_province_column(df: pd.DataFrame) -> pd.DataFrame:
    """在 df 上添加 ip_province 列"""
    df = df.copy()
    df["ip_province"] = df["ip_location"].apply(parse_ip_province)
    return df


# ─────────────────────────────────────────
# 词云生成
# ─────────────────────────────────────────

def generate_wordcloud_image(
    word_freq: dict[str, int],
    width: int = 900,
    height: int = 450,
) -> bytes | None:
    """
    生成词云并返回 PNG bytes。
    需要 wordcloud 和 matplotlib。
    若没有中文字体则返回 None。
    """
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    font_path = get_chinese_font()
    if not font_path:
        return None

    wc = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        background_color="white",
        max_words=120,
        colormap="Blues",
        prefer_horizontal=0.9,
        min_font_size=10,
        max_font_size=100,
        random_state=42,
    )
    wc.generate_from_frequencies(word_freq)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────
# AI 主题命名（DeepSeek）
# ─────────────────────────────────────────

def ai_name_topics(
    topic_words_df: pd.DataFrame,
    api_key: str,
) -> pd.DataFrame:
    """
    为每个主题的 top 词调用 DeepSeek 起名和描述。
    返回更新了 topic_name 和 topic_description 列的新 DataFrame。
    """
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    df = topic_words_df.copy()
    names = []
    descriptions = []

    for _, row in df.iterrows():
        keywords = row.get("keywords") or ""
        prompt = (
            f"The following topic keywords were extracted from short-video comments using the BTM topic model: \n"
            f"Keywords: {keywords}\n\n"
            "Based on the semantic relationships among these words, complete two tasks:\n"
            "1. Give this topic a concise English name (2-6 words).\n"
            "2. Describe the core meaning of this topic in one concise English sentence.\n\n"
            "Return strictly in the following JSON format and include nothing else:\n"
            '{"name": "Topic name", "description": "One-sentence description"}'
        )

        name, desc = f"Topic {int(row['topic_id']) + 1}", ""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional text analysis assistant skilled at summarizing topics."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=100,
                )
                raw = resp.choices[0].message.content or ""
                # 去掉 code fence
                raw = raw.strip()
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
                data = json.loads(raw)
                name = str(data.get("name") or name).strip()
                desc = str(data.get("description") or "").strip()
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1.5)

        names.append(name)
        descriptions.append(desc)
        time.sleep(0.5)

    df["topic_name"] = names
    df["topic_description"] = descriptions
    return df


# ─────────────────────────────────────────
# 侧边栏配置渲染（每个页面调用）
# ─────────────────────────────────────────

def render_sidebar_config():
    """在侧边栏渲染 Cookie 和 API Key 配置框，同步到 session_state"""
    st.session_state.setdefault("bili_cookie", "")
    st.session_state.setdefault("deepseek_key", "")
    st.session_state.setdefault("openai_asr_key", "")
    st.session_state.setdefault("data_file", None)
    st.session_state.setdefault("sentiment_file", None)
    st.session_state.setdefault("topic_result", None)
    st.session_state.setdefault("subtitle_summary_done", False)
    st.session_state.setdefault("current_bvid", "")
    st.session_state.setdefault("video_title", "")
    st.session_state.setdefault("video_meta", {})
    st.session_state.setdefault("sentiment_prompt_template", "")
    st.session_state.setdefault("sentiment_custom_prompt", "")
    st.session_state.setdefault("custom_stopwords_text", default_stopwords_text())

    with st.sidebar:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:first-child a span {
                display: none;
            }
            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:first-child a::after {
                content: "Home";
                color: inherit;
                font-size: 1rem;
                font-weight: inherit;
                line-height: inherit;
            }
            .sidebar-brand {
                margin: 0.25rem 0 1rem;
            }
            .sidebar-brand__title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #111827;
                line-height: 1.35;
            }
            .sidebar-brand__subtitle {
                margin-top: 0.2rem;
                color: #6b7280;
                font-size: 0.82rem;
                line-height: 1.35;
            }
            .sidebar-section-title {
                margin: 1.1rem 0 0.45rem;
                color: #6b7280;
                font-size: 0.82rem;
                font-weight: 600;
                letter-spacing: 0;
            }
            .sidebar-task {
                padding: 0.65rem 0.75rem;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                background: #ffffff;
                margin-bottom: 0.65rem;
            }
            .sidebar-task__label {
                color: #6b7280;
                font-size: 0.78rem;
                margin-bottom: 0.25rem;
            }
            .sidebar-task__title {
                color: #111827;
                font-size: 0.9rem;
                line-height: 1.45;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            .flow-step {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.65rem;
                padding: 0.48rem 0;
                border-bottom: 1px solid #eef2f7;
            }
            .flow-step:last-child {
                border-bottom: 0;
            }
            .flow-step__name {
                color: #111827;
                font-size: 0.9rem;
                line-height: 1.25;
            }
            .flow-badge {
                flex: 0 0 auto;
                border-radius: 999px;
                padding: 0.16rem 0.5rem;
                font-size: 0.74rem;
                font-weight: 600;
                line-height: 1.25;
                border: 1px solid transparent;
            }
            .flow-badge--done {
                color: #047857;
                background: #ecfdf5;
                border-color: #a7f3d0;
            }
            .flow-badge--ready {
                color: #1d4ed8;
                background: #eff6ff;
                border-color: #bfdbfe;
            }
            .flow-badge--pending {
                color: #6b7280;
                background: #f9fafb;
                border-color: #e5e7eb;
            }
            </style>
            <div class="sidebar-brand">
                <div class="sidebar-brand__title">Short-Video Comment AI Analysis Platform</div>
                <div class="sidebar-brand__subtitle">Bilibili Comment Analysis Demo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("⚙️ System Settings", expanded=False):
            cookie_val = st.text_area(
                "Bilibili Cookie",
                value=st.session_state["bili_cookie"],
                height=80,
                placeholder="Paste your Bilibili Cookie (copy it from your browser after logging in)",
            )
            st.session_state["bili_cookie"] = cookie_val

            key_val = st.text_input(
                "DeepSeek API Key",
                value=st.session_state["deepseek_key"],
                type="password",
                placeholder="sk-...",
            )
            st.session_state["deepseek_key"] = key_val

            asr_key_val = st.text_input(
                "OpenAI ASR API Key",
                value=st.session_state["openai_asr_key"],
                type="password",
                placeholder="sk-...",
                help="Used for cloud speech-to-text when subtitles are unstable. This does not replace the DeepSeek summary key.",
            )
            st.session_state["openai_asr_key"] = asr_key_val

            st.markdown("**Stopword Settings**")
            st.caption(
                "Stopwords are common low-value terms filtered during tokenization, word-frequency statistics, "
                "and topic modeling. Customizing them affects Top 10 keywords in sentiment charts and topic "
                "analysis results. Adding filler words, meme terms, or platform-specific noise for the current "
                "video usually improves the output."
            )
            stopwords_val = st.text_area(
                "Global Stopwords",
                value=st.session_state["custom_stopwords_text"],
                height=160,
                help="Supports newline, space, or comma separators. If left empty, restoring defaults is recommended; otherwise word-frequency and topic results may contain many filler words.",
            )
            st.session_state["custom_stopwords_text"] = stopwords_val

        raw_ok = st.session_state.get("data_file") is not None
        sent_ok = st.session_state.get("sentiment_file") is not None
        topic_ok = st.session_state.get("topic_result") is not None
        summary_ok = bool(st.session_state.get("subtitle_summary_done"))

        bvid = st.session_state.get("current_bvid") or ""
        vtitle = st.session_state.get("video_title") or ""
        current_task = f"《{vtitle}》" if vtitle else (bvid or "No video yet")
        has_video = bool(bvid or vtitle)

        def flow_badge(text: str, kind: str) -> str:
            return f'<span class="flow-badge flow-badge--{kind}">{html.escape(text)}</span>'

        flow_steps = [
            ("Data Crawling", "Done", "done") if raw_ok else ("Data Crawling", "Not Started", "pending"),
            ("Data View", "Ready", "ready") if raw_ok else ("Data View", "Needs Data", "pending"),
            ("Sentiment Analysis", "Done", "done") if sent_ok else (
                ("Sentiment Analysis", "Ready", "ready") if raw_ok else ("Sentiment Analysis", "Not Started", "pending")
            ),
            ("Topic Analysis", "Done", "done") if topic_ok else (
                ("Topic Analysis", "Ready", "ready") if raw_ok else ("Topic Analysis", "Not Started", "pending")
            ),
            ("Video Summary", "Done", "done") if summary_ok else (
                ("Video Summary", "Ready", "ready") if has_video else ("Video Summary", "Not Started", "pending")
            ),
        ]
        flow_html = "".join(
            f"""
            <div class="flow-step">
                <span class="flow-step__name">{html.escape(name)}</span>
                {flow_badge(status, kind)}
            </div>
            """
            for name, status, kind in flow_steps
        )

        st.markdown(
            f"""
            <div class="sidebar-section-title">Current Task</div>
            <div class="sidebar-task" title="{html.escape(current_task, quote=True)}">
                <div class="sidebar-task__label">Video</div>
                <div class="sidebar-task__title">{html.escape(current_task)}</div>
            </div>
            <div class="sidebar-section-title">Workflow Status</div>
            {flow_html}
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────
# CSV 保存 / 读取
# ─────────────────────────────────────────

def save_df(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_df(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")
