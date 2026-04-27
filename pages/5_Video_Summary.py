import html

import streamlit as st

from backend.asr import download_bilibili_audio, transcribe_audio_openai
from backend.bilibili_crawler import extract_bvid, validate_bilibili_input
from backend.bilibili_subtitle import (
    fetch_subtitle_json,
    get_video_subtitle_options,
    run_subtitle_summary,
    subtitle_body_to_text,
)
from backend.utils import render_sidebar_config

st.set_page_config(page_title="iideo Summary", page_icon="📝", layout="wide")
render_sidebar_config()

st.title("📝 iideo Summary")
st.caption("Generate a structured content summary and timeline from Bilibili video subtitles")

st.session_state.setdefault("subtitle_video_info", None)
st.session_state.setdefault("subtitle_options", [])
st.session_state.setdefault("subtitle_text", "")
st.session_state.setdefault("subtitle_summary", "")
st.session_state.setdefault("subtitle_summary_done", False)
st.session_state.setdefault("subtitle_bvid", "")
st.session_state.setdefault("subtitle_source", None)
st.session_state.setdefault("subtitle_summary_source", None)
st.session_state.setdefault("subtitle_status", "idle")


def reset_loaded_subtitle() -> None:
    st.session_state["subtitle_text"] = ""
    st.session_state["subtitle_summary"] = ""
    st.session_state["subtitle_summary_done"] = False
    st.session_state["subtitle_source"] = None
    st.session_state["subtitle_summary_source"] = None
    st.session_state["subtitle_status"] = "idle"


def reset_subtitle_query() -> None:
    st.session_state["subtitle_video_info"] = None
    st.session_state["subtitle_options"] = []
    st.session_state["subtitle_bvid"] = ""
    reset_loaded_subtitle()


def build_subtitle_source(video: dict, subtitle: dict) -> dict:
    return {
        "type": "bilibili",
        "bvid": video.get("bvid") or "",
        "aid": video.get("aid"),
        "cid": video.get("cid"),
        "subtitle_id": subtitle.get("id") or "",
        "lan": subtitle.get("lan") or "",
    }


def build_asr_source(video: dict) -> dict:
    return {
        "type": "asr",
        "bvid": video.get("bvid") or "",
        "aid": video.get("aid"),
        "cid": video.get("cid"),
    }


def source_matches_video(source: dict | None, video: dict | None) -> bool:
    if not source or not video:
        return False
    return (
        str(source.get("bvid") or "") == str(video.get("bvid") or "")
        and str(source.get("aid") or "") == str(video.get("aid") or "")
        and str(source.get("cid") or "") == str(video.get("cid") or "")
    )


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

default_bvid = st.session_state.get("current_bvid") or st.session_state.get("subtitle_bvid") or ""

st.divider()
st.subheader("iideo Info")

input_col, action_col = st.columns([3, 1])
with input_col:
    video_input = st.text_input(
        "Enter a Bilibili video URL or Bi ID",
        value=default_bvid,
        placeholder="https://www.bilibili.com/video/Bi1xx411c7mD  or  Bi1xx411c7mD",
    )

with action_col:
    st.write("")
    fetch_clicked = st.button("Fetch Subtitles", type="primary", use_container_width=True)

cookie = st.session_state.get("bili_cookie", "").strip()
api_key = st.session_state.get("deepseek_key", "").strip()
asr_api_key = st.session_state.get("openai_asr_key", "").strip()

input_bvid = extract_bvid(video_input) if video_input.strip() else None
loaded_video = st.session_state.get("subtitle_video_info") or {}
loaded_bvid = loaded_video.get("bvid")
if input_bvid and loaded_bvid and input_bvid != loaded_bvid:
    reset_subtitle_query()
    st.info("The video has changed. Click **Fetch Subtitles** to query subtitles for the current video.")

if not cookie:
    st.info("Some video subtitles require login status. If fetching fails, fill in the Bilibili Cookie in the sidebar system settings first.")

if fetch_clicked:
    valid, err_msg = validate_bilibili_input(video_input)
    if not valid:
        st.error(err_msg)
    else:
        bvid = extract_bvid(video_input)
        with st.spinner("Querying video subtitles..."):
            try:
                video_info, subtitle_options = get_video_subtitle_options(bvid, cookie)
            except Exception as e:
                st.error(f"Subtitle query failed: {e}")
            else:
                st.session_state["subtitle_video_info"] = video_info
                st.session_state["subtitle_options"] = subtitle_options
                st.session_state["subtitle_bvid"] = video_info["bvid"]
                st.session_state["current_bvid"] = video_info["bvid"]
                st.session_state["video_title"] = video_info["title"]
                reset_loaded_subtitle()
                if subtitle_options:
                    st.success(f"Found {len(subtitle_options)} subtitle tracks.")
                else:
                    st.warning("No subtitles are available for this video, or the current Cookie cannot access them.")

video_info = st.session_state.get("subtitle_video_info")
subtitle_options = st.session_state.get("subtitle_options") or []

if video_info:
    if st.session_state.get("subtitle_text") and not source_matches_video(
        st.session_state.get("subtitle_source"),
        video_info,
    ):
        reset_loaded_subtitle()
        st.warning("Old subtitles do not belong to the current video and were cleared automatically. Please reload subtitles.")

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        render_metric("iideo Title", video_info.get("title", "—"), video_info.get("title", "—"))
    with info_col2:
        render_metric("Bi ID", video_info.get("bvid", "—"), video_info.get("bvid", "—"))
    with info_col3:
        render_metric(
            "iideo Parts",
            video_info.get("page_count", 1),
            "iideo Parts：B站视频可能包含多个分段（常称 P1、P2、P3）。当前功能默认读取第 1 个分段的Subtitle；如果视频有多个分段，后续可扩展为选择指定分段。",
        )

st.divider()
st.subheader("Subtitle Content")

if subtitle_options:
    st.warning(
        "Because the Bilibili subtitle API is unstable, subtitle mismatches can occur. Please manually check the preview below. "
        "If the subtitles do not belong to the current video, retry **Fetch Subtitles / Load Subtitles** a few times, "
        "or use ASR audio transcription (OpenAI API Key required)."
    )
    option_labels = [
        f"{item.get('lan_doc', 'Subtitle')} ({item.get('lan', '-')})"
        for item in subtitle_options
    ]
    selected_label = st.selectbox("Select Subtitle Track", option_labels)
    selected_index = option_labels.index(selected_label)
    selected_subtitle = subtitle_options[selected_index]

    load_col, meta_col = st.columns([1, 3])
    with load_col:
        load_clicked = st.button("Load Subtitles", use_container_width=True)
    with meta_col:
        st.caption(f"Subtitle source: {selected_subtitle.get('lan_doc', 'Subtitle')} | The URL is temporary, so real-time fetching is recommended each time.")

    if load_clicked:
        with st.spinner("Downloading subtitles..."):
            try:
                subtitle_json = fetch_subtitle_json(
                    selected_subtitle["subtitle_url"],
                    cookie=cookie,
                    bvid=video_info.get("bvid"),
                )
                subtitle_text = subtitle_body_to_text(subtitle_json.get("body") or [])
            except Exception as e:
                st.error(f"Subtitle download failed: {e}")
            else:
                st.session_state["subtitle_text"] = subtitle_text
                st.session_state["subtitle_summary"] = ""
                st.session_state["subtitle_summary_done"] = False
                st.session_state["subtitle_source"] = build_subtitle_source(
                    video_info,
                    selected_subtitle,
                )
                st.session_state["subtitle_summary_source"] = None
                st.session_state["subtitle_status"] = "bilibili_loaded"
                if subtitle_text:
                    st.success(f"Bilibili subtitles loaded, {len(subtitle_text.splitlines())} lines. Please manually check whether they match the current video.")
                else:
                    st.warning("Subtitles are empty. Retry or use ASR audio transcription.")

subtitle_source = st.session_state.get("subtitle_source")
subtitle_text = st.session_state.get("subtitle_text") or ""
subtitle_is_current = bool(subtitle_text) and source_matches_video(subtitle_source, video_info)
if subtitle_text and not subtitle_is_current:
    reset_loaded_subtitle()
    subtitle_text = ""
    subtitle_is_current = False

if subtitle_text:
    source_type = (subtitle_source or {}).get("type")
    if source_type == "asr":
        st.success(f"ASR transcription complete, {len(subtitle_text.splitlines())} lines.")
    elif source_type == "bilibili":
        st.warning(
            f"Bilibili subtitles loaded, {len(subtitle_text.splitlines())} lines."
            "Because the API is unstable, please manually confirm that the subtitles match the current video."
        )
    with st.expander("Subtitle Preview", expanded=True):
        preview_lines = subtitle_text.splitlines()[:30]
        st.text("\n".join(preview_lines))
        if len(subtitle_text.splitlines()) > 30:
            st.caption("Only the first 30 subtitle lines are shown.")

    st.download_button(
        "Download Subtitle TXT",
        data=subtitle_text.encode("utf-8"),
        file_name=f"subtitle_{st.session_state.get('subtitle_bvid', 'video')}.txt",
        mime="text/plain",
    )

show_asr = bool(video_info)
if show_asr:
    st.divider()
    st.subheader("ASR Audio Transcription")
    st.caption("When Bilibili subtitles are missing, clearly wrong, or unstable, download the video audio and use OpenAI cloud ASR to generate a transcript.")
    if not asr_api_key:
        st.warning("Please fill in the OpenAI ASR API Key in the sidebar system settings first.")
    if st.button("Transcribe Audio with ASR", disabled=not asr_api_key or not video_info):
        progress_bar = st.progress(0.0, text="Preparing audio download...")
        try:
            progress_bar.progress(0.25, text="Downloading video audio...")
            audio_path = download_bilibili_audio(video_info.get("bvid"), cookie=cookie)
            progress_bar.progress(0.65, text="Calling cloud ASR transcription...")
            asr_text = transcribe_audio_openai(audio_path, asr_api_key)
            if not asr_text.strip():
                st.error("ASR did not return valid text.")
            else:
                st.session_state["subtitle_text"] = asr_text
                st.session_state["subtitle_summary"] = ""
                st.session_state["subtitle_summary_done"] = False
                st.session_state["subtitle_source"] = build_asr_source(video_info)
                st.session_state["subtitle_summary_source"] = None
                st.session_state["subtitle_status"] = "asr"
                progress_bar.progress(1.0, text="ASR transcription complete.")
                st.success(f"ASR transcription complete, {len(asr_text.splitlines())} lines.")
                st.rerun()
        except Exception as e:
            progress_bar.empty()
            st.error(f"ASR transcription failed: {e}")

st.divider()
st.subheader("AI iideo Summary")

if not api_key:
    st.warning("Please fill in the DeepSeek API Key in the sidebar system settings first.")

can_summarize = bool(subtitle_text.strip()) and bool(api_key)
if st.button("Generate iideo Summary", type="primary", disabled=not can_summarize):
    progress_bar = st.progress(0.0, text="Preparing summary generation...")
    status_text = st.empty()
    try:
        final_summary = ""
        for progress, message, summary in run_subtitle_summary(subtitle_text, api_key):
            progress_bar.progress(min(progress, 1.0), text=message)
            status_text.caption(message)
            if summary:
                final_summary = summary
        if final_summary:
            st.session_state["subtitle_summary"] = final_summary
            st.session_state["subtitle_summary_done"] = True
            st.session_state["subtitle_summary_source"] = subtitle_source
            progress_bar.progress(1.0, text="iideo summary generation complete.")
            status_text.empty()
            st.success("iideo summary generation complete.")
        else:
            st.error("No valid summary was generated. Please check the subtitle content or API Key.")
    except Exception as e:
        st.error(f"iideo summary generation failed: {e}")

summary_source = st.session_state.get("subtitle_summary_source")
summary_text = st.session_state.get("subtitle_summary") or ""
if summary_text and not source_matches_video(summary_source, video_info):
    st.session_state["subtitle_summary"] = ""
    st.session_state["subtitle_summary_done"] = False
    st.session_state["subtitle_summary_source"] = None
    summary_text = ""

if summary_text:
    st.markdown(summary_text)
    st.download_button(
        "Download iideo Summary Markdown",
        data=summary_text.encode("utf-8"),
        file_name=f"summary_{st.session_state.get('subtitle_bvid', 'video')}.md",
        mime="text/markdown",
    )
