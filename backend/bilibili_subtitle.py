"""
Bilibili video subtitle retrieval and DeepSeek summary helpers.
"""
from __future__ import annotations

import time
from typing import Generator

import requests
from openai import OpenAI

from backend.config import BILIBILI_API_VIEW, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

BILIBILI_API_PLAYER = "https://api.bilibili.com/x/player/v2"
MAX_CHUNK_CHARS = 12000
MAX_CHUNKS = 8
REQUEST_SLEEP = 0.3


def _build_headers(cookie: str, bvid: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
    }
    if bvid:
        headers["Referer"] = f"https://www.bilibili.com/video/{bvid}"
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _normalize_subtitle_url(url: str) -> str:
    url = (url or "").strip()
    if url.startswith("//"):
        return "https:" + url
    return url


def _format_time(seconds: float | int | None) -> str:
    if seconds is None:
        return "00:00"
    total_seconds = max(0, int(float(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{sec:02}"
    return f"{minutes:02}:{sec:02}"


def get_video_subtitle_options(
    bvid: str,
    cookie: str = "",
) -> tuple[dict, list[dict]]:
    """
    Return video metadata and available subtitle options for the first page of a BV.
    MVP scope: first page only for multi-part videos.
    """
    session = requests.Session()
    headers = _build_headers(cookie, bvid)

    view_resp = session.get(
        BILIBILI_API_VIEW,
        params={"bvid": bvid},
        headers=headers,
        timeout=10,
    )
    view_resp.raise_for_status()
    view_json = view_resp.json()
    if view_json.get("code") != 0:
        raise RuntimeError(view_json.get("message") or "Failed to fetch video information")

    video = view_json["data"]
    aid = video["aid"]
    cid = video["cid"]
    pages = video.get("pages") or []
    if pages:
        cid = pages[0].get("cid") or cid

    player_resp = session.get(
        BILIBILI_API_PLAYER,
        params={"aid": aid, "cid": cid, "bvid": bvid},
        headers=headers,
        timeout=10,
    )
    player_resp.raise_for_status()
    player_json = player_resp.json()
    if player_json.get("code") != 0:
        raise RuntimeError(player_json.get("message") or "Failed to fetch player subtitle information")

    subtitles = (
        player_json.get("data", {})
        .get("subtitle", {})
        .get("subtitles", [])
    )

    options: list[dict] = []
    for item in subtitles:
        subtitle_url = _normalize_subtitle_url(item.get("subtitle_url") or "")
        if not subtitle_url:
            continue
        options.append(
            {
                "id": str(item.get("id") or ""),
                "aid": aid,
                "cid": cid,
                "page": 1,
                "lan": item.get("lan") or "",
                "lan_doc": item.get("lan_doc") or item.get("lan") or "Subtitle",
                "subtitle_url": subtitle_url,
                "is_ai": str(item.get("lan") or "").startswith("ai-"),
            }
        )

    def subtitle_rank(item: dict) -> tuple[int, int, str]:
        lan = str(item.get("lan") or "")
        lan_doc = str(item.get("lan_doc") or "")
        is_chinese = "zh" in lan or "Chinese" in lan_doc
        return (0 if is_chinese else 1, 0 if item.get("is_ai") else 1, lan_doc)

    options.sort(key=subtitle_rank)
    video_info = {
        "bvid": video.get("bvid") or bvid,
        "aid": aid,
        "cid": cid,
        "title": video.get("title") or bvid,
        "duration": int(video.get("duration") or 0),
        "page_count": len(pages) or 1,
    }
    return video_info, options


def fetch_subtitle_json(subtitle_url: str, cookie: str = "", bvid: str | None = None) -> dict:
    headers = _build_headers(cookie, bvid)
    resp = requests.get(subtitle_url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "body" not in data:
        raise RuntimeError("Invalid subtitle data format: missing body field")
    return data


def subtitle_body_to_text(body: list[dict]) -> str:
    lines = []
    for item in body:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        start = _format_time(item.get("from"))
        lines.append(f"[{start}] {content}")
    return "\n".join(lines)


def _split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))
    return chunks


def _call_deepseek(client: OpenAI, prompt: str, max_tokens: int = 1600) -> str:
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a rigorous video content analysis assistant. Summarize only based on subtitle content and do not invent information not present in the subtitles.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


SummaryYield = tuple[float, str, str | None]
"""(progress, message, summary_or_none)"""


def run_subtitle_summary(
    subtitle_text: str,
    api_key: str,
) -> Generator[SummaryYield, None, None]:
    """
    Generate a structured video summary from subtitle text.
    Long subtitles are summarized by chunks first, then merged.
    """
    text = (subtitle_text or "").strip()
    if not text:
        yield 1.0, "Subtitles are empty, so a summary cannot be generated.", None
        return

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    all_chunks = _split_text(text)
    if not all_chunks:
        yield 1.0, "Subtitles are empty, so a summary cannot be generated.", None
        return

    truncated = len(all_chunks) > MAX_CHUNKS
    chunks = all_chunks[:MAX_CHUNKS]

    if len(chunks) == 1:
        yield 0.2, "Generating video summary...", None
        prompt = _final_summary_prompt(chunks[0], truncated=truncated)
        summary = _call_deepseek(client, prompt, max_tokens=1800)
        yield 1.0, "Video summary generation complete.", summary
        return

    partial_summaries: list[str] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        yield idx / (total + 1), f"Summarizing subtitle chunk {idx}/{total}...", None
        prompt = _chunk_summary_prompt(chunk, idx, total)
        partial_summaries.append(_call_deepseek(client, prompt, max_tokens=900))
        time.sleep(REQUEST_SLEEP)

    yield total / (total + 1), "Merging chunk summaries...", None
    joined = "\n\n".join(f"[Chunk {i+1}]\n{summary}" for i, summary in enumerate(partial_summaries))
    summary = _call_deepseek(client, _final_summary_prompt(joined, truncated=truncated), max_tokens=1800)
    yield 1.0, "Video summary generation complete.", summary


def _chunk_summary_prompt(text: str, index: int, total: int) -> str:
    return f"""The following is subtitle chunk {index}/{total}. Generate a concise summary based only on this chunk.

Requirements:
1. Summarize the main content of this chunk.
2. Extract important timestamps appearing in this chunk, formatted as "mm:ss - content".
3. Do not invent information not present in the subtitles.

[Subtitle Chunk]
{text}
"""


def _final_summary_prompt(text: str, truncated: bool = False) -> str:
    truncation_note = "Note: the subtitles are long, and the system processed only the first several chunks. Mention the limited coverage in the summary." if truncated else ""
    return f"""Generate a structured English summary based on the following Bilibili video subtitles.

{truncation_note}

Use exactly the following structure:

## Core Video Content
Summarize the main content of the video in 3-5 sentences.

## Key Timeline
List important moments in chronological order, formatted as:
- mm:ss - content summary

## Main Points or Events
Extract points that repeatedly appear or are emphasized in the video.

## Comment Analysis Reference
Explain what comment focus points this video may trigger for later comment analysis.

Requirements:
- Summarize only based on subtitle content.
- Do not invent information not present in the subtitles.
- Use subtitle timestamps whenever possible.

[Subtitle Content]
{text}
"""
