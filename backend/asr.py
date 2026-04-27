"""
Cloud-friendly audio extraction and ASR transcription helpers.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from openai import OpenAI

from backend.config import BASE_DIR, OPENAI_TRANSCRIBE_MODEL

AUDIO_DIR = BASE_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_AUDIO_EXTS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
OPENAI_AUDIO_LIMIT_BYTES = 25 * 1024 * 1024


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return safe[:80] or "video"


def _format_time(seconds: float | int | None) -> str:
    if seconds is None:
        return "00:00"
    total_seconds = max(0, int(float(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{sec:02}"
    return f"{minutes:02}:{sec:02}"


def download_bilibili_audio(
    bvid: str,
    cookie: str = "",
) -> Path:
    """
    Download the best available audio stream with yt-dlp.

    This avoids local speech models on Streamlit Cloud; the file is uploaded to
    a cloud ASR service afterwards.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("Missing yt-dlp dependency. Please install yt-dlp in requirements.txt first.") from exc

    safe_bvid = _safe_name(bvid)
    video_url = f"https://www.bilibili.com/video/{bvid}"
    outtmpl = str(AUDIO_DIR / f"{safe_bvid}.%(ext)s")

    for existing in AUDIO_DIR.glob(f"{safe_bvid}.*"):
        if existing.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            try:
                existing.unlink()
            except OSError:
                pass

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": video_url,
        },
    }
    if cookie:
        ydl_opts["http_headers"]["Cookie"] = cookie

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)
    except Exception as exc:
        raise RuntimeError(f"Audio download failed: {exc}") from exc

    candidates = [
        path
        for path in AUDIO_DIR.glob(f"{safe_bvid}.*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]
    if not candidates:
        raise RuntimeError("Audio download completed, but no uploadable audio file was found.")

    audio_path = max(candidates, key=lambda path: path.stat().st_mtime)
    size = audio_path.stat().st_size
    if size > OPENAI_AUDIO_LIMIT_BYTES:
        mb = size / 1024 / 1024
        raise RuntimeError(f"Audio file is too large ({mb:.1f} MB), exceeding the OpenAI transcription API 25 MB upload limit.")

    return audio_path


def _segment_to_text(segment) -> str:
    start = getattr(segment, "start", None)
    text = str(getattr(segment, "text", "") or "").strip()
    if not text and isinstance(segment, dict):
        start = segment.get("start")
        text = str(segment.get("text") or "").strip()
    if not text:
        return ""
    return f"[{_format_time(start)}] {text}"


def transcribe_audio_openai(
    audio_path: str | os.PathLike[str],
    api_key: str,
    model: str = OPENAI_TRANSCRIBE_MODEL,
) -> str:
    """
    Transcribe audio with OpenAI's cloud ASR endpoint.
    """
    if not api_key.strip():
        raise RuntimeError("Please configure the OpenAI ASR API Key first.")

    path = Path(audio_path)
    if not path.exists():
        raise RuntimeError("The audio file does not exist and cannot be transcribed.")

    client = OpenAI(api_key=api_key)
    with path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="zh",
        )

    segments = getattr(result, "segments", None)
    if segments:
        lines = [_segment_to_text(segment) for segment in segments]
        return "\n".join(line for line in lines if line).strip()

    text = str(getattr(result, "text", "") or "").strip()
    if not text and isinstance(result, dict):
        text = str(result.get("text") or "").strip()
    return f"[00:00] {text}" if text else ""
