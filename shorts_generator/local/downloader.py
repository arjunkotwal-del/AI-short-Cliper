"""Local YouTube download via yt-dlp.

Skips re-downloading if a cached source file already exists on disk.

Returns a local mp4 path so the rest of the local pipeline can read it
directly off disk.
"""
import os
import re
from typing import Optional

from ..config import LOCAL_OUTPUT_DIR

_YOUTUBE_URL_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com|youtu\.be)/",
    re.IGNORECASE,
)


def _validate_youtube_url(url: str) -> None:
    """Raise ValueError if the URL is not a youtube.com / youtu.be URL."""
    if not _YOUTUBE_URL_RE.match(url):
        raise ValueError(
            f"Invalid URL: {url!r}\n"
            "Only youtube.com and youtu.be URLs are supported."
        )


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required. Install it with:\n"
            "    pip install -r requirements.txt"
        ) from e
    return yt_dlp


def _format_for(fmt: str) -> str:
    """Map our '720' / '1080' shorthand to a yt-dlp format selector."""
    try:
        height = int(fmt)
    except ValueError:
        height = 720
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={height}][ext=mp4]/best"
    )


def _video_id_from_url(url: str) -> Optional[str]:
    """Extract YouTube video ID from a URL, or None if not parseable."""
    import re
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def download_youtube_local(video_url: str, fmt: str = "720", out_dir: Optional[str] = None) -> str:
    """Download to disk and return the local mp4 path. Skips download if cached."""
    _validate_youtube_url(video_url)
    yt_dlp = _import_ytdlp()
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Check cache first
    vid_id = _video_id_from_url(video_url)
    if vid_id:
        cached = os.path.join(out_dir, f"source_{vid_id}.mp4")
        if os.path.exists(cached) and os.path.getsize(cached) > 100_000:
            print(f"[download/local] cache hit: {cached}", flush=True)
            return cached

    print(f"[download/local] {video_url} @ {fmt}p -> {out_dir}/", flush=True)
    ydl_opts = {
        "format": _format_for(fmt),
        "outtmpl": os.path.join(out_dir, "source_%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        path = ydl.prepare_filename(info)
        # merge_output_format may rename the extension after merge
        if not os.path.exists(path):
            stem, _ = os.path.splitext(path)
            for ext in (".mp4", ".mkv", ".webm"):
                if os.path.exists(stem + ext):
                    path = stem + ext
                    break

    print(f"[download/local] ready: {path}", flush=True)
    return path
