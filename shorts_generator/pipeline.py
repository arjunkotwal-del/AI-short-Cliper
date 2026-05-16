"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI + ffmpeg/opencv.
                              Self-hosted, OPENAI_API_KEY required for the LLM.
"""
import os
from typing import Dict, List, Optional

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, generate_social_copy, get_highlights
from .transcriber import transcribe


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    import re as _re
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_openai_llm
    from .local.transcriber import transcribe_local

    # Per-video output subfolder so clips from different videos never mix
    _vid_match = _re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", youtube_url)
    _vid_id = _vid_match.group(1) if _vid_match else "unknown"
    from .config import LOCAL_OUTPUT_DIR
    video_out_dir = os.path.join(LOCAL_OUTPUT_DIR, _vid_id)

    source_path = download_youtube_local(youtube_url, fmt=download_format)

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_openai_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    ranked = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)
    video_duration = float(transcript.get("duration", 0))

    MIN_DURATION = 30.0
    MAX_DURATION = 60.0
    TARGET = 45.0  # ideal clip length when padding

    def _pad_to_window(h: Dict) -> Dict:
        s, e = float(h["start_time"]), float(h["end_time"])
        dur = e - s
        if MIN_DURATION <= dur <= MAX_DURATION:
            return h
        if dur > MAX_DURATION:
            # trim: keep from start, cap at MAX
            e = s + MAX_DURATION
        else:
            # Asymmetric padding: 35% before the hook, 65% after the end
            # so reactions and payoffs that land after the LLM's end_time get included.
            pad = TARGET - dur
            s = max(0.0, s - pad * 0.35)
            e = min(video_duration, e + pad * 0.65)
            # If we hit the video boundary, redistribute the remaining pad to the other side
            actual = e - s
            if actual < TARGET - 0.5:
                if e >= video_duration:
                    s = max(0.0, e - TARGET)
                else:
                    e = min(video_duration, s + TARGET)
        return {**h, "start_time": round(s, 3), "end_time": round(e, 3)}

    padded = [_pad_to_window(h) for h in ranked]

    # Dedupe after padding: drop clips that overlap >50% with a higher-scored one.
    kept: List[Dict] = []
    for h in padded:
        hs, he = float(h["start_time"]), float(h["end_time"])
        hd = he - hs
        overlap = any(
            max(0.0, min(he, float(k["end_time"])) - max(hs, float(k["start_time"]))) > 0.5 * hd
            for k in kept
        )
        if not overlap:
            kept.append(h)

    top = kept[:num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates (top {num_clips} requested)", flush=True)

    words = transcript.get("words") or []
    shorts = crop_highlights_local(source_path, top, aspect_ratio=aspect_ratio, words=words or None, out_dir=video_out_dir)

    # Generate social copy (.txt) for each successful clip
    print("[pipeline/local] generating social captions...", flush=True)
    for s in shorts:
        if not s.get("clip_url"):
            continue
        try:
            copy = generate_social_copy(s, llm_fn=call_openai_llm)
            caption = copy.get("caption", "")
            hashtags = " ".join(copy.get("hashtags", []))
            txt_path = os.path.splitext(s["clip_url"])[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"{caption}\n\n{hashtags}\n")
            s["social_copy"] = copy
        except Exception as e:
            print(f"[pipeline/local] social copy failed for {s.get('title')}: {e}", flush=True)

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(source_url, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI) or "local" (yt-dlp + faster-whisper +
            OpenAI + ffmpeg).

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with clip_url / local path
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(youtube_url, num_clips, aspect_ratio, download_format, language)
    if mode == "api":
        return _run_api(youtube_url, num_clips, aspect_ratio, download_format, language)
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
