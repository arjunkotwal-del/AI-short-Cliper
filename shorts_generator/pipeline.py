"""End-to-end pipeline orchestrator.

Downloads, transcribes, scores, and renders highlights in one call.

Pipeline:
  yt-dlp  ->  faster-whisper  ->  OpenAI LLM scoring  ->  ffmpeg crop + captions
"""
import os
import re
from typing import Dict, List, Optional

from .highlights import call_openai_llm, generate_social_copy, get_highlights
from .local.clipper import crop_highlights_local
from .local.downloader import download_youtube_local
from .local.transcriber import transcribe_local


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    min_score: int = 0,
) -> Dict:
    """Run the full local pipeline and return a structured result.

    Args:
        youtube_url:     Source YouTube URL.
        num_clips:       Max number of shorts to render.
        aspect_ratio:    e.g. "9:16" (default).
        download_format: Source resolution — "360" / "480" / "720" / "1080".
        language:        ISO-639-1 code to force Whisper language detection.
        min_score:       Drop clips below this virality score (0 = keep all).

    Returns:
        {
          "source_video_url": str,   # local path to the downloaded source
          "transcript": {...},
          "highlights": [...],       # all scored candidates
          "shorts": [...],           # rendered clips with clip_url / thumbnail
        }
    """
    # Per-video output subfolder
    _m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", youtube_url)
    vid_id = _m.group(1) if _m else "unknown"
    from .config import LOCAL_OUTPUT_DIR
    video_out_dir = os.path.join(LOCAL_OUTPUT_DIR, vid_id)

    # 1. Download
    source_path = download_youtube_local(youtube_url, fmt=download_format)

    # 2. Transcribe
    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError("Whisper produced no segments — video may have no detectable speech.")

    # 3. Score highlights
    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_openai_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    ranked = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)
    video_duration = float(transcript.get("duration", 0))

    # 4. Pad each clip to 35-60 s (asymmetric: 20% before hook, 80% after payoff)
    MIN_DUR, MAX_DUR, TARGET = 35.0, 60.0, 50.0

    def _pad(h: Dict) -> Dict:
        s, e = float(h["start_time"]), float(h["end_time"])
        dur = e - s
        if MIN_DUR <= dur <= MAX_DUR:
            return h
        if dur > MAX_DUR:
            e = s + MAX_DUR
        else:
            pad = TARGET - dur
            s = max(0.0, s - pad * 0.20)
            e = min(video_duration, e + pad * 0.80)
            if e - s < TARGET - 0.5:
                s = max(0.0, e - TARGET) if e >= video_duration else s
                e = min(video_duration, s + TARGET)
        return {**h, "start_time": round(s, 3), "end_time": round(e, 3)}

    padded = [_pad(h) for h in ranked]

    # 5. Dedupe: drop clips that overlap >50% with a higher-scored one
    kept: List[Dict] = []
    for h in padded:
        hs, he = float(h["start_time"]), float(h["end_time"])
        hd = he - hs
        if not any(
            max(0.0, min(he, float(k["end_time"])) - max(hs, float(k["start_time"]))) > 0.5 * hd
            for k in kept
        ):
            kept.append(h)

    # 6. Apply min-score filter and num-clips cap
    if min_score > 0:
        kept = [h for h in kept if int(h.get("score", 0)) >= min_score]
        print(f"[pipeline] {len(kept)} clips pass --min-score {min_score}", flush=True)
    top = kept[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    # 7. Render clips
    words = transcript.get("words") or []
    shorts = crop_highlights_local(
        source_path, top,
        aspect_ratio=aspect_ratio,
        words=words or None,
        out_dir=video_out_dir,
    )

    # 8. Generate social copy (.txt sidecar) for each successful clip
    print("[pipeline] generating social captions...", flush=True)
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
            print(f"[pipeline] social copy failed for {s.get('title')}: {e}", flush=True)

    return {
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }
