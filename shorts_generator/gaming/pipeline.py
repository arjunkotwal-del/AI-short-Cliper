"""Gaming/streamer shorts pipeline.

Pipeline:
  yt-dlp  ->  audio peak detection  ->  narrator hooks (GPT + TTS)
          ->  smart crop + captions  ->  audio ducking assembly

Usage:  python main.py URL --mode gaming --num-clips 5
"""
import os
import re
from typing import Dict, List, Optional

from .audio_peaks import detect_audio_peaks
from .narrator import create_narrator_hook
from .assembler import assemble_gaming_clip


def _slug(title: str, idx: int, max_len: int = 45) -> str:
    """Safe filename from clip title."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    slug = re.sub(r"[;:=\[\]{}()/\\]", "", slug)
    slug = slug[:max_len] or f"clip_{idx:02d}"
    return f"{idx:02d}_{slug}.mp4"


def generate_gaming_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    output_dir: Optional[str] = None,
    clip_duration: float = 22.0,
    min_gap: float = 60.0,
) -> Dict:
    """Run the gaming pipeline end-to-end.

    1. Download video
    2. Detect audio peaks -> clip boundaries
    3. Transcribe (for captions + narrator context)
    4. Generate narrator hooks (GPT script + TTS)
    5. Assemble clips (smart crop + captions + audio ducking)
    """
    from ..local.downloader import download_youtube_local
    from ..local.transcriber import transcribe_local
    from ..config import LOCAL_OUTPUT_DIR

    # Output subfolder
    _m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", youtube_url)
    vid_id = _m.group(1) if _m else "unknown"
    base_dir = output_dir or LOCAL_OUTPUT_DIR
    video_out_dir = os.path.join(base_dir, f"{vid_id}_gaming")
    os.makedirs(video_out_dir, exist_ok=True)

    # 1. Download
    print("[gaming] downloading video...", flush=True)
    source_path = download_youtube_local(youtube_url, fmt=download_format)

    # 2. Audio peak detection
    print(f"[gaming] detecting top {num_clips} audio peaks...", flush=True)
    clips = detect_audio_peaks(
        source_path,
        num_clips=num_clips,
        clip_duration=clip_duration,
        min_gap=min_gap,
    )
    if not clips:
        raise RuntimeError("No audio peaks detected — video may be too quiet or too short.")

    print(f"[gaming] found {len(clips)} peaks:", flush=True)
    for i, c in enumerate(clips, 1):
        print(f"  peak {i}: {c['start_time']:.1f}s-{c['end_time']:.1f}s "
              f"(peak @ {c['peak_time']:.1f}s, {c['peak_db']:.1f} dB)", flush=True)

    # 3. Transcribe for captions + narrator context
    print("[gaming] transcribing...", flush=True)
    transcript = transcribe_local(source_path, language=language)
    words = transcript.get("words") or []
    segments = transcript.get("segments") or []

    def _get_transcript_snippet(start: float, end: float) -> str:
        """Get transcript text for a time range."""
        return " ".join(
            seg.get("text", "").strip()
            for seg in segments
            if float(seg.get("start", 0)) >= start - 2
            and float(seg.get("end", 0)) <= end + 2
        )

    # 4. Generate narrator hooks
    print("[gaming] generating narrator hooks...", flush=True)
    hooks = []
    for i, clip in enumerate(clips):
        snippet = _get_transcript_snippet(clip["start_time"], clip["end_time"])
        hook = create_narrator_hook(snippet, i + 1, video_out_dir)
        hooks.append(hook)

    # 5. Assemble clips
    print("[gaming] assembling clips...", flush=True)
    results = []
    for i, (clip, hook) in enumerate(zip(clips, hooks)):
        idx = i + 1
        title = f"gaming_peak_{idx}"
        if hook and hook.get("text"):
            # Use hook text as filename basis
            title = hook["text"][:40]

        out_filename = _slug(title, idx)
        out_path = os.path.join(video_out_dir, out_filename)

        try:
            print(f"[gaming] rendering clip {idx}/{len(clips)}: "
                  f"{clip['start_time']:.1f}s-{clip['end_time']:.1f}s", flush=True)

            assemble_gaming_clip(
                source_path=source_path,
                clip=clip,
                clip_index=idx,
                out_path=out_path,
                hook=hook,
                aspect_ratio=aspect_ratio,
                words=words if words else None,
            )

            results.append({
                "clip_url": out_path,
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "peak_time": clip["peak_time"],
                "peak_db": clip["peak_db"],
                "title": hook["text"] if hook else f"Peak {idx}",
                "hook_text": hook["text"] if hook else None,
                "score": None,
            })
            print(f"[gaming] clip {idx} done: {out_path}", flush=True)

        except Exception as e:
            print(f"[gaming] clip {idx} FAILED: {e}", flush=True)
            results.append({
                "clip_url": None,
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "error": str(e),
                "title": f"Peak {idx}",
                "score": None,
            })

    # Clean up hook audio files
    for hook in hooks:
        if hook and hook.get("audio_path") and os.path.exists(hook["audio_path"]):
            try:
                os.remove(hook["audio_path"])
            except OSError:
                pass

    return {
        "source_video_url": source_path,
        "mode": "gaming",
        "transcript": transcript,
        "highlights": clips,
        "shorts": results,
    }
