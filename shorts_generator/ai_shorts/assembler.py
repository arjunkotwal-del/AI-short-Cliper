"""Assembler — concat video clips + voiceover + captions.

Takes pre-generated video clips (from Runway or zoompan fallback),
concatenates them, overlays the TTS voiceover, and burns karaoke captions.
"""
import os
import shutil
import subprocess
from typing import Dict, List, Optional

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_TIMEOUT = 300

# Output dimensions (9:16 vertical)
OUT_W = 720
OUT_H = 1280


def _get_duration(path: str) -> float:
    """Get media duration in seconds."""
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return float(r.stdout.strip())


def _speed_adjust_clip(in_path: str, target_duration: float, out_path: str) -> str:
    """Speed up or slow down a video clip to match target duration.

    Runway clips are fixed 5s or 10s — we need to stretch/compress
    them to match the TTS narration duration for each scene.
    """
    actual_dur = _get_duration(in_path)
    if abs(actual_dur - target_duration) < 0.3:
        # Close enough — just copy
        shutil.copy2(in_path, out_path)
        return out_path

    speed_factor = actual_dur / target_duration
    # Video speed: setpts divides by factor (>1 = faster, <1 = slower)
    # Audio: we strip audio since we overlay voiceover anyway
    vf = f"setpts=PTS/{speed_factor:.4f},scale={OUT_W}:{OUT_H}:flags=lanczos"

    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", vf,
        "-an",  # strip audio — we add voiceover later
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-t", f"{target_duration:.3f}",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


def _concat_clips(clip_paths: List[str], out_path: str) -> str:
    """Concatenate video clips using ffmpeg concat demuxer."""
    concat_txt = out_path + ".concat.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in clip_paths:
            safe = p.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c", "copy", out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    os.remove(concat_txt)
    return out_path


def _overlay_audio(video_path: str, audio_path: str, out_path: str) -> str:
    """Replace video's audio with the voiceover."""
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


def assemble_ai_short(
    video_paths: List[str],
    scene_timings: List[dict],
    audio_path: str,
    word_timestamps: List[dict],
    out_path: str,
    title: str = "",
) -> str:
    """Full assembly: speed-adjust clips → concat → voiceover → captions.

    Args:
        video_paths: pre-generated video clips (from Runway or fallback)
        scene_timings: [{start_time, end_time, duration, narration}, ...]
        audio_path: full voiceover audio file
        word_timestamps: [{word, start, end}, ...] for karaoke captions
        out_path: final output video path
        title: video title (unused for now)
    """
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."

    adjusted_clips = []
    concat_path = os.path.join(out_dir, "_concat.mp4")
    with_audio_path = os.path.join(out_dir, "_with_audio.mp4")

    try:
        # 1. Speed-adjust each clip to match TTS duration
        for i, (vid, timing) in enumerate(zip(video_paths, scene_timings)):
            adjusted_path = os.path.join(out_dir, f"_adjusted_{i:02d}.mp4")
            target_dur = timing["duration"]
            print(f"[assemble] scene {i+1}: adjusting to {target_dur:.1f}s", flush=True)
            _speed_adjust_clip(vid, target_dur, adjusted_path)
            adjusted_clips.append(adjusted_path)

        # 2. Concatenate all clips
        print("[assemble] concatenating clips...", flush=True)
        _concat_clips(adjusted_clips, concat_path)

        # 3. Overlay voiceover audio
        print("[assemble] overlaying voiceover...", flush=True)
        _overlay_audio(concat_path, audio_path, with_audio_path)

        # 4. Burn karaoke captions
        captioned = False
        if word_timestamps:
            try:
                from ..local.clipper import _generate_ass, _burn_captions

                ass_words = [
                    {"word": w["word"], "start": w["start"], "end": w["end"]}
                    for w in word_timestamps
                ]

                total_dur = scene_timings[-1]["end_time"] if scene_timings else 60
                ass_content = _generate_ass(
                    ass_words, 0, total_dur, OUT_W, OUT_H,
                    hook_sentence=None,
                )
                if ass_content:
                    ass_filename = os.path.basename(out_path) + ".ass"
                    ass_path = os.path.join(out_dir, ass_filename)
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(ass_content)
                    _burn_captions(with_audio_path, out_path, ass_filename, out_dir)
                    captioned = True
                    try:
                        os.remove(ass_path)
                    except OSError:
                        pass
            except Exception as e:
                print(f"[assemble] captions failed: {e}", flush=True)

        if not captioned:
            shutil.copy2(with_audio_path, out_path)

        print(f"[assemble] done: {out_path}", flush=True)

    finally:
        # Cleanup temp files
        for p in adjusted_clips + [concat_path, with_audio_path]:
            if p and os.path.exists(p) and p != out_path:
                try:
                    os.remove(p)
                except OSError:
                    pass

    return out_path
