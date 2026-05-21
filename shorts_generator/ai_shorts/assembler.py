"""Assembler — stitch scene images into video with zoompan + voiceover + captions.

Each scene image gets a Ken Burns effect (slow zoom/pan) for its duration,
then all scenes are concatenated, voiceover is overlaid, and karaoke
captions are burned on.
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


def _image_to_clip(
    image_path: str,
    duration: float,
    out_path: str,
    effect: str = "zoom_in",
) -> str:
    """Convert a static image into a video clip with Ken Burns effect.

    Effects:
        zoom_in:  slow zoom from 100% to 120%
        zoom_out: slow zoom from 120% to 100%
        pan_left: slow pan from right to left
        pan_right: slow pan from left to right
    """
    fps = 30
    total_frames = int(duration * fps)

    if effect == "zoom_in":
        # Zoom from 1.0x to 1.2x, centered
        zp = (f"zoompan=z='1+0.2*on/{total_frames}':"
              f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
              f"d={total_frames}:s={OUT_W}x{OUT_H}:fps={fps}")
    elif effect == "zoom_out":
        zp = (f"zoompan=z='1.2-0.2*on/{total_frames}':"
              f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
              f"d={total_frames}:s={OUT_W}x{OUT_H}:fps={fps}")
    elif effect == "pan_left":
        zp = (f"zoompan=z='1.15':"
              f"x='iw*0.15*(1-on/{total_frames})':y='ih/2-(ih/zoom/2)':"
              f"d={total_frames}:s={OUT_W}x{OUT_H}:fps={fps}")
    elif effect == "pan_right":
        zp = (f"zoompan=z='1.15':"
              f"x='iw*0.15*on/{total_frames}':y='ih/2-(ih/zoom/2)':"
              f"d={total_frames}:s={OUT_W}x{OUT_H}:fps={fps}")
    else:
        zp = (f"zoompan=z='1.1':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
              f"d={total_frames}:s={OUT_W}x{OUT_H}:fps={fps}")

    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-i", image_path,
        "-vf", zp,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
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
    image_paths: List[str],
    scene_timings: List[dict],
    audio_path: str,
    word_timestamps: List[dict],
    out_path: str,
    title: str = "",
) -> str:
    """Full assembly: images → zoompan clips → concat → voiceover → captions.

    Args:
        image_paths: one image per scene
        scene_timings: [{start_time, end_time, duration, narration}, ...]
        audio_path: full voiceover audio file
        word_timestamps: [{word, start, end}, ...] for karaoke captions
        out_path: final output video path
        title: video title (unused for now)
    """
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."

    # Alternate effects for visual variety
    effects = ["zoom_in", "pan_right", "zoom_out", "pan_left", "zoom_in", "pan_right"]

    scene_clips = []
    try:
        # 1. Convert each image to a zoompan clip
        for i, (img, timing) in enumerate(zip(image_paths, scene_timings)):
            clip_path = os.path.join(out_dir, f"_scene_clip_{i:02d}.mp4")
            effect = effects[i % len(effects)]
            duration = timing["duration"]

            print(f"[assemble] scene {i+1}: {duration:.1f}s with {effect} effect", flush=True)
            _image_to_clip(img, duration, clip_path, effect=effect)
            scene_clips.append(clip_path)

        # 2. Concatenate all scene clips
        concat_path = os.path.join(out_dir, "_concat.mp4")
        _concat_clips(scene_clips, concat_path)

        # 3. Overlay voiceover audio
        with_audio_path = os.path.join(out_dir, "_with_audio.mp4")
        _overlay_audio(concat_path, audio_path, with_audio_path)

        # 4. Burn karaoke captions
        captioned = False
        if word_timestamps:
            try:
                from ..local.clipper import _generate_ass, _burn_captions

                # Build word list with adjusted timestamps (start from 0)
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
        for p in scene_clips + [concat_path, with_audio_path]:
            if os.path.exists(p) and p != out_path:
                try:
                    os.remove(p)
                except OSError:
                    pass

    return out_path
