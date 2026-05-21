"""Voiceover generator — full narration TTS with per-word timing.

Generates TTS audio for each scene, then concatenates into one
continuous voiceover. Also provides word-level timestamps for
karaoke captions.
"""
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import require_openai_key

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_TIMEOUT = 120

TTS_MODEL = "tts-1"
TTS_VOICE = "onyx"  # deep, engaging


def _get_duration(path: str) -> float:
    """Get audio duration in seconds."""
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, timeout=_TIMEOUT,
    )
    return float(r.stdout.strip())


def generate_scene_audio(
    scenes: List[dict],
    out_dir: str,
    voice: str = TTS_VOICE,
) -> Tuple[List[dict], str]:
    """Generate TTS for each scene, concatenate, and return timing info.

    Returns:
        (scene_timings, full_audio_path)

        scene_timings: list of {scene_index, start_time, end_time, duration, narration}
        full_audio_path: path to concatenated audio file
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required: pip install openai") from e

    client = OpenAI(api_key=require_openai_key(), timeout=90, max_retries=2)
    os.makedirs(out_dir, exist_ok=True)

    scene_audios = []
    scene_timings = []
    current_time = 0.0

    for i, scene in enumerate(scenes):
        idx = i + 1
        mp3_path = os.path.join(out_dir, f"voice_{idx:02d}.mp3")

        print(f"[voice] generating TTS for scene {idx}/{len(scenes)}...", flush=True)

        try:
            response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=voice,
                input=scene["narration"],
                response_format="mp3",
            )
            response.stream_to_file(mp3_path)

            duration = _get_duration(mp3_path)
            scene_audios.append(mp3_path)
            scene_timings.append({
                "scene_index": i,
                "start_time": round(current_time, 3),
                "end_time": round(current_time + duration, 3),
                "duration": round(duration, 3),
                "narration": scene["narration"],
            })
            current_time += duration
            print(f"[voice] scene {idx}: {duration:.1f}s", flush=True)

        except Exception as e:
            print(f"[voice] scene {idx} TTS failed: {e}", flush=True)
            # Use silence as fallback
            duration = float(scene.get("duration_hint", 8))
            silence_path = os.path.join(out_dir, f"silence_{idx:02d}.mp3")
            subprocess.run(
                [_FFMPEG, "-y", "-loglevel", "error",
                 "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
                 "-t", str(duration), "-c:a", "libmp3lame", silence_path],
                check=True, timeout=30,
            )
            scene_audios.append(silence_path)
            scene_timings.append({
                "scene_index": i,
                "start_time": round(current_time, 3),
                "end_time": round(current_time + duration, 3),
                "duration": round(duration, 3),
                "narration": scene["narration"],
            })
            current_time += duration

    # Concatenate all scene audios into one file
    full_audio_path = os.path.join(out_dir, "voiceover_full.mp3")
    if len(scene_audios) == 1:
        shutil.copy2(scene_audios[0], full_audio_path)
    else:
        # ffmpeg concat demuxer
        concat_list = os.path.join(out_dir, "audio_concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in scene_audios:
                safe = p.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        subprocess.run(
            [_FFMPEG, "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0",
             "-i", concat_list,
             "-c", "copy", full_audio_path],
            check=True, timeout=_TIMEOUT,
        )
        os.remove(concat_list)

    total_dur = _get_duration(full_audio_path)
    print(f"[voice] full voiceover: {total_dur:.1f}s", flush=True)

    # Clean up individual scene audio files
    for p in scene_audios:
        try:
            os.remove(p)
        except OSError:
            pass

    return scene_timings, full_audio_path


def estimate_word_timestamps(
    scene_timings: List[dict],
) -> List[dict]:
    """Estimate per-word timestamps from scene timings.

    Since OpenAI TTS doesn't provide word-level timestamps,
    we estimate them by splitting narration into words and
    distributing time evenly across them within each scene.

    Returns list of {word, start, end} dicts.
    """
    words = []
    for timing in scene_timings:
        narration = timing["narration"]
        scene_words = narration.split()
        if not scene_words:
            continue

        scene_start = timing["start_time"]
        scene_dur = timing["duration"]
        time_per_word = scene_dur / len(scene_words)

        for j, word in enumerate(scene_words):
            w_start = scene_start + j * time_per_word
            w_end = w_start + time_per_word
            words.append({
                "word": word,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
            })

    return words
