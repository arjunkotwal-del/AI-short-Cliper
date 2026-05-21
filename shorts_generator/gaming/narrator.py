"""Module 2: AI narrator hooks — GPT-4o-mini script + OpenAI TTS.

Generates a short (1-2 sentence) hook narration for each clip, then
synthesizes it via OpenAI TTS. The narration plays at the start of
the clip while the original audio is ducked.
"""
import os
import subprocess
import shutil
import tempfile
from typing import Optional

from ..config import OPENAI_MODEL, require_openai_key

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_TIMEOUT = 120

# TTS config
TTS_MODEL = "tts-1"
TTS_VOICE = "onyx"  # deep, engaging male voice


def generate_hook_script(
    transcript_snippet: str,
    clip_index: int,
) -> str:
    """Generate a 1-2 sentence narrator hook for a gaming clip.

    Uses the transcript snippet (if available) or a generic gaming hook.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required: pip install openai") from e

    client = OpenAI(api_key=require_openai_key(), timeout=60, max_retries=2)

    prompt = f"""You write short narrator hooks for gaming/streamer TikTok clips.

Rules:
- 1-2 sentences MAX (under 15 words total)
- Hype/dramatic energy, like a sports commentator
- Present tense, direct address ("watch this", "he just...")
- No hashtags, no emojis, no quotes
- Must work as a spoken voice-over

Transcript context from the clip:
{transcript_snippet[:500] if transcript_snippet else "(no speech detected — pure gameplay moment)"}

Write the narrator hook:"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.9,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip().strip('"')


def synthesize_tts(text: str, out_path: str, voice: str = TTS_VOICE) -> str:
    """Convert text to speech using OpenAI TTS API. Returns path to mp3 file."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required: pip install openai") from e

    client = OpenAI(api_key=require_openai_key(), timeout=60, max_retries=2)

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        response_format="mp3",
    )
    response.stream_to_file(out_path)
    return out_path


def get_audio_duration(path: str) -> float:
    """Get audio file duration in seconds."""
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, timeout=_TIMEOUT,
    )
    return float(r.stdout.strip())


def create_narrator_hook(
    transcript_snippet: str,
    clip_index: int,
    out_dir: str,
) -> Optional[dict]:
    """Generate hook script + TTS audio. Returns {text, audio_path, duration} or None."""
    try:
        hook_text = generate_hook_script(transcript_snippet, clip_index)
        if not hook_text:
            return None

        mp3_path = os.path.join(out_dir, f"hook_{clip_index:02d}.mp3")
        synthesize_tts(hook_text, mp3_path)
        duration = get_audio_duration(mp3_path)

        print(f"[narrator] clip {clip_index}: \"{hook_text}\" ({duration:.1f}s)", flush=True)

        return {
            "text": hook_text,
            "audio_path": mp3_path,
            "duration": duration,
        }
    except Exception as e:
        print(f"[narrator] hook generation failed for clip {clip_index}: {e}", flush=True)
        return None
