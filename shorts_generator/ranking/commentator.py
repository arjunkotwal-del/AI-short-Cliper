"""Generate punchy commentary + TTS audio per ranked clip.

Energy scales with rank: rank N (lowest) is casual/dismissive,
rank 1 (highest) is fully hyped.
"""
import os
import subprocess
import shutil
from typing import Dict, Optional

from ..config import OPENAI_MODEL, require_openai_key

_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_TIMEOUT = 120

TTS_MODEL = "tts-1"
TTS_VOICE = "onyx"  # deep, engaging male voice

# Energy descriptors by rank position (1 = most extreme)
_ENERGY = {
    1: "MAX hype — this is the peak moment, absolutely explosive energy",
    2: "very hyped — almost the best, can barely contain the excitement",
    3: "building hype — this is getting serious",
    4: "mild, setting the scene — pretty decent but not crazy",
    5: "casual, almost dismissive — this is the tamest one",
}


def generate_rank_commentary(rank: int, total: int, title: str) -> str:
    """Use GPT to write punchy 1-2 sentence narration for a rank clip.

    rank=1 is most extreme/best; rank=total is least.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required: pip install openai") from e

    client = OpenAI(api_key=require_openai_key(), timeout=60, max_retries=2)

    # How many spots from the bottom?
    position = total - rank + 1  # 1 = first revealed (tamest), total = last (best)
    energy_hint = _ENERGY.get(rank, f"building toward rank 1, moderate excitement")

    prompt = f"""You write punchy countdown narration for a viral ranking TikTok.

Title of ranking video: "{title}"
Current rank being revealed: #{rank} out of {total} (rank 1 is the most extreme/best)
Energy level: {energy_hint}

Rules:
- 1-2 sentences ONLY — under 25 words total
- Start with the rank announcement ("Coming in at number {rank}...", "At number {rank}...", "Number {rank}...")
- Match the energy level above — rank {total} is chill, rank 1 is explosive
- No hashtags, no emojis, no filler phrases like "let's go"
- Spoken English only — this will be read aloud as a voiceover

Write the narration (just the text, no quotes):"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.85,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip().strip('"').strip("'")


def synthesize_tts(text: str, out_path: str) -> str:
    """Convert text to speech using OpenAI TTS. Returns path to mp3."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required: pip install openai") from e

    client = OpenAI(api_key=require_openai_key(), timeout=60, max_retries=2)
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="mp3",
    )
    response.stream_to_file(out_path)
    return out_path


def get_audio_duration(path: str) -> float:
    """Return audio/video duration in seconds via ffprobe."""
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, timeout=_TIMEOUT,
    )
    return float(r.stdout.strip())


def generate_clip_names(title: str, total: int) -> Dict[int, str]:
    """Use GPT to generate a 2-3 word label for each rank (rank 1 = most extreme).

    Returns {rank: "short label"} dict.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {}

    client = OpenAI(api_key=require_openai_key(), timeout=60, max_retries=2)
    prompt = f"""You label clips in a ranking video titled "{title}".

Write a SHORT label (2-4 words max, ALL CAPS) for each rank from 1 to {total}.
Rank 1 = most extreme/best moment. Rank {total} = least extreme.
Labels must be punchy and specific, like: "TOTAL DISASTER", "CLOSE CALL", "ALMOST PERFECT"

Respond ONLY with a JSON object: {{"1": "LABEL", "2": "LABEL", ..., "{total}": "LABEL"}}"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.8, max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        import json, re
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return {int(k): str(v) for k, v in data.items()}
    except Exception as e:
        print(f"[ranking/names] label generation failed: {e}", flush=True)
        return {}


def create_rank_commentary(rank: int, total: int, title: str, out_dir: str) -> Optional[dict]:
    """Generate commentary script + TTS audio for a ranked clip.

    Returns {"text": str, "audio_path": str, "duration": float} or None on failure.
    """
    try:
        text = generate_rank_commentary(rank, total, title)
        if not text:
            return None

        mp3_path = os.path.join(out_dir, f"commentary_rank{rank:02d}.mp3")
        synthesize_tts(text, mp3_path)
        duration = get_audio_duration(mp3_path)

        print(f"[ranking/commentary] rank #{rank}: \"{text}\" ({duration:.1f}s)", flush=True)
        return {"text": text, "audio_path": mp3_path, "duration": duration}

    except Exception as e:
        print(f"[ranking/commentary] failed for rank #{rank}: {e}", flush=True)
        return None
