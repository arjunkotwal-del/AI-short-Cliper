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


def generate_clip_commentary(
    rank: int,
    total: int,
    title: str,
    label: str,
    clip_duration: float,
) -> str:
    """GPT writes a human, funny, opinionated voiceover for a ranked clip.

    Rules enforced in the prompt:
    - Open with the rank + a creative, specific hook (NOT generic)
    - Explain WHY this clip is at THIS rank — personal opinion, not description
    - Personality: funny, irreverent, like a friend reacting on the couch
    - Target ~70% of clip runtime at ~130 wpm
    - Never describe what is visually obvious on screen
    """
    try:
        from openai import OpenAI
    except ImportError:
        return f"Coming in at number {rank}... {label.title()}."

    client = OpenAI(api_key=require_openai_key(), timeout=60, max_retries=2)

    # ── Short clip mode (≤ 7 s) ──────────────────────────────────────────────
    # For very short clips there is only time for one punchy sentence.
    # Target 80% of duration; 130 wpm; hard cap so GPT can't ramble.
    # ── Long clip mode  (> 7 s) ──────────────────────────────────────────────
    # Full analytical commentary targeting 65-70% coverage.
    is_short_clip = clip_duration <= 7.0
    if is_short_clip:
        target_words  = max(6, int(clip_duration * 0.80 * 130 / 60))
        max_tokens    = int(target_words / 0.75) + 8   # tight budget, 1 sentence
    else:
        target_words  = int(clip_duration * 0.68 * 130 / 60)
        max_tokens    = min(350, int(target_words / 0.75) + 30)

    # Energy AND tone guidance per rank
    energy_map = {
        1: ("MAX hype — losing your mind, this is the greatest thing you've ever seen",
            "explosive, almost screaming, can't believe it"),
        2: ("extremely hyped — this nearly took the top spot and you're not over it",
            "breathless, emphatic"),
        3: ("solid hype — genuinely impressed, building toward the climax",
            "energetic but controlled"),
        4: ("warm but measured — this is good but you knew better was coming",
            "conversational, slightly amused"),
        5: ("casual, almost dismissive but still entertained — setting the baseline",
            "relaxed, dry wit, maybe a little sarcastic"),
    }
    energy, tone = energy_map.get(rank, ("building excitement", "energetic"))

    if is_short_clip:
        prompt = f"""You write ONE punchy voiceover sentence for a viral TikTok ranking video called "{title}".

Clip rank: #{rank} out of {total}  |  Label: "{label}"
Tone: {tone}

RULES (this is a {clip_duration:.0f}-second clip — you have room for ONE sentence only):
- Start with the rank: "At number {rank}," or "Coming in at {rank},"
- Make it creative and funny — NOT generic
- Include a quick WHY: why is it at this rank and not higher?
- Exactly {target_words} words. No more. No emojis, no hashtags.

Write ONLY the sentence:"""
    else:
        prompt = f"""You are writing the VOICEOVER for a viral TikTok ranking video called "{title}".

Clip rank: #{rank} out of {total}  |  Label: "{label}"
Energy: {energy}
Tone: {tone}

STRICT RULES — violating any of these makes this unusable:
1. START with the rank but make it punchy and creative:
   BAD:  "At number {rank}, we have an impressive moment..."
   GOOD: "At number {rank}, this guy looked Mother Nature dead in the eye and said 'not today'..."

2. Explain the WHY — explicitly say why THIS clip is at rank #{rank} and not higher or lower.
   BAD:  "The tension is palpable."
   GOOD: "I put this at #{rank} and not higher because you can see the exact moment he realizes
          he has absolutely no plan, and that panic face is worth a thousand words."

3. Use SPECIFICITY — reference something unique about this exact clip.

4. PERSONALITY — write like a funny friend on the couch. Reactions, opinions, rhetorical questions.

5. Target: ~{target_words} words (fills ~{clip_duration * 0.68:.0f}s of a {clip_duration:.0f}s clip).

6. Spoken English only. No hashtags, no emojis, no stage directions.

Write ONLY the voiceover (raw text):"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.92,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip().strip('"').strip("'")
    except Exception as e:
        print(f"[ranking/commentary] GPT failed: {e}", flush=True)
        return f"Coming in at number {rank}... {label.title()}."


def generate_pattern_interrupt_text(rank: int, total: int, title: str, label: str) -> str:
    """GPT picks a punchy 1-3 word flash text for the pattern interrupt graphic.

    This text flashes on screen at peak moment of the clip (e.g. 'WASTED', 'NO WAY',
    'LOOK AT THAT', 'INSANE'). It syncs the visual with the voiceover energy.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "WASTED" if rank > total // 2 else "INSANE"

    client = OpenAI(api_key=require_openai_key(), timeout=60, max_retries=2)
    prompt = f"""Ranking video "{title}", clip rank #{rank}/{total}, label "{label}".

Pick ONE punchy 1-3 word FLASH TEXT that appears on screen at the peak moment of this clip.
Style: like a meme reaction text. Examples: WASTED, NO WAY, WAIT WHAT, INSANE, LOOK AT THAT,
TOO EASY, LEGEND, GOAT, ACTUALLY NOT, CARRIED, SKILL ISSUE, COPE, MOMENT OF THE YEAR.

Rules:
- ALL CAPS
- 1-3 words max
- Match the energy: rank {total} (tamest) = dry/sarcastic, rank 1 (best) = explosive praise
- Never use the label text verbatim

Respond with ONLY the flash text, nothing else:"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.95,
            max_tokens=15,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "WASTED").strip().upper()
        # Safety: strip quotes, limit length
        text = text.strip('"\'').strip()
        return text[:30]
    except Exception:
        return "WASTED" if rank > total // 2 else "INSANE"


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
