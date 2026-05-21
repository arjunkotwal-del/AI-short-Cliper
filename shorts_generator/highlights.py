"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
  - chunking for long videos with overlap
  - virality-criteria prompt
  - score-based dedupe with overlap suppression

The LLM call is pluggable via the `llm_fn` argument so the same prompts can
drive either MuAPI (default, --mode api) or a direct OpenAI client
(--mode local).
"""
import json
import re
from typing import Callable, Dict, List, Optional


LLMFn = Callable[[str], str]


def call_openai_llm(prompt: str) -> str:
    """Default LLM backend: OpenAI via OPENAI_API_KEY."""
    from .local.llm import call_openai_llm as _call
    return _call(prompt)


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Score every clip across four dimensions (0-100 each):

HOOK (35% weight) — Does the opening line immediately stop the scroll?
  - Exclamations, questions, or bold statements that demand attention
  - Viewer must feel compelled to keep watching within 3 seconds
  - A strong hook names a curiosity gap, a surprising claim, or raw emotion

FLOW (20% weight) — Does the clip feel seamless and easy to follow?
  - Logical progression; no confusing mid-sentence cuts
  - Speaker transitions are smooth; no dead air or awkward pauses
  - Viewer never has to rewatch to understand what's happening

VALUE (25% weight) — Does watching this clip give the viewer something?
  - Entertainment: genuine reactions, celebrity moments, humor, drama
  - Information: tips, facts, revelations, lessons
  - Context enriches the moment — the viewer feels rewarded, not cheated

TREND (20% weight) — Is this riding a current cultural wave?
  - References trending people (athletes, influencers, celebrities), memes, events
  - Taps into ongoing conversations (online culture, sports, pop culture)
  - Real-time relevance increases share probability and algorithm boost

Final score = round(hook*0.35 + flow*0.20 + value*0.25 + trend*0.20) as a 0-100 integer.

CALIBRATION — Use the full 0-100 range honestly:
  - 90-100: Genuinely exceptional, would go viral with millions of views
  - 70-89: Strong content, would perform well on shorts platforms
  - 50-69: Decent but not standout — average engagement expected
  - 30-49: Below average, weak hook or incomplete payoff
  - 0-29: Poor content, no viral potential
Most clips should score between 50-80. A score above 90 should be rare (1 in 10 clips).
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

Content type: {content_type} | Density: {density}

Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- Find the SPECIFIC viral moment — the punchline, the shocking reveal, the funny reaction, the wild statement. That moment is the CENTER of the clip, not the end.
- start_time should be 5-15 seconds BEFORE the viral moment — just enough setup so the viewer understands what's happening, no more. Cut out unnecessary buildup.
- end_time should be 3-8 seconds AFTER the viral moment — enough for the reaction to land, then cut. Don't let it drag.
- Target 20-40 seconds per clip. Tight and punchy beats long and complete.
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- {num_clips_instruction}
- title must be short (3-6 words), punchy, and TikTok-style — think "he didn't see that coming" or "caught lying on camera" not "The Unexpected Revelation". Lowercase, no quotes, no generic words like "shocking" or "unexpected"
- For each highlight, write a "hook_sentence" that will be spoken as a voiceover BEFORE the clip plays. It must EXPLAIN THE CONTEXT of what's happening — like you're briefing a friend who has never seen this video. Do NOT quote or repeat lines from the transcript. Instead describe the situation, the game being played, the dynamic between people, or what's at stake. Keep it to 1 sentence, max 20 words, conversational and punchy. No profanity. Example: "These guys have to guess a celebrity using only one clue each."
- Explain in one sentence why this clip is viral ("virality_reason")
- Score each dimension independently then compute the weighted final score

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"hook_score":int,"flow_score":int,"value_score":int,"trend_score":int,"score":int,"hook_sentence":"string","virality_reason":"string"}}]}}"""


CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
GPT_CALL_TIMEOUT_SECONDS = 300  # 5-minute hard cap per LLM call




def _parse_json_loose(raw: str) -> Dict:
    """gpt-5-4 sometimes wraps JSON in markdown fences — strip and parse."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_openai_llm) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict, offset: float = 0.0) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start'] - offset:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_openai_llm,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which times out gpt-5-mini).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_clips_instruction=f"Generate at least {min_clips} highlights",
    )
    full_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    raw = llm_fn(full_prompt)
    return _parse_json_loose(raw)


def _recompute_score(h: Dict) -> Dict:
    """Recompute final score from dimension scores if all four are present."""
    hs = h.get("hook_score")
    fs = h.get("flow_score")
    vs = h.get("value_score")
    ts = h.get("trend_score")
    if all(isinstance(x, (int, float)) for x in [hs, fs, vs, ts]):
        computed = round(hs * 0.35 + fs * 0.20 + vs * 0.25 + ts * 0.20)
        return {**h, "score": max(0, min(100, computed))}
    return h


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in an OpenAI-backed callable.
    """
    llm_fn = llm_fn or call_openai_llm
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        all_highlights: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk, offset=offset)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            result = call_highlight_api(text, content_info, chunk["duration"], num_clips=num_clips, is_chunk=True, llm_fn=llm_fn)
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
        highlights = dedupe_highlights(all_highlights)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(text, content_info, duration, num_clips=num_clips, llm_fn=llm_fn)
        highlights = dedupe_highlights(result.get("highlights", []))

    highlights = [_recompute_score(h) for h in highlights]
    return {"highlights": highlights}


# ---------------------------------------------------------------------------
# Social copy generation (caption + hashtags per clip)
# ---------------------------------------------------------------------------

SOCIAL_COPY_PROMPT = """You are a viral social media strategist for TikTok, Instagram Reels, and YouTube Shorts.

Given this clip, write:
1. A punchy 150-character caption — hook the viewer instantly, no hashtags, use the energy of the moment
2. 8 hashtags (mix of broad trending tags and niche-specific ones relevant to the content)

Clip title: {title}
Opening hook: {hook}
Why it's viral: {reason}

Respond ONLY with valid JSON: {{"caption": "...", "hashtags": ["#...", "#..."]}}"""


def generate_social_copy(highlight: Dict, llm_fn: Optional[LLMFn] = None) -> Dict:
    """Generate a TikTok/IG/Shorts caption and hashtags for a clip."""
    llm_fn = llm_fn or call_openai_llm
    prompt = SOCIAL_COPY_PROMPT.format(
        title=highlight.get("title", ""),
        hook=highlight.get("hook_sentence", ""),
        reason=highlight.get("virality_reason", ""),
    )
    try:
        raw = llm_fn(prompt)
        result = _parse_json_loose(raw)
        return {
            "caption": result.get("caption", ""),
            "hashtags": result.get("hashtags", []),
        }
    except Exception:
        return {"caption": "", "hashtags": []}
