"""Script generator — turns a user prompt into a narrated short script.

Outputs a structured script with:
- Full narration text (what TTS will read)
- Scene breakdowns: each scene has narration chunk + DALL-E image prompt
- Timing hints per scene
"""
import json
import re
from typing import Dict, List

from ..config import OPENAI_MODEL, require_openai_key


SCRIPT_SYSTEM_PROMPT = """You are a viral TikTok/YouTube Shorts script writer.

You write scripts for AI-narrated "what if" / "did you know" / "imagine if" style shorts.

Rules:
- Total narration: 45-60 seconds when spoken (roughly 120-160 words)
- Split into exactly 5-6 SCENES
- Each scene is 1-3 sentences of narration
- Start with a killer hook in scene 1 (question or shocking statement)
- Build tension through the middle scenes
- End with a mind-blowing conclusion or twist
- Use simple, conversational language — like explaining to a friend
- Present tense, direct ("you wake up", "your brain starts...")
- No emojis, no hashtags

For each scene, also write a DALL-E image prompt that matches the narration.
Image prompts must specify: "3D cinematic render, dramatic lighting, dark background"

Respond in this EXACT JSON format (no markdown, no code fences):
{
  "title": "short catchy lowercase title for the video",
  "scenes": [
    {
      "narration": "the text TTS will read for this scene",
      "image_prompt": "detailed DALL-E prompt for the visual, 3D cinematic render style",
      "duration_hint": 8
    }
  ]
}

duration_hint is approximate seconds for that scene's narration."""


def generate_script(user_prompt: str) -> Dict:
    """Generate a structured script from a user's concept/prompt.

    Returns dict with 'title' and 'scenes' list.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required: pip install openai") from e

    client = OpenAI(api_key=require_openai_key(), timeout=90, max_retries=2)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.85,
        messages=[
            {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Write a script for this concept:\n\n{user_prompt}"},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()

    # Strip markdown code fences if GPT wraps it
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        script = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Script generation returned invalid JSON: {e}\nRaw: {raw[:500]}")

    scenes = script.get("scenes", [])
    if not scenes:
        raise RuntimeError("Script has no scenes")
    if len(scenes) < 3:
        raise RuntimeError(f"Script only has {len(scenes)} scenes, need at least 3")

    # Validate each scene has required fields
    for i, scene in enumerate(scenes):
        if not scene.get("narration"):
            raise RuntimeError(f"Scene {i+1} missing narration")
        if not scene.get("image_prompt"):
            raise RuntimeError(f"Scene {i+1} missing image_prompt")
        scene.setdefault("duration_hint", 8)

    title = script.get("title", "ai generated short")
    print(f"[script] title: {title}", flush=True)
    print(f"[script] {len(scenes)} scenes, ~{sum(s['duration_hint'] for s in scenes)}s total", flush=True)
    for i, s in enumerate(scenes, 1):
        print(f"  scene {i}: {s['narration'][:60]}...", flush=True)

    return {"title": title, "scenes": scenes}
