"""Script generator — turns a user prompt into a narrated short script.

Outputs a structured script with:
- Full narration text (what TTS will read)
- Scene breakdowns: each scene gets an image prompt (skeleton character)
  and a motion prompt (for Runway image-to-video)
- Timing hints per scene
"""
import json
import re
from typing import Dict, List

from ..config import OPENAI_MODEL, require_openai_key


# Base skeleton character prompt — every scene image uses this as foundation
SKELETON_BASE = (
    "ultra realistic medical anatomy visualization, full body transparent human figure with "
    "clear glass-like skin revealing the complete human skeleton inside, anatomically "
    "accurate bones including skull, rib cage, spine, pelvis, arms, hands, legs and feet "
    "clearly visible through the transparent body shell, "
    "bones have a natural realistic ivory / pale yellow bone color with subtle beige tones, "
    "realistic bone texture and shading, medical-grade anatomical accuracy, "
    "outer body made of clear translucent silicone or glass material, smooth human body "
    "shape surrounding the skeleton while remaining fully transparent, "
    "extremely detailed bones, photorealistic 3D render, ultra sharp focus"
)


SCRIPT_SYSTEM_PROMPT = """You are a viral TikTok/YouTube Shorts script writer.

You write scripts for AI-narrated "what if" / "did you know" / "imagine if" style shorts.
The visual style is a transparent glass-skin skeleton character in different scenes.

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

For each scene, write:
1. "narration" — what TTS reads
2. "scene_context" — short description of what the skeleton character is doing / where it is (e.g. "standing in a glowing classroom", "floating in outer space", "brain glowing inside the skull"). This will be appended to the base skeleton prompt.
3. "motion_prompt" — short description of the motion/camera movement for the video clip (e.g. "slow zoom into the skull as the brain pulses with light", "camera orbits around the figure as energy radiates outward"). Keep it under 20 words.
4. "duration_hint" — approximate seconds

Respond in this EXACT JSON format (no markdown, no code fences):
{
  "title": "short catchy lowercase title for the video",
  "scenes": [
    {
      "narration": "the text TTS will read for this scene",
      "scene_context": "what the skeleton is doing and the environment",
      "motion_prompt": "camera/motion description for video generation",
      "duration_hint": 8
    }
  ]
}"""


def generate_script(user_prompt: str) -> Dict:
    """Generate a structured script from a user's concept/prompt.

    Returns dict with 'title', 'scenes' list, and 'skeleton_base' prompt.
    Each scene gets a full 'image_prompt' built from skeleton base + scene_context.
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

    # Build full image prompts from skeleton base + scene context
    for i, scene in enumerate(scenes):
        if not scene.get("narration"):
            raise RuntimeError(f"Scene {i+1} missing narration")
        scene.setdefault("scene_context", "standing in a dark studio")
        scene.setdefault("motion_prompt", "slow cinematic zoom in")
        scene.setdefault("duration_hint", 8)

        # Combine skeleton base with scene-specific context
        scene["image_prompt"] = (
            f"{SKELETON_BASE}, {scene['scene_context']}, "
            "plain dark background, soft dramatic studio lighting, subtle shadow under the feet"
        )

    title = script.get("title", "ai generated short")
    print(f"[script] title: {title}", flush=True)
    print(f"[script] {len(scenes)} scenes, ~{sum(s['duration_hint'] for s in scenes)}s total", flush=True)
    for i, s in enumerate(scenes, 1):
        print(f"  scene {i}: {s['narration'][:60]}...", flush=True)
        print(f"           motion: {s['motion_prompt']}", flush=True)

    return {"title": title, "scenes": scenes, "skeleton_base": SKELETON_BASE}
