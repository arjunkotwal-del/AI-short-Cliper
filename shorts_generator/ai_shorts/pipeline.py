"""AI-generated shorts pipeline.

Takes a text prompt → generates a complete short from scratch:
  GPT script → AI images (skeleton character) → Runway video clips
  → TTS voiceover → concat + captions

Usage:  python main.py "What if humans could photosynthesize" --mode ai
"""
import os
import re
from typing import Dict, Optional

from .scriptwriter import generate_script
from .image_gen import generate_scene_images
from .video_gen import generate_scene_videos
from .voiceover import generate_scene_audio, estimate_word_timestamps
from .assembler import assemble_ai_short


def _slug(title: str, max_len: int = 45) -> str:
    """Safe filename from title."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    slug = re.sub(r"[;:=\[\]{}()/\\]", "", slug)
    return (slug[:max_len] or "ai_short") + ".mp4"


def generate_ai_short(
    prompt: str,
    output_dir: Optional[str] = None,
    voice: str = "onyx",
) -> Dict:
    """Generate a complete AI short from a text prompt.

    Pipeline:
        1. GPT writes script with scene breakdowns + skeleton context
        2. GPT Image generates skeleton character images per scene
        3. Runway converts each image → 5s video clip with motion
        4. OpenAI TTS narrates each scene
        5. Concat video clips + overlay voiceover + karaoke captions

    Returns:
        {
            "prompt": str,
            "mode": "ai",
            "title": str,
            "script": {...},
            "clip_url": str or None,
            "error": str or None,
        }
    """
    from ..config import LOCAL_OUTPUT_DIR

    base_dir = output_dir or LOCAL_OUTPUT_DIR
    slug = re.sub(r"[^\w\s-]", "", prompt.lower())[:30]
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_") or "ai_short"
    video_out_dir = os.path.join(base_dir, f"ai_{slug}")
    os.makedirs(video_out_dir, exist_ok=True)

    result = {
        "prompt": prompt,
        "mode": "ai",
        "title": "",
        "script": None,
        "clip_url": None,
        "error": None,
    }

    try:
        # 1. Script
        print("[ai] generating script...", flush=True)
        script = generate_script(prompt)
        result["script"] = script
        result["title"] = script["title"]
        scenes = script["scenes"]

        # 2. Scene images (skeleton character in different contexts)
        print(f"[ai] generating {len(scenes)} skeleton scene images...", flush=True)
        image_paths = generate_scene_images(scenes, video_out_dir)

        # 3. Voiceover (do this before video gen so we know exact durations)
        print("[ai] generating voiceover...", flush=True)
        scene_timings, audio_path = generate_scene_audio(scenes, video_out_dir, voice=voice)

        # 4. Image → Video via Runway
        motion_prompts = [s.get("motion_prompt", "slow cinematic zoom") for s in scenes]
        print(f"[ai] generating {len(scenes)} video clips via Runway...", flush=True)
        video_paths = generate_scene_videos(
            image_paths=image_paths,
            scene_prompts=motion_prompts,
            out_dir=video_out_dir,
            duration=5,  # Runway clips are 5s each
        )

        # 5. Word timestamps for captions
        word_timestamps = estimate_word_timestamps(scene_timings)

        # 6. Assemble final video
        out_filename = _slug(script["title"])
        out_path = os.path.join(video_out_dir, out_filename)

        print("[ai] assembling final video...", flush=True)
        assemble_ai_short(
            video_paths=video_paths,
            scene_timings=scene_timings,
            audio_path=audio_path,
            word_timestamps=word_timestamps,
            out_path=out_path,
            title=script["title"],
        )

        result["clip_url"] = out_path

        # Cleanup intermediate files
        for p in image_paths + video_paths:
            if p and os.path.exists(p) and p != out_path:
                try:
                    os.remove(p)
                except OSError:
                    pass
        try:
            os.remove(audio_path)
        except OSError:
            pass

        print(f"\n[ai] done! Output: {out_path}", flush=True)

    except Exception as e:
        result["error"] = str(e)
        print(f"\n[ai] FAILED: {e}", flush=True)

    return result
