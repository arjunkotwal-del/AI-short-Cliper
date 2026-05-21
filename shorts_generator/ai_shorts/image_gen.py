"""Image generator — DALL-E 3 scene images.

Generates one image per scene using the image_prompt from the script.
Downloads each image to disk for ffmpeg processing.
"""
import os
import urllib.request
from typing import List

from ..config import require_openai_key

# DALL-E 3 settings
IMAGE_MODEL = "dall-e-3"
IMAGE_SIZE = "1024x1792"  # Vertical — native 9:16 ratio
IMAGE_QUALITY = "standard"  # "standard" or "hd"


def generate_scene_images(
    scenes: List[dict],
    out_dir: str,
) -> List[str]:
    """Generate one DALL-E 3 image per scene.

    Returns list of image file paths (PNG).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required: pip install openai") from e

    client = OpenAI(api_key=require_openai_key(), timeout=120, max_retries=2)
    os.makedirs(out_dir, exist_ok=True)

    image_paths = []
    for i, scene in enumerate(scenes):
        prompt = scene["image_prompt"]
        idx = i + 1
        out_path = os.path.join(out_dir, f"scene_{idx:02d}.png")

        print(f"[image] generating scene {idx}/{len(scenes)}...", flush=True)

        try:
            response = client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                n=1,
            )

            image_url = response.data[0].url
            # Download the image
            urllib.request.urlretrieve(image_url, out_path)
            image_paths.append(out_path)
            print(f"[image] scene {idx} done: {out_path}", flush=True)

        except Exception as e:
            print(f"[image] scene {idx} FAILED: {e}", flush=True)
            # Create a black placeholder image so the pipeline doesn't break
            _create_placeholder(out_path)
            image_paths.append(out_path)

    return image_paths


def _create_placeholder(out_path: str) -> None:
    """Create a black 1024x1792 placeholder PNG using ffmpeg."""
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c=black:s=1024x1792:d=1",
         "-frames:v", "1", out_path],
        check=True, timeout=30,
    )
