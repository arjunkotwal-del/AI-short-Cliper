"""Image generator — GPT Image scene images.

Generates one image per scene using the image_prompt from the script.
Uses gpt-image-1 model with base64 output, saves to disk for ffmpeg.
"""
import base64
import os
from typing import List

from ..config import require_openai_key

# Image model settings
IMAGE_MODEL = "gpt-image-1"
IMAGE_SIZE = "1024x1536"  # Closest vertical ratio available (2:3)
IMAGE_QUALITY = "low"  # "low", "medium", or "high"


def generate_scene_images(
    scenes: List[dict],
    out_dir: str,
) -> List[str]:
    """Generate one image per scene using GPT Image model.

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
            result = client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                n=1,
            )

            # gpt-image-1 returns b64_json by default
            image_data = result.data[0]
            if hasattr(image_data, "b64_json") and image_data.b64_json:
                img_bytes = base64.b64decode(image_data.b64_json)
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
            elif hasattr(image_data, "url") and image_data.url:
                import urllib.request
                urllib.request.urlretrieve(image_data.url, out_path)
            else:
                raise RuntimeError("No image data in response")

            image_paths.append(out_path)
            print(f"[image] scene {idx} done: {out_path}", flush=True)

        except Exception as e:
            print(f"[image] scene {idx} FAILED: {e}", flush=True)
            _create_placeholder(out_path)
            image_paths.append(out_path)

    return image_paths


def _create_placeholder(out_path: str) -> None:
    """Create a black 1024x1536 placeholder PNG using ffmpeg."""
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=1024x1536:d=1",
         "-frames:v", "1", out_path],
        check=True, timeout=30,
    )
