"""Video generator — Runway Gen-3/Gen-4 image-to-video.

Takes AI-generated scene images and converts each into a 5-second
video clip with actual motion using Runway's API.
"""
import os
import time
import urllib.request
from typing import List, Optional

_TIMEOUT_SECS = 300  # max wait per video generation
_POLL_INTERVAL = 10  # seconds between status checks

# Runway model — gen4_turbo is latest, falls back to gen3a_turbo
RUNWAY_MODEL = "gen3a_turbo"


def _get_runway_client():
    """Get RunwayML client with API key from env."""
    try:
        from runwayml import RunwayML
    except ImportError:
        raise RuntimeError(
            "runwayml is required for AI video generation. Install it with:\n"
            "    pip install runwayml\n"
            "Then set RUNWAYML_API_SECRET in your .env file."
        )

    api_key = os.getenv("RUNWAYML_API_SECRET", "").strip()
    if not api_key:
        raise RuntimeError(
            "RUNWAYML_API_SECRET is not set. Add it to your .env file:\n"
            "    RUNWAYML_API_SECRET=your_runway_api_key"
        )

    return RunwayML(api_key=api_key)


def _upload_image_for_runway(image_path: str) -> str:
    """Convert a local image to a data URI for Runway API.

    Runway accepts either a URL or a base64 data URI.
    """
    import base64
    import mimetypes

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/png"

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{data}"


def generate_scene_videos(
    image_paths: List[str],
    scene_prompts: List[str],
    out_dir: str,
    duration: int = 5,
) -> List[str]:
    """Generate video clips from scene images using Runway API.

    Args:
        image_paths: list of local image file paths
        scene_prompts: motion/action prompts for each scene
        out_dir: directory to save output videos
        duration: clip duration in seconds (5 or 10)

    Returns:
        list of output video file paths
    """
    client = _get_runway_client()
    os.makedirs(out_dir, exist_ok=True)

    video_paths = []

    for i, (img_path, motion_prompt) in enumerate(zip(image_paths, scene_prompts)):
        idx = i + 1
        out_path = os.path.join(out_dir, f"scene_video_{idx:02d}.mp4")

        print(f"[video] generating scene {idx}/{len(image_paths)} video...", flush=True)

        try:
            # Upload image as data URI
            image_uri = _upload_image_for_runway(img_path)

            # Create image-to-video task
            task = client.image_to_video.create(
                model=RUNWAY_MODEL,
                prompt_image=image_uri,
                prompt_text=motion_prompt,
                duration=duration,
                ratio="720:1280",  # 9:16 vertical
            )

            task_id = task.id
            print(f"[video] scene {idx}: task {task_id} started, polling...", flush=True)

            # Poll for completion
            elapsed = 0
            time.sleep(10)
            elapsed += 10

            result = client.tasks.retrieve(task_id)
            while result.status not in ("SUCCEEDED", "FAILED"):
                if elapsed >= _TIMEOUT_SECS:
                    print(f"[video] scene {idx}: timed out after {_TIMEOUT_SECS}s", flush=True)
                    break
                time.sleep(_POLL_INTERVAL)
                elapsed += _POLL_INTERVAL
                result = client.tasks.retrieve(task_id)

            if result.status == "SUCCEEDED" and result.output:
                video_url = result.output[0]
                urllib.request.urlretrieve(video_url, out_path)
                video_paths.append(out_path)
                print(f"[video] scene {idx} done ({elapsed}s): {out_path}", flush=True)
            else:
                print(f"[video] scene {idx} FAILED: status={result.status}", flush=True)
                # Fall back to zoompan on the image
                _fallback_zoompan(img_path, duration, out_path)
                video_paths.append(out_path)

        except Exception as e:
            print(f"[video] scene {idx} error: {e}", flush=True)
            _fallback_zoompan(img_path, duration, out_path)
            video_paths.append(out_path)

    return video_paths


def _fallback_zoompan(image_path: str, duration: float, out_path: str) -> str:
    """Fallback: create zoompan clip from static image if Runway fails."""
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    fps = 30
    total_frames = int(duration * fps)

    zp = (f"zoompan=z='1+0.2*on/{total_frames}':"
          f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
          f"d={total_frames}:s=720x1280:fps={fps}")

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-loop", "1", "-i", image_path,
        "-vf", zp,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=120)
    print(f"[video] fallback zoompan created: {out_path}", flush=True)
    return out_path
