"""Module 4: 9:16 assembly with smart single-crop framing + audio ducking.

Reuses MediaPipe face detection from the podcast pipeline for smart
single-crop framing. Adds audio ducking during narrator hook overlay.
"""
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_TIMEOUT = 600


def _probe(path: str) -> Tuple[int, int, float]:
    """Return (width, height, fps)."""
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, timeout=_TIMEOUT,
    )
    parts = r.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_str = parts[2] if len(parts) > 2 else "30/1"
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0
    return w, h, fps


def _cut_clip(source: str, start: float, end: float, out: str) -> str:
    """Cut subclip with re-encode."""
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", source,
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out


def _smart_crop_gaming(in_path: str, out_path: str, aspect_ratio: str = "9:16") -> str:
    """Smart single-crop reframe for gaming clips.

    Uses MediaPipe face detection from the podcast pipeline.
    Falls back to center crop if no faces detected.
    """
    from ..local.clipper import (
        _get_mediapipe_model,
        _sample_frames_1fps,
        _detect_faces_mediapipe,
        _cluster_speakers,
        _build_crop_keyframes,
        _build_crop_x_expr,
        _probe_video,
        _reframe_vertical,
    )

    src_w, src_h, fps = _probe_video(in_path)

    # Parse aspect ratio
    try:
        aw, ah = aspect_ratio.split(":")
        ratio = float(aw) / float(ah)
    except Exception:
        ratio = 9.0 / 16.0

    crop_w = min(src_w, int(src_h * ratio))
    crop_h = src_h

    # Sample 1 frame/sec and detect faces
    frames = _sample_frames_1fps(in_path)
    if not frames:
        _reframe_vertical(in_path, out_path, aspect_ratio)
        return out_path

    detections = _detect_faces_mediapipe(frames)

    # Count frames with any face
    face_frames = sum(1 for d in detections if d)
    if face_frames < len(frames) * 0.10:
        # Too few faces — center crop (typical for gameplay-heavy clips)
        _center_crop(in_path, out_path, crop_w, crop_h, src_w, src_h)
        return out_path

    # Cluster and build keyframes
    speakers = _cluster_speakers(detections, src_w)
    if not speakers:
        _center_crop(in_path, out_path, crop_w, crop_h, src_w, src_h)
        return out_path

    # Use the most prominent speaker (largest cluster)
    primary = max(speakers, key=lambda sp: sp[2])  # (cx, cy, count)
    keyframes = _build_crop_keyframes(detections, src_w, src_h, target_speaker=primary)

    if not keyframes:
        _center_crop(in_path, out_path, crop_w, crop_h, src_w, src_h)
        return out_path

    # Build ffmpeg crop expression
    crop_x_expr = _build_crop_x_expr(keyframes, crop_w, src_w)
    crop_y = max(0, (src_h - crop_h) // 2)

    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x_expr}:{crop_y},scale=720:1280:flags=lanczos",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


def _center_crop(in_path: str, out_path: str, crop_w: int, crop_h: int,
                 src_w: int, src_h: int) -> str:
    """Simple center crop — for gameplay-dominant clips with no visible face."""
    cx = (src_w - crop_w) // 2
    cy = (src_h - crop_h) // 2
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", f"crop={crop_w}:{crop_h}:{cx}:{cy},scale=720:1280:flags=lanczos",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


def _overlay_narrator_with_ducking(
    video_path: str,
    hook_audio_path: str,
    hook_duration: float,
    out_path: str,
    duck_level: float = 0.15,
) -> str:
    """Overlay narrator audio at start of clip with original audio ducked.

    During hook (0 to hook_duration): original audio at duck_level, narrator at full.
    After hook: original audio back to full.
    """
    # Build volume expression for ducking
    # Duck to 15% during narration, snap back to 100% after
    duck_expr = f"if(lt(t,{hook_duration:.2f}),{duck_level},1.0)"

    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", video_path,          # input 0: video with original audio
        "-i", hook_audio_path,     # input 1: narrator audio
        "-filter_complex",
        (
            f"[0:a]volume='{duck_expr}':eval=frame[ducked];"
            f"[1:a]adelay=0|0[narrator];"
            f"[ducked][narrator]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        ),
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


def assemble_gaming_clip(
    source_path: str,
    clip: dict,
    clip_index: int,
    out_path: str,
    hook: Optional[dict] = None,
    aspect_ratio: str = "9:16",
    words: Optional[List[Dict]] = None,
) -> str:
    """Full assembly for one gaming clip.

    1. Cut subclip from source
    2. Smart single-crop reframe (face tracking or center crop)
    3. Burn karaoke captions if words available
    4. Overlay narrator hook with audio ducking
    """
    cut_path = out_path + ".cut.mp4"
    framed_path = out_path + ".framed.mp4"
    captioned_path = out_path + ".captioned.mp4"

    try:
        # 1. Cut
        _cut_clip(source_path, clip["start_time"], clip["end_time"], cut_path)

        # 2. Smart crop
        _smart_crop_gaming(cut_path, framed_path, aspect_ratio)

        # 3. Captions
        final_before_hook = framed_path
        if words:
            try:
                from ..local.clipper import _generate_ass, _burn_captions
                fw, fh, _ = _probe(framed_path)

                ass_content = _generate_ass(
                    words, clip["start_time"], clip["end_time"], fw, fh,
                    hook_sentence=None,
                )
                if ass_content:
                    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
                    ass_filename = os.path.basename(out_path) + ".ass"
                    ass_path = os.path.join(out_dir, ass_filename)
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(ass_content)
                    _burn_captions(framed_path, captioned_path, ass_filename, out_dir)
                    final_before_hook = captioned_path
                    # Clean up ASS file
                    try:
                        os.remove(ass_path)
                    except OSError:
                        pass
            except Exception as e:
                print(f"[assembler] captions failed for clip {clip_index}: {e}", flush=True)

        # 4. Narrator hook overlay with ducking
        if hook and hook.get("audio_path") and os.path.exists(hook["audio_path"]):
            _overlay_narrator_with_ducking(
                final_before_hook,
                hook["audio_path"],
                hook["duration"],
                out_path,
                duck_level=0.15,
            )
        else:
            # No hook — just move the result
            if final_before_hook != out_path:
                shutil.copy2(final_before_hook, out_path)

    finally:
        for p in [cut_path, framed_path, captioned_path]:
            if p and os.path.exists(p) and p != out_path:
                try:
                    os.remove(p)
                except OSError:
                    pass

    return out_path
