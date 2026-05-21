"""Local clipping: ffmpeg subclip + dynamic speaker crop + ASS captions.

Pipeline per highlight:
  1. Cut [start, end] with ffmpeg.
  2. Sample 1 frame/sec, detect faces with MediaPipe, cluster speakers.
  3a. 0 faces → letterbox fallback.
  3b. 1 speaker → smooth dynamic crop that pans to follow them.
  3c. 2+ speakers → vertical split-screen, each panel tracks its speaker.
  4. Generate ASS subtitle file with word-level karaoke highlighting.
  5. Burn subtitles into the reframed video with ffmpeg.
"""
import os
import shutil
import subprocess
import tempfile
import urllib.request
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR

# ---------------------------------------------------------------------------
# Resolve ffmpeg / ffprobe at import time so PATH hijacking is caught early
# ---------------------------------------------------------------------------

def _resolve_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"{name!r} not found. Install ffmpeg and make sure it's on your PATH."
        )
    return path

FFMPEG = _resolve_binary("ffmpeg")
FFPROBE = _resolve_binary("ffprobe")

# Subprocess timeout (seconds) — a hung ffmpeg call will be killed after this
_FFMPEG_TIMEOUT = 600  # 10 min max per clip operation


# ---------------------------------------------------------------------------
# Aspect ratio helpers
# ---------------------------------------------------------------------------

def _ratio(aspect_ratio: str) -> float:
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


# ---------------------------------------------------------------------------
# ffmpeg cut
# ---------------------------------------------------------------------------

def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT)
    return out_path


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------

_MP_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
_MP_MODEL_BUFFER: Optional[bytes] = None  # cached in-memory after first load


def _get_mediapipe_model() -> Optional[bytes]:
    """Download (once) and cache the MediaPipe face detection model in memory.

    The .tflite file is ~225 KB.  We load via model_asset_buffer to avoid
    Cyrillic/Unicode path issues in MediaPipe's C++ runtime.
    """
    global _MP_MODEL_BUFFER
    if _MP_MODEL_BUFFER is not None:
        return _MP_MODEL_BUFFER

    cache_dir = os.path.join(tempfile.gettempdir(), "aishorts_models")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "blaze_face_short_range.tflite")

    try:
        if not os.path.exists(model_path):
            print("[clip/face] downloading MediaPipe face model...", flush=True)
            urllib.request.urlretrieve(_MP_MODEL_URL, model_path)
        with open(model_path, "rb") as f:
            _MP_MODEL_BUFFER = f.read()
        return _MP_MODEL_BUFFER
    except Exception as e:
        print(f"[clip/face] MediaPipe model download failed: {e}", flush=True)
        return None


def _probe_video(path: str) -> Tuple[int, int, float]:
    """Return (width, height, fps) via ffprobe."""
    probe = subprocess.run(
        [
            FFPROBE, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT,
    )
    parts = probe.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    fps_str = parts[2] if len(parts) > 2 else "30/1"
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0
    return w, h, fps


def _get_video_duration(path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    dur_probe = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT,
    )
    return float(dur_probe.stdout.strip())


def _sample_frames_1fps(video_path: str) -> List:
    """Extract 1 frame per second as numpy arrays via a single ffmpeg call.

    Uses the fps=1 filter so ffmpeg decodes once and outputs all frames
    through a pipe — much faster than one subprocess per frame.
    """
    try:
        import numpy as np

        w, h, _ = _probe_video(video_path)

        cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", "fps=1",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)
        if result.returncode != 0:
            return []

        raw = result.stdout
        frame_bytes = w * h * 3
        n_frames = len(raw) // frame_bytes
        frames = []
        for i in range(n_frames):
            start = i * frame_bytes
            frame = np.frombuffer(raw[start:start + frame_bytes], dtype=np.uint8).reshape(h, w, 3)
            frames.append(frame)
        return frames
    except Exception as e:
        print(f"[clip/face] frame sampling failed: {e}", flush=True)
        return []


def _detect_faces_mediapipe(frames: List) -> List[List[Tuple[int, int, int]]]:
    """Run MediaPipe face detection on each frame.

    Returns a list-of-lists: per_frame_detections[i] = [(cx, cy, area), ...]
    for frame i.  Each detection is in pixel coordinates of the source frame.
    """
    model_buf = _get_mediapipe_model()
    if model_buf is None or not frames:
        return [[] for _ in frames]

    try:
        import mediapipe as mp

        base_options = mp.tasks.BaseOptions(model_asset_buffer=model_buf)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        detector = mp.tasks.vision.FaceDetector.create_from_options(options)

        per_frame: List[List[Tuple[int, int, int]]] = []
        for frame in frames:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect(image)
            dets = []
            h, w = frame.shape[:2]
            for det in result.detections:
                bb = det.bounding_box
                cx = bb.origin_x + bb.width // 2
                cy = bb.origin_y + bb.height // 2
                area = bb.width * bb.height
                dets.append((cx, cy, area))
            per_frame.append(dets)

        detector.close()
        return per_frame
    except Exception as e:
        print(f"[clip/face] MediaPipe detection failed: {e}", flush=True)
        return [[] for _ in frames]


def _cluster_speakers(
    per_frame_dets: List[List[Tuple[int, int, int]]],
    src_w: int,
    max_speakers: int = 4,
) -> List[Tuple[int, int]]:
    """Cluster all face detections across frames into speaker positions.

    Returns list of (cx, cy) speaker centers sorted left-to-right.
    """
    all_dets = [d for frame_dets in per_frame_dets for d in frame_dets]
    if not all_dets:
        return []

    proximity = src_w * 0.35
    clusters: List[List[Tuple[int, int, int]]] = []

    for det in all_dets:
        cx = det[0]
        assigned = False
        for cluster in clusters:
            cluster_cx = sum(d[0] for d in cluster) / len(cluster)
            if abs(cx - cluster_cx) < proximity:
                cluster.append(det)
                assigned = True
                break
        if not assigned:
            clusters.append([det])

    # Drop tiny clusters (< 10% of total detections) — likely noise / brief glances
    min_count = max(2, len(all_dets) * 0.10)
    clusters = [c for c in clusters if len(c) >= min_count] or clusters[:1]

    # Keep the most prominent speakers
    clusters.sort(key=lambda c: sum(d[2] for d in c), reverse=True)
    clusters = clusters[:max_speakers]

    # Weighted centroid per cluster
    centers = []
    for cluster in clusters:
        total_area = sum(d[2] for d in cluster)
        cx = int(sum(d[0] * d[2] for d in cluster) / total_area)
        cy = int(sum(d[1] * d[2] for d in cluster) / total_area)
        centers.append((cx, cy))

    centers.sort(key=lambda c: c[0])
    return centers


# ---------------------------------------------------------------------------
# Dynamic crop: per-second keyframes with instant snap + hysteresis
# ---------------------------------------------------------------------------

def _build_crop_keyframes(
    per_frame_dets: List[List[Tuple[int, int, int]]],
    src_w: int,
    src_h: int,
    target_speaker: Optional[Tuple[int, int]] = None,
    hysteresis: bool = True,
) -> List[Tuple[float, int]]:
    """Build a list of (time_sec, crop_center_x) keyframes from per-frame detections.

    If target_speaker is given, prefer detections near that speaker's x position.
    Falls back to frame center when no face is detected in a frame.

    When hysteresis=True (default for single-crop), small face movements are
    suppressed — only movements > 15% of frame width emit a new keyframe.
    This prevents jitter while keeping instant snaps for real position changes.
    """
    proximity = src_w * 0.25
    raw_positions: List[Tuple[float, int]] = []

    for t, dets in enumerate(per_frame_dets):
        if dets and target_speaker:
            best = min(dets, key=lambda d: abs(d[0] - target_speaker[0]))
            if abs(best[0] - target_speaker[0]) < proximity:
                raw_positions.append((float(t), best[0]))
            else:
                raw_positions.append((float(t), target_speaker[0]))
        elif dets:
            biggest = max(dets, key=lambda d: d[2])
            raw_positions.append((float(t), biggest[0]))
        else:
            if raw_positions:
                raw_positions.append((float(t), raw_positions[-1][1]))
            else:
                raw_positions.append((float(t), src_w // 2))

    if not hysteresis:
        return raw_positions

    # Hysteresis pass: only emit a keyframe when face moves significantly
    threshold = int(src_w * 0.15)
    keyframes: List[Tuple[float, int]] = []
    held_x = raw_positions[0][1] if raw_positions else src_w // 2

    for t, cx in raw_positions:
        if abs(cx - held_x) > threshold:
            held_x = cx
            keyframes.append((t, cx))
        elif not keyframes:
            keyframes.append((t, held_x))

    # Ensure we have at least the first position
    if not keyframes and raw_positions:
        keyframes.append(raw_positions[0])

    return keyframes


def _smooth_keyframes(keyframes: List[Tuple[float, int]], window: int = 3) -> List[Tuple[float, int]]:
    """Apply a rolling average to keyframe x positions for gentle panning.

    Used only for split-screen panels, NOT for single-crop (which uses instant snap).
    """
    if len(keyframes) <= 1:
        return keyframes

    smoothed = []
    half = window // 2
    for i in range(len(keyframes)):
        start = max(0, i - half)
        end = min(len(keyframes), i + half + 1)
        avg_x = int(sum(kf[1] for kf in keyframes[start:end]) / (end - start))
        smoothed.append((keyframes[i][0], avg_x))
    return smoothed


def _build_crop_x_expr(keyframes: List[Tuple[float, int]], crop_w: int, src_w: int) -> str:
    """Build an ffmpeg expression string for the crop x position.

    Uses instant step-function snaps between keyframes (no smooth panning).
    Each keyframe is (time_sec, center_x).  The expression clamps x so the
    crop window stays within [0, src_w - crop_w].
    """
    max_x = src_w - crop_w

    if not keyframes:
        x = max(0, min(src_w // 2 - crop_w // 2, max_x))
        return str(x)

    if len(keyframes) == 1:
        x = max(0, min(keyframes[0][1] - crop_w // 2, max_x))
        return str(x)

    # Convert center_x → left_x, clamped
    def _left(cx: int) -> int:
        return max(0, min(cx - crop_w // 2, max_x))

    # Build nested if(lt(t,t1), x0, if(lt(t,t2), x1, ...))
    # Step function — instant snap at each keyframe time, no interpolation
    parts = str(_left(keyframes[-1][1]))
    for i in range(len(keyframes) - 2, -1, -1):
        t1 = keyframes[i + 1][0]
        x0 = _left(keyframes[i][1])
        parts = f"if(lt(t\\,{t1:.1f})\\,{x0}\\,{parts})"

    return parts


# ---------------------------------------------------------------------------
# Framing: single crop vs split-screen
# ---------------------------------------------------------------------------

# Output dimensions for 9:16 Shorts
OUT_W = 720
OUT_H = 1280


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Letterbox the full horizontal frame into a 9:16 container with black bars top/bottom."""
    print("[clip/framing] letterbox — full horizontal frame, black bars top/bottom", flush=True)
    vf = f"scale={OUT_W}:-2,pad={OUT_W}:{OUT_H}:0:(oh-ih)/2:black"
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT)
    return out_path


def _dynamic_single_crop(
    in_path: str,
    out_path: str,
    keyframes: List[Tuple[float, int]],
    src_w: int,
    src_h: int,
    ar: float,
) -> str:
    """Dynamically crop to follow a single speaker using smooth keyframe interpolation."""
    crop_w = min(int(src_h * ar), src_w)
    # Make crop_w even for ffmpeg
    crop_w = crop_w - (crop_w % 2)

    x_expr = _build_crop_x_expr(keyframes, crop_w, src_w)
    crop_filter = f"crop={crop_w}:{src_h}:{x_expr}:0,scale={OUT_W}:{OUT_H}"

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", crop_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT)
    return out_path


def _dynamic_split_screen(
    in_path: str,
    out_path: str,
    per_frame_dets: List[List[Tuple[int, int, int]]],
    speakers: List[Tuple[int, int]],
    src_w: int,
    src_h: int,
    ar: float,
) -> str:
    """Build a vertical split-screen where each panel dynamically tracks its speaker."""
    n = min(len(speakers), 2)  # cap at 2 panels for clean look
    panel_h = OUT_H // n
    panel_w = OUT_W

    panel_ar = panel_w / panel_h
    crop_w = min(int(src_h * panel_ar), src_w)
    crop_w = crop_w - (crop_w % 2)

    filter_parts = []
    panel_labels = []

    for i in range(n):
        kf = _build_crop_keyframes(per_frame_dets, src_w, src_h, target_speaker=speakers[i])
        kf = _smooth_keyframes(kf)
        x_expr = _build_crop_x_expr(kf, crop_w, src_w)
        label = f"p{i}"
        filter_parts.append(
            f"[0:v]crop={crop_w}:{src_h}:{x_expr}:0,scale={panel_w}:{panel_h}[{label}]"
        )
        panel_labels.append(f"[{label}]")

    stack_inputs = "".join(panel_labels)
    filter_parts.append(f"{stack_inputs}vstack=inputs={n}[v]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", in_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT)
    return out_path


def _classify_frames(per_frame_dets: List[List[Tuple[int, int, int]]]) -> List[str]:
    """Label each frame as 'single' (≤1 face) or 'multi' (≥2 faces)."""
    return ["multi" if len(d) >= 2 else "single" for d in per_frame_dets]


def _collapse_segments(labels: List[str], min_run: int = 3) -> List[Tuple[str, int, int]]:
    """Collapse frame labels into (mode, start_sec, end_sec) segments.

    Short runs (< min_run seconds) are merged into the surrounding segment
    to avoid flickering between layouts every second.
    """
    if not labels:
        return []

    # Build raw runs
    runs: List[Tuple[str, int, int]] = []  # (mode, start, end) inclusive
    cur_mode = labels[0]
    cur_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur_mode:
            runs.append((cur_mode, cur_start, i - 1))
            cur_mode = labels[i]
            cur_start = i
    runs.append((cur_mode, cur_start, len(labels) - 1))

    # Merge short runs into the previous segment
    merged: List[Tuple[str, int, int]] = [runs[0]]
    for mode, s, e in runs[1:]:
        duration = e - s + 1
        if duration < min_run:
            # Absorb into previous segment
            prev = merged[-1]
            merged[-1] = (prev[0], prev[1], e)
        else:
            merged.append((mode, s, e))

    return merged


def _cut_segment(in_path: str, out_path: str, start: float, end: float) -> str:
    """Cut a time range from a video with re-encoding for reliable short segments."""
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", in_path,
        "-to", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT)
    return out_path


def _hybrid_reframe(
    in_path: str,
    out_path: str,
    per_frame_dets: List[List[Tuple[int, int, int]]],
    speakers: List[Tuple[int, int]],
    src_w: int,
    src_h: int,
    ar: float,
    aspect_ratio: str,
) -> str:
    """Hybrid framing: switch between single-crop and split-screen within one clip.

    Segments the clip by whether 1 or 2+ faces are visible in each second.
    Short segments (<3s) are merged into neighbors to avoid flickering.
    Each segment is rendered with the appropriate layout, then concatenated.
    """
    labels = _classify_frames(per_frame_dets)
    segments = _collapse_segments(labels, min_run=3)

    # If all segments are the same mode, skip segmenting
    if len(segments) == 1:
        mode = segments[0][0]
        if mode == "single":
            kf = _build_crop_keyframes(per_frame_dets, src_w, src_h, hysteresis=True)
            return _dynamic_single_crop(in_path, out_path, kf, src_w, src_h, ar)
        else:
            return _dynamic_split_screen(in_path, out_path, per_frame_dets, speakers, src_w, src_h, ar)

    duration = _get_video_duration(in_path)
    tmp_dir = tempfile.mkdtemp(prefix="aishorts_hybrid_")
    segment_files = []

    try:
        for idx, (mode, frame_start, frame_end) in enumerate(segments):
            t_start = float(frame_start)
            # end is inclusive, so add 1 to get the end time; clamp to duration
            t_end = min(float(frame_end + 1), duration)
            if t_end <= t_start:
                continue

            seg_cut = os.path.join(tmp_dir, f"seg_{idx:02d}_cut.mp4")
            seg_framed = os.path.join(tmp_dir, f"seg_{idx:02d}.mp4")

            _cut_segment(in_path, seg_cut, t_start, t_end)

            # Slice the per-frame detections for this segment
            seg_dets = per_frame_dets[frame_start:frame_end + 1]

            if mode == "single":
                # Track the biggest face each frame — no target speaker lock
                kf = _build_crop_keyframes(seg_dets, src_w, src_h, hysteresis=True)
                _dynamic_single_crop(seg_cut, seg_framed, kf, src_w, src_h, ar)
            else:
                # Re-cluster within this segment's frames for accurate panels
                seg_speakers = _cluster_speakers(seg_dets, src_w)
                if len(seg_speakers) < 2:
                    seg_speakers = speakers[:2]  # fall back to global speakers
                kf_dets = [_build_crop_keyframes(seg_dets, src_w, src_h, target_speaker=sp, hysteresis=False)
                           for sp in seg_speakers[:2]]
                for kf in kf_dets:
                    _smooth_keyframes(kf)
                _dynamic_split_screen(seg_cut, seg_framed, seg_dets, seg_speakers, src_w, src_h, ar)

            segment_files.append(seg_framed)

        if len(segment_files) == 1:
            shutil.copy2(segment_files[0], out_path)
        else:
            # Concatenate all segments
            concat_list = os.path.join(tmp_dir, "concat.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for sf in segment_files:
                    # ffmpeg concat demuxer needs forward slashes or escaped backslashes
                    f.write(f"file '{sf.replace(os.sep, '/')}'\n")

            cmd = [
                FFMPEG, "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                out_path,
            ]
            subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT)

        return out_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _smart_reframe(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Detect faces and choose the best framing strategy automatically.

    - 0 faces → letterbox fallback
    - 1 speaker → instant-snap dynamic crop following them
    - 2+ speakers → hybrid: switches between single-crop and split-screen
      within the clip based on how many faces are visible each second
    """
    ar = _ratio(aspect_ratio)
    src_w, src_h, _ = _probe_video(in_path)

    # Sample 1 frame per second
    frames = _sample_frames_1fps(in_path)
    if not frames:
        print("[clip/framing] no frames sampled — letterbox fallback", flush=True)
        return _reframe_vertical(in_path, out_path, aspect_ratio)

    # Detect faces in every frame
    per_frame_dets = _detect_faces_mediapipe(frames)
    total_faces = sum(len(d) for d in per_frame_dets)
    frames_with_faces = sum(1 for d in per_frame_dets if d)

    if total_faces == 0:
        print("[clip/framing] no faces detected — letterbox fallback", flush=True)
        return _reframe_vertical(in_path, out_path, aspect_ratio)

    print(f"[clip/framing] {total_faces} face detections in {frames_with_faces}/{len(frames)} frames", flush=True)

    # Cluster into speaker positions
    speakers = _cluster_speakers(per_frame_dets, src_w)

    # Classify each second as single/multi face
    multi_face_frames = sum(1 for d in per_frame_dets if len(d) >= 2)

    if len(speakers) <= 1 or multi_face_frames < len(frames) * 0.15:
        # Pure single-speaker tracking
        print(f"[clip/framing] 1 speaker — instant-snap tracking", flush=True)
        target = speakers[0] if speakers else None
        kf = _build_crop_keyframes(per_frame_dets, src_w, src_h, target_speaker=target, hysteresis=True)
        return _dynamic_single_crop(in_path, out_path, kf, src_w, src_h, ar)
    else:
        # Hybrid: switch between single-crop and split-screen within the clip
        print(f"[clip/framing] {len(speakers)} speakers — hybrid mode ({multi_face_frames}/{len(frames)} multi-face frames)", flush=True)
        return _hybrid_reframe(in_path, out_path, per_frame_dets, speakers, src_w, src_h, ar, aspect_ratio)


# ---------------------------------------------------------------------------
# ASS caption generation — TikTok-style word-by-word karaoke highlighting
# ---------------------------------------------------------------------------

_WORDS_PER_CHUNK = 3  # words shown at once


def _fmt_ass_time(seconds: float) -> str:
    """Format seconds -> H:MM:SS.CC (ASS timestamp format)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    """Escape special characters for an ASS dialogue line."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")



def _generate_ass(
    words: List[Dict],
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    hook_sentence: Optional[str] = None,
) -> Optional[str]:
    """Build an ASS subtitle file with karaoke-style word highlighting.

    Primary colour (yellow) = active/just-spoken word.
    Secondary colour (white) = upcoming words in the same chunk.
    Each chunk pops in with a quick scale animation.
    If hook_sentence is provided, it is burned as large top text for the first 2.5 s.
    """
    clip_words = []
    for w in words:
        ws = float(w["start"])
        we = float(w["end"])
        word_text = w.get("word", "").strip()
        if not word_text:
            continue
        if ws >= clip_start - 0.1 and ws < clip_end + 0.1:
            clip_words.append({
                "word": word_text.upper(),
                "start": max(0.0, ws - clip_start),
                "end": min(clip_end - clip_start, we - clip_start),
            })

    if not clip_words and not hook_sentence:
        return None

    font_size = max(60, int(height * 0.055))
    hook_size = max(72, int(height * 0.065))
    margin_v = max(60, int(height * 0.06))
    margin_top = max(120, int(height * 0.15))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,Impact,{font_size},&H0000FFFF,&H00FFFFFF,&H00000000,&HCC000000,-1,0,0,0,100,100,1,0,1,4,1,2,30,30,{margin_v},1
Style: Hook,Impact,{hook_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&HE6000000,-1,0,0,0,100,100,1,0,1,5,2,8,20,20,{margin_top},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    # Hook overlay: top-center for first 2.5 seconds
    if hook_sentence:
        safe_hook = _escape_ass(hook_sentence.upper())
        events.append(
            f"Dialogue: 1,{_fmt_ass_time(0)},{_fmt_ass_time(2.5)},"
            f"Hook,,0,0,0,,{{\\an8\\fscx90\\fscy90\\t(0,200,\\fscx100\\fscy100)}}{safe_hook}"
        )

    # Karaoke word chunks
    chunks = [clip_words[i:i + _WORDS_PER_CHUNK] for i in range(0, len(clip_words), _WORDS_PER_CHUNK)]
    for chunk in chunks:
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        if chunk_end <= chunk_start:
            chunk_end = chunk_start + 1.0

        parts = [r"{\fscx90\fscy90\t(0,150,\fscx100\fscy100)}"]
        for w in chunk:
            dur_s = max(0.05, w["end"] - w["start"])
            dur_cs = int(round(dur_s * 100))
            parts.append(f"{{\\k{dur_cs}}}{w['word']} ")

        text = "".join(parts).rstrip()
        events.append(
            f"Dialogue: 0,{_fmt_ass_time(chunk_start)},{_fmt_ass_time(chunk_end)},"
            f"Pop,,0,0,0,,{text}"
        )

    return header + "\n".join(events) + "\n"


def _burn_captions(in_path: str, out_path: str, ass_filename: str, out_dir: str) -> str:
    """Burn an ASS subtitle file into a video using ffmpeg.

    Runs ffmpeg with cwd=out_dir so the ass= filter uses a relative path,
    avoiding Windows drive-letter colon escaping issues.
    """
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", os.path.abspath(in_path),
        "-vf", f"ass={ass_filename}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        os.path.abspath(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=out_dir, timeout=_FFMPEG_TIMEOUT)
    return out_path


# ---------------------------------------------------------------------------
# Silence / dead-air removal
# ---------------------------------------------------------------------------

def _remove_silence(
    in_path: str,
    out_path: str,
    silence_db: float = -40.0,
    min_silence_dur: float = 1.5,
    max_gaps: int = 6,
) -> str:
    """Detect silent gaps and hard-cut them out.

    Uses ffmpeg silencedetect to find gaps > min_silence_dur seconds at
    < silence_db dBFS, then builds a filter_complex that trims + concatenates
    only the non-silent segments.  Falls back to a plain copy if no silence
    is found or if filtering fails.
    """
    import re as _re

    # Step 1: detect silence
    detect = subprocess.run(
        [
            FFMPEG, "-i", in_path,
            "-af", f"silencedetect=n={silence_db}dB:d={min_silence_dur}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT,
    )
    stderr = detect.stderr

    starts = [float(m) for m in _re.findall(r"silence_start: (\S+)", stderr)]
    ends   = [float(m) for m in _re.findall(r"silence_end: (\S+)", stderr)]

    if not starts:
        shutil.copy2(in_path, out_path)
        return out_path

    # Too many gaps = fast-paced delivery, skip removal to avoid jarring cuts
    if len(starts) > max_gaps:
        print(f"[clip/local] silence removal: {len(starts)} gaps exceeds max ({max_gaps}), skipping", flush=True)
        shutil.copy2(in_path, out_path)
        return out_path

    # Step 2: get total duration
    try:
        dur_probe = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", in_path],
            capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT,
        )
        duration = float(dur_probe.stdout.strip())
    except Exception:
        shutil.copy2(in_path, out_path)
        return out_path

    # Step 3: build non-silent intervals
    intervals = []
    pos = 0.0
    for i, ss in enumerate(starts):
        if ss > pos + 0.05:
            intervals.append((pos, ss))
        pos = ends[i] if i < len(ends) else duration
    if pos < duration - 0.05:
        intervals.append((pos, duration))

    if not intervals or (len(intervals) == 1 and intervals[0][0] < 0.1 and intervals[0][1] > duration - 0.1):
        shutil.copy2(in_path, out_path)
        return out_path

    # Step 4: build filter_complex trim + concat
    n = len(intervals)
    filter_parts = []
    v_labels, a_labels = [], []
    for i, (s, e) in enumerate(intervals):
        filter_parts.append(f"[0:v]trim={s:.3f}:{e:.3f},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS[a{i}]")
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")

    filter_parts.append("".join(v_labels) + f"concat=n={n}:v=1:a=0[vout]")
    filter_parts.append("".join(a_labels) + f"concat=n={n}:v=0:a=1[aout]")

    try:
        subprocess.run(
            [
                FFMPEG, "-y", "-loglevel", "error",
                "-i", in_path,
                "-filter_complex", ";".join(filter_parts),
                "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                out_path,
            ],
            check=True, timeout=_FFMPEG_TIMEOUT,
        )
        removed = len(intervals)
        original_gaps = len(starts)
        print(
            f"[clip/local] silence removal: cut {original_gaps} gap(s), kept {removed} segment(s)",
            flush=True,
        )
    except Exception as e:
        print(f"[clip/local] silence removal failed ({e}), using original", flush=True)
        shutil.copy2(in_path, out_path)

    return out_path


# ---------------------------------------------------------------------------
# Thumbnail extraction
# ---------------------------------------------------------------------------

def _extract_thumbnail(clip_path: str, thumb_path: str, at_pct: float = 0.25) -> Optional[str]:
    """Extract a single frame at at_pct of the clip duration as a JPEG thumbnail."""
    try:
        dur_probe = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", clip_path],
            capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT,
        )
        duration = float(dur_probe.stdout.strip())
        at = max(0.5, duration * at_pct)
        cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-ss", f"{at:.3f}", "-i", clip_path,
            "-frames:v", "1", "-q:v", "2",
            thumb_path,
        ]
        subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT)
        return thumb_path
    except Exception as e:
        print(f"[clip/local] thumbnail failed ({e})", flush=True)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    words: Optional[List[Dict]] = None,
    hook_sentence: Optional[str] = None,
    remove_silence: bool = False,
    letterbox: bool = False,
) -> str:
    """Cut + smart reframe (or letterbox) + (optionally) burn captions for one highlight."""
    cut_path = out_path + ".cut.mp4"
    dejumped_path = out_path + ".dejumped.mp4"
    framed_path = out_path + ".framed.mp4"
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    ass_filename = os.path.basename(out_path) + ".ass"
    ass_path = os.path.join(out_dir, ass_filename)

    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        if remove_silence:
            _remove_silence(cut_path, dejumped_path)
        else:
            shutil.copy2(cut_path, dejumped_path)
        if letterbox:
            _reframe_vertical(dejumped_path, framed_path, aspect_ratio)
        else:
            _smart_reframe(dejumped_path, framed_path, aspect_ratio)

        captioned = False
        if words or hook_sentence:
            try:
                probe = subprocess.run(
                    [
                        FFPROBE, "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=width,height",
                        "-of", "csv=p=0",
                        framed_path,
                    ],
                    capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT,
                )
                parts = probe.stdout.strip().split(",")
                w, h = int(parts[0]), int(parts[1])

                ass_content = _generate_ass(
                    words or [], start_time, end_time, w, h,
                    hook_sentence=hook_sentence,
                )
                if ass_content:
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(ass_content)
                    _burn_captions(framed_path, out_path, ass_filename, out_dir)
                    captioned = True
            except Exception as e:
                print(f"[clip/local] caption burn failed ({e}), skipping captions", flush=True)

        if not captioned:
            shutil.move(framed_path, out_path)
            framed_path = None

    finally:
        for p in [cut_path, dejumped_path, framed_path, ass_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    return out_path


def _slug(title: str, idx: int, max_len: int = 45) -> str:
    """Turn a clip title into a safe filename like '01_scared_to_open_the_gift.mp4'.

    Also strips ASS/ffmpeg filter-special characters so the filename can be
    safely used inside an `ass=<filename>` ffmpeg -vf filter.
    """
    import re
    # Strip characters unsafe for filenames and ASS filter strings
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    # Extra safety: remove any remaining shell/filter-special chars
    slug = re.sub(r"[;:=\[\]{}()/\\]", "", slug)
    slug = slug[:max_len] or f"clip_{idx:02d}"
    return f"{idx:02d}_{slug}.mp4"


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    words: Optional[List[Dict]] = None,
    remove_silence: bool = False,
    letterbox: bool = False,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    total = len(highlights)
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, _slug(h.get("title", ""), i))
        print(f"[clip/local] {i}/{total}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                words=words,
                hook_sentence=h.get("hook_sentence"),
                remove_silence=remove_silence,
                letterbox=letterbox,
            )
            thumb_path = os.path.splitext(out_path)[0] + ".jpg"
            _extract_thumbnail(out_path, thumb_path)
            results.append({**h, "clip_url": out_path, "thumbnail": thumb_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
