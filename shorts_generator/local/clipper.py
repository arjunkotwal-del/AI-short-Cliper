"""Local clipping: ffmpeg subclip + face-aware vertical crop/split-screen + ASS captions.

Pipeline per highlight:
  1. Cut [start, end] with ffmpeg.
  2. Sample frames, detect faces, cluster into speaker groups (≤4).
  3a. If all speakers fit in one 9:16 window → single crop centered on them.
  3b. If speakers are spread out → filter_complex vstack split-screen (up to 4 panels).
  4. Generate ASS subtitle file with word-level karaoke highlighting.
  5. Burn subtitles into the reframed video with ffmpeg.
"""
import os
import shutil
import subprocess
import tempfile
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

_CASCADE_CACHE: Optional[object] = None  # cv2.CascadeClassifier or None sentinel


def _get_face_cascade():
    """Return a cv2 Haar cascade for frontal faces, or None if unavailable.

    OpenCV's C++ loader can't handle Unicode paths (the project lives under
    'Документы'). We copy the XML to a temp dir with an ASCII-only path.
    """
    global _CASCADE_CACHE
    if _CASCADE_CACHE is not None:
        return _CASCADE_CACHE

    try:
        import cv2

        # Find the cascade bundled with the cv2 package
        cv2_dir = os.path.dirname(cv2.__file__)
        candidates = [
            os.path.join(cv2_dir, "data", "haarcascade_frontalface_default.xml"),
            os.path.join(cv2_dir, "haarcascade_frontalface_default.xml"),
        ]
        src_xml = next((p for p in candidates if os.path.exists(p)), None)

        if src_xml is None:
            _CASCADE_CACHE = False
            return None

        # Copy to a process-private ASCII temp path (avoids shared /tmp race)
        tmp_dir = tempfile.mkdtemp(prefix="aishorts_cv_")
        dst_xml = os.path.join(tmp_dir, "haarcascade_frontalface_default.xml")
        shutil.copy2(src_xml, dst_xml)

        cascade = cv2.CascadeClassifier(dst_xml)
        if cascade.empty():
            _CASCADE_CACHE = False
            return None

        _CASCADE_CACHE = cascade
        return cascade
    except Exception:
        _CASCADE_CACHE = False
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


def _sample_frames(video_path: str, n_samples: int = 30) -> List[object]:
    """Extract n_samples evenly-spaced frames as numpy arrays via ffmpeg pipe."""
    try:
        import numpy as np

        w, h, fps = _probe_video(video_path)

        # Get duration
        dur_probe = subprocess.run(
            [
                FFPROBE, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, check=True, timeout=_FFMPEG_TIMEOUT,
        )
        duration = float(dur_probe.stdout.strip())

        frames = []
        for i in range(n_samples):
            t = duration * i / max(1, n_samples - 1)
            cmd = [
                FFMPEG, "-y", "-loglevel", "error",
                "-ss", f"{t:.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT)
            if result.returncode == 0 and len(result.stdout) == w * h * 3:
                frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(h, w, 3)
                frames.append(frame)
        return frames
    except Exception:
        return []


def _detect_faces_in_frames(frames: List[object]) -> List[Tuple[int, int, int]]:
    """Return list of (cx, cy, area) face detections across all frames."""
    cascade = _get_face_cascade()
    if cascade is None or not frames:
        return []

    try:
        import cv2
        detections = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            if len(faces) > 0:
                for (x, y, fw, fh) in faces:
                    cx = x + fw // 2
                    cy = y + fh // 2
                    detections.append((cx, cy, fw * fh))
        return detections
    except Exception:
        return []


def _cluster_speakers(
    detections: List[Tuple[int, int, int]],
    src_w: int,
    max_speakers: int = 4,
) -> List[Tuple[int, int]]:
    """Greedy proximity clustering → list of (cx, cy) speaker centers, sorted left-to-right.

    Two detections merge into the same cluster if their horizontal centers are
    within 20% of the source width.
    """
    if not detections:
        return []

    proximity = src_w * 0.20
    clusters: List[List[Tuple[int, int, int]]] = []

    for det in detections:
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

    # Sort clusters by total area (prominence) descending, take top max_speakers
    clusters.sort(key=lambda c: sum(d[2] for d in c), reverse=True)
    clusters = clusters[:max_speakers]

    # Compute weighted centroid per cluster (weight = area)
    centers = []
    for cluster in clusters:
        total_area = sum(d[2] for d in cluster)
        cx = int(sum(d[0] * d[2] for d in cluster) / total_area)
        cy = int(sum(d[1] * d[2] for d in cluster) / total_area)
        centers.append((cx, cy))

    # Sort left-to-right for consistent panel ordering
    centers.sort(key=lambda c: c[0])
    return centers


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


def _center_crop_ffmpeg(in_path: str, out_path: str, ar: float) -> str:
    """Pure center-crop fallback — no face detection."""
    crop_filter = (
        "crop=trunc(min(iw\\,ih*{r})/2)*2:ih:(iw-trunc(min(iw\\,ih*{r})/2)*2)/2:0"
    ).format(r=ar)
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", f"{crop_filter},scale={OUT_W}:{OUT_H}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_FFMPEG_TIMEOUT)
    return out_path


def _single_crop_ffmpeg(
    in_path: str, out_path: str, src_w: int, src_h: int, center_x: int, ar: float
) -> str:
    """Crop to ar ratio centered on center_x, scale to OUT_W x OUT_H."""
    crop_w = int(src_h * ar)
    crop_w = min(crop_w, src_w)
    # Clamp x so crop doesn't go out of bounds
    x = max(0, min(center_x - crop_w // 2, src_w - crop_w))
    crop_filter = f"crop={crop_w}:{src_h}:{x}:0,scale={OUT_W}:{OUT_H}"
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


def _split_screen_ffmpeg(
    in_path: str,
    out_path: str,
    src_w: int,
    src_h: int,
    speakers: List[Tuple[int, int]],
    ar: float,
) -> str:
    """Build a vertical split-screen with one panel per speaker."""
    n = len(speakers)
    panel_h = OUT_H // n  # each panel height in output pixels
    panel_w = OUT_W       # full output width for each panel

    # Each panel: crop a 9:16 window around the speaker, scale to panel_w x panel_h
    # The crop from source: width = src_h * (panel_w / panel_h), height = src_h
    panel_ar = panel_w / panel_h  # same as overall ar for uniform panels
    crop_w = int(src_h * panel_ar)
    crop_w = min(crop_w, src_w)

    filter_parts = []
    panel_labels = []

    for i, (cx, cy) in enumerate(speakers):
        x = max(0, min(cx - crop_w // 2, src_w - crop_w))
        label = f"p{i}"
        filter_parts.append(
            f"[0:v]crop={crop_w}:{src_h}:{x}:0,scale={panel_w}:{panel_h}[{label}]"
        )
        panel_labels.append(f"[{label}]")

    # Stack panels vertically
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
    margin_top = max(40, int(height * 0.04))

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
    silence_db: float = -35.0,
    min_silence_dur: float = 0.4,
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
) -> str:
    """Cut + smart-reframe + (optionally) burn captions for one highlight."""
    cut_path = out_path + ".cut.mp4"
    dejumped_path = out_path + ".dejumped.mp4"
    framed_path = out_path + ".framed.mp4"
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    ass_filename = os.path.basename(out_path) + ".ass"
    ass_path = os.path.join(out_dir, ass_filename)

    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _remove_silence(cut_path, dejumped_path)
        _reframe_vertical(dejumped_path, framed_path, aspect_ratio)

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
            )
            thumb_path = os.path.splitext(out_path)[0] + ".jpg"
            _extract_thumbnail(out_path, thumb_path)
            results.append({**h, "clip_url": out_path, "thumbnail": thumb_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
