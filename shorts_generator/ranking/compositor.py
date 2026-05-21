"""Pillow overlay rendering + ffmpeg assembly for ranking clips.

Responsibilities:
  - render_ranking_overlay()  -> RGBA PNG: top title bar + left rank list
  - render_title_card_png()   -> PNG for the intro title card
  - make_title_card_video()   -> convert PNG -> short silent MP4
  - reframe_clip()            -> smart 9:16 reframe (or letterbox fallback)
  - assemble_rank_clip()      -> overlay PNG + mix TTS with audio ducking
  - concat_clips()            -> join all clips into final output
"""
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_TIMEOUT = 600

# ── Rank colours matching reference image ─────────────────────────────────────
# rank 1 = gold/warm (best), descending to green (least extreme)
_RANK_PALETTE = [
    (255, 215,   0),  # rank 1 — gold
    (255,  80,  50),  # rank 2 — red-orange
    (255, 155,   0),  # rank 3 — orange
    (120, 220,  50),  # rank 4 — lime
    ( 60, 185,  80),  # rank 5 — green
    ( 30, 155, 110),  # rank 6 — teal-green
    (180, 120, 255),  # rank 7 — purple (fallback)
    ( 80, 200, 200),  # rank 8 — cyan  (fallback)
]


def _rank_color(rank: int) -> Tuple[int, int, int]:
    idx = max(0, rank - 1)
    return _RANK_PALETTE[idx % len(_RANK_PALETTE)]


# ── Font helper ───────────────────────────────────────────────────────────────

def _get_font(size: int):
    from PIL import ImageFont
    for name in ["impact.ttf", "Impact.ttf", "arialbd.ttf", "Arial Bold.ttf",
                 "DejaVuSans-Bold.ttf", "FreeSansBold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _text_size(font, text: str) -> Tuple[int, int]:
    """Return (width, height) of rendered text."""
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        return len(text) * (font.size if hasattr(font, "size") else 12), 20


def _draw_outlined(draw, xy, text, font, fill, outline=(0, 0, 0), stroke=3):
    """Draw text with a solid outline (for readability over video)."""
    x, y = xy
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font,
                          fill=(*outline, fill[3] if len(fill) == 4 else 255))
    draw.text((x, y), text, font=font, fill=fill)


# ── Ranking overlay PNG ───────────────────────────────────────────────────────

def render_ranking_overlay(
    current_rank: int,
    total_ranks: int,
    width: int,
    height: int,
    out_path: str,
    clip_names: Optional[Dict[int, str]] = None,   # {rank: "short label"}
    title: Optional[str] = None,
) -> str:
    """Render a transparent RGBA overlay matching the reference design:

    - Top black bar with coloured title text
    - Left-aligned rank list: [coloured number] [white label]
    - Current rank: larger text, full opacity
    - Other ranks: smaller, 65% opacity
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Top title bar ─────────────────────────────────────────────────────────
    bar_h = int(height * 0.145)
    draw.rectangle([0, 0, width, bar_h], fill=(0, 0, 0, 220))

    if title:
        words = title.upper().split()
        title_font_size = max(44, int(height * 0.040))
        font = _get_font(title_font_size)

        # "RANKING" = lime, rest = white (all caps, Impact)
        LIME = (100, 255, 80, 255)
        WHITE = (255, 255, 255, 255)
        word_colors = [LIME if i == 0 else WHITE for i in range(len(words))]

        # Wrap words into lines that fit within bar width (90% of frame)
        max_w = int(width * 0.90)
        lines: List[List[Tuple[str, tuple]]] = []
        current_line: List[Tuple[str, tuple]] = []
        current_w = 0
        gap = int(title_font_size * 0.28)  # gap between words

        for word, color in zip(words, word_colors):
            ww, _ = _text_size(font, word)
            needed = ww + (gap if current_line else 0)
            if current_line and current_w + needed > max_w:
                lines.append(current_line)
                current_line = [(word, color)]
                current_w = ww
            else:
                current_line.append((word, color))
                current_w += needed

        if current_line:
            lines.append(current_line)

        # Measure total text block height and center vertically in bar
        line_h = int(title_font_size * 1.20)
        total_text_h = len(lines) * line_h
        y0 = (bar_h - total_text_h) // 2

        for li, line_words in enumerate(lines):
            # Measure line total width
            line_w = sum(_text_size(font, w)[0] for w, _ in line_words) + gap * (len(line_words) - 1)
            x = (width - line_w) // 2
            ly = y0 + li * line_h
            for word, color in line_words:
                ww, _ = _text_size(font, word)
                _draw_outlined(draw, (x, ly), word, font, color, stroke=3)
                x += ww + gap

    # ── Rank list (left side, below title bar) ────────────────────────────────
    list_top = bar_h + int(height * 0.035)
    list_bottom = int(height * 0.97)
    list_h = list_bottom - list_top
    item_h = list_h / total_ranks

    left_x = int(width * 0.030)  # left margin

    for i in range(total_ranks):
        rank = i + 1  # rank 1 at top, rank N at bottom
        is_current = (rank == current_rank)

        color = _rank_color(rank)
        alpha = 255 if is_current else 165   # 65% for non-current

        num_size  = max(52, int(height * (0.060 if is_current else 0.048)))
        label_size = max(32, int(height * (0.036 if is_current else 0.029)))

        num_font   = _get_font(num_size)
        label_font = _get_font(label_size)

        y_center = int(list_top + item_h * i + item_h / 2)

        # Draw rank number
        num_text = str(rank)
        nw, nh = _text_size(num_font, num_text)
        ny = y_center - nh // 2
        _draw_outlined(draw, (left_x, ny), num_text, num_font,
                       (*color, alpha), stroke=4)

        # Draw label next to number (if provided)
        label = (clip_names or {}).get(rank, "")
        if label:
            lw, lh = _text_size(label_font, label)
            lx = left_x + nw + int(width * 0.018)
            ly = y_center - lh // 2
            _draw_outlined(draw, (lx, ly), label.upper(), label_font,
                           (255, 255, 255, alpha), stroke=3)

    img.save(out_path, "PNG")
    return out_path


# ── Title card (intro) ────────────────────────────────────────────────────────

def render_title_card_png(
    title: str,
    width: int,
    height: int,
    out_path: str,
) -> str:
    """Render intro title card: black BG, lime 'RANKING' + white topic text."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (8, 8, 8))
    draw = ImageDraw.Draw(img)

    words = title.upper().split()
    font_size = max(72, int(height * 0.065))
    font = _get_font(font_size)

    LIME  = (100, 255, 80)
    WHITE = (255, 255, 255)
    word_colors = [LIME if i == 0 else WHITE for i in range(len(words))]

    # Wrap into lines
    max_w = int(width * 0.88)
    gap = int(font_size * 0.28)
    lines: List[List[Tuple[str, tuple]]] = []
    cur_line: List[Tuple[str, tuple]] = []
    cur_w = 0

    for word, color in zip(words, word_colors):
        ww, _ = _text_size(font, word)
        needed = ww + (gap if cur_line else 0)
        if cur_line and cur_w + needed > max_w:
            lines.append(cur_line)
            cur_line = [(word, color)]
            cur_w = ww
        else:
            cur_line.append((word, color))
            cur_w += needed
    if cur_line:
        lines.append(cur_line)

    line_h = int(font_size * 1.25)
    total_h = len(lines) * line_h
    y0 = height // 2 - total_h // 2 - int(height * 0.05)

    for li, line_words in enumerate(lines):
        lw = sum(_text_size(font, w)[0] for w, _ in line_words) + gap * (len(line_words) - 1)
        x = (width - lw) // 2
        ly = y0 + li * line_h
        for word, color in line_words:
            ww, _ = _text_size(font, word)
            _draw_outlined(draw, (x, ly), word, font, (*color, 255), stroke=4)
            x += ww + gap

    # Gold accent line below title
    accent_y = y0 + total_h + int(height * 0.04)
    draw.rectangle([width // 2 - 80, accent_y, width // 2 + 80, accent_y + 5],
                   fill=(255, 215, 0))

    img.save(out_path, "PNG")
    return out_path


def make_title_card_video(
    png_path: str,
    duration: float,
    width: int,
    height: int,
    out_path: str,
    audio_path: Optional[str] = None,
) -> str:
    """Convert a PNG into a short MP4, optionally with a TTS audio track."""
    if audio_path and os.path.exists(audio_path):
        # Mix TTS voice with silence pad so the video is exactly `duration` long
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error",
            "-loop", "1", "-i", png_path,
            "-i", audio_path,
            "-t", f"{duration:.2f}",
            "-vf", f"scale={width}:{height}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out_path,
        ]
    else:
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error",
            "-loop", "1", "-i", png_path,
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", f"{duration:.2f}",
            "-vf", f"scale={width}:{height}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            out_path,
        ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


# ── Video probing & reframing ─────────────────────────────────────────────────

def _probe(path: str) -> Tuple[int, int, float]:
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


def reframe_clip(in_path: str, out_path: str, width: int, height: int,
                 letterbox: bool = False) -> str:
    """Reframe clip to target dimensions using smart face tracking or letterbox."""
    src_w, src_h, _ = _probe(in_path)
    src_ratio = src_w / src_h
    target_ratio = width / height

    # Already portrait-ish — just scale/pad
    if src_ratio <= target_ratio * 1.10 and not letterbox:
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error", "-i", in_path,
            "-vf", (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", out_path,
        ]
        subprocess.run(cmd, check=True, timeout=_TIMEOUT)
        return out_path

    if letterbox:
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error", "-i", in_path,
            "-vf", (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", out_path,
        ]
        subprocess.run(cmd, check=True, timeout=_TIMEOUT)
        return out_path

    # Smart MediaPipe reframing
    try:
        from ..local.clipper import _smart_reframe
        _smart_reframe(in_path, out_path, f"{width}:{height}")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            # Re-encode to ensure yuv420p
            tmp = out_path + ".tmp.mp4"
            os.rename(out_path, tmp)
            cmd = [
                _FFMPEG, "-y", "-loglevel", "error", "-i", tmp,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", out_path,
            ]
            subprocess.run(cmd, check=True, timeout=_TIMEOUT)
            os.remove(tmp)
            return out_path
    except Exception as e:
        print(f"[ranking/reframe] smart reframe failed ({e}), using letterbox", flush=True)

    # Letterbox fallback
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error", "-i", in_path,
        "-vf", (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


# ── Per-clip assembly ─────────────────────────────────────────────────────────

def _make_announcement_clip(
    overlay_png: str,
    tts_audio: str,
    tts_duration: float,
    width: int,
    height: int,
    out_path: str,
) -> str:
    """Brief black-frame clip with overlay + TTS audio at full volume.

    This plays BEFORE the actual clip so the voice is always clearly audible.
    Duration = TTS length + 0.4 s of silence padding.
    """
    total_dur = tts_duration + 0.4
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        # black video source
        "-f", "lavfi", "-i", f"color=c=black:size={width}x{height}:rate=30",
        # overlay PNG
        "-i", overlay_png,
        # TTS audio
        "-i", tts_audio,
        "-filter_complex",
        # overlay on black, then pad audio to exactly total_dur
        f"[0:v][1:v]overlay=0:0[vout];"
        f"[2:a]apad,atrim=0:{total_dur:.3f}[aout]",
        "-map", "[vout]", "-map", "[aout]",
        "-t", f"{total_dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


def _apply_overlay(clip_path: str, overlay_png: str, out_path: str) -> str:
    """Overlay ranking PNG on clip, preserving original audio untouched."""
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", clip_path,
        "-i", overlay_png,
        "-filter_complex", "[0:v][1:v]overlay=0:0[vout]",
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


def assemble_rank_clip(
    clip_path: str,
    rank: int,
    total_ranks: int,
    overlay_png: str,
    commentary_audio: Optional[str],
    commentary_duration: float,
    out_path: str,
) -> str:
    """Build a rank clip as: [announcement] + [clip with overlay].

    The announcement is a brief black-frame segment where the voice reads
    the rank label at full volume — no ducking, no mixing complexity.
    The actual clip follows immediately with its original audio fully intact.
    """
    tmp_dir = tempfile.mkdtemp(prefix="rank_assemble_")
    try:
        w, h, _ = _probe(clip_path)
        parts: List[str] = []

        # Part 1: announcement (voice reads label over black + overlay)
        if commentary_audio and os.path.exists(commentary_audio) and commentary_duration > 0:
            ann_path = os.path.join(tmp_dir, "announcement.mp4")
            _make_announcement_clip(
                overlay_png, commentary_audio, commentary_duration,
                w, h, ann_path,
            )
            parts.append(ann_path)

        # Part 2: actual clip with overlay PNG (original audio preserved)
        clip_overlay = os.path.join(tmp_dir, "clip_overlay.mp4")
        _apply_overlay(clip_path, overlay_png, clip_overlay)
        parts.append(clip_overlay)

        # Concat announcement + clip
        if len(parts) == 1:
            import shutil as _shutil
            _shutil.copy2(parts[0], out_path)
        else:
            concat_clips(parts, out_path)

        return out_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Final concat ──────────────────────────────────────────────────────────────

def concat_clips(clip_paths: List[str], out_path: str) -> str:
    """Concatenate a list of MP4 clips into a single output file."""
    tmp_dir = tempfile.mkdtemp(prefix="ranking_concat_")
    try:
        concat_list = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in clip_paths:
                f.write(f"file '{p.replace(os.sep, '/')}'\n")

        cmd = [
            _FFMPEG, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]
        subprocess.run(cmd, check=True, timeout=_TIMEOUT)
        return out_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
