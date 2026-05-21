"""Pillow overlay rendering + ffmpeg assembly for ranking clips.

Responsibilities:
  - render_ranking_overlay()  → RGBA PNG with stacked rank numbers on the left
  - render_title_card_png()   → PNG for the intro title card
  - make_title_card_video()   → convert PNG → short silent MP4
  - reframe_clip()            → smart 9:16 reframe (or letterbox fallback)
  - assemble_rank_clip()      → overlay PNG + mix TTS with audio ducking
  - concat_clips()            → join all clips into final output
"""
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_TIMEOUT = 600

# ── Rank colours (rank number → RGB) ─────────────────────────────────────────
RANK_COLORS: dict = {
    1: (255, 60, 60),      # red
    2: (74, 144, 217),     # blue
    3: (255, 215, 0),      # gold/yellow
    4: (160, 160, 160),    # gray
    5: (230, 230, 230),    # white
}

# Fall back for ranks 6+ (unusual but handle gracefully)
_EXTRA_COLORS = [
    (180, 120, 255),   # purple
    (80, 220, 120),    # green
    (255, 140, 0),     # orange
]


def _rank_color(rank: int) -> Tuple[int, int, int]:
    if rank in RANK_COLORS:
        return RANK_COLORS[rank]
    idx = (rank - 6) % len(_EXTRA_COLORS)
    return _EXTRA_COLORS[idx]


# ── Font helper ───────────────────────────────────────────────────────────────

def _get_font(size: int):
    from PIL import ImageFont
    for name in ["impact.ttf", "Impact.ttf", "arialbd.ttf", "Arial Bold.ttf",
                 "DejaVuSans-Bold.ttf", "FreeSansBold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    # Last resort: PIL's built-in bitmap font (tiny but always available)
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


# ── Ranking overlay PNG ───────────────────────────────────────────────────────

def render_ranking_overlay(
    current_rank: int,
    total_ranks: int,
    width: int,
    height: int,
    out_path: str,
) -> str:
    """Render a transparent RGBA PNG with stacked rank numbers on the left.

    Current rank: full color, large, glowing.
    Other ranks: same color at 35% opacity, smaller.
    """
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Background strip (semi-transparent dark)
    strip_w = max(110, int(width * 0.155))
    bg = Image.new("RGBA", (strip_w, height), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    bg_draw.rectangle([0, 0, strip_w - 1, height - 1], fill=(0, 0, 0, 170))
    img.alpha_composite(bg, (0, 0))

    # Distribute ranks top→bottom (highest rank number first, e.g. 5→4→3→2→1)
    ranks_top_to_bottom = list(range(total_ranks, 0, -1))

    usable_h = int(height * 0.90)
    top_pad = int(height * 0.05)
    spacing = usable_h / total_ranks

    for i, rank in enumerate(ranks_top_to_bottom):
        cy = int(top_pad + spacing * i + spacing / 2)
        cx = strip_w // 2
        is_current = (rank == current_rank)
        color = _rank_color(rank)

        if is_current:
            # ── Active rank: large + glow ──────────────────────────────────
            font_size = max(88, int(height * 0.075))
            font = _get_font(font_size)

            # Glow layer: draw text in color on a transparent layer, blur it
            glow_layer = Image.new("RGBA", (strip_w, height), (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow_layer)
            text = str(rank)
            # getbbox returns (left, top, right, bottom) relative to the anchor
            try:
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = font_size, font_size
            tx = cx - tw // 2
            ty = cy - th // 2
            # Draw glow copies
            for dx in range(-10, 11, 5):
                for dy in range(-10, 11, 5):
                    gd.text((tx + dx, ty + dy), text, font=font,
                             fill=(*color, 80))
            blurred = glow_layer.filter(ImageFilter.GaussianBlur(radius=12))
            img.alpha_composite(blurred, (0, 0))

            # Sharp text on top
            sharp = Image.new("RGBA", (strip_w, height), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sharp)
            sd.text((tx, ty), text, font=font, fill=(*color, 255))
            img.alpha_composite(sharp, (0, 0))

        else:
            # ── Inactive rank: small, dim ──────────────────────────────────
            font_size = max(54, int(height * 0.047))
            font = _get_font(font_size)
            text = str(rank)
            try:
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = font_size, font_size
            tx = cx - tw // 2
            ty = cy - th // 2

            dim = Image.new("RGBA", (strip_w, height), (0, 0, 0, 0))
            dd = ImageDraw.Draw(dim)
            dd.text((tx, ty), text, font=font, fill=(*color, 90))  # ~35% alpha
            img.alpha_composite(dim, (0, 0))

    img.save(out_path, "PNG")
    return out_path


# ── Title card ────────────────────────────────────────────────────────────────

def render_title_card_png(
    title: str,
    width: int,
    height: int,
    out_path: str,
) -> str:
    """Render a title card PNG: dark background, bold colored title text."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (10, 10, 10))
    draw = ImageDraw.Draw(img)

    # "RANKING" header — white, smaller
    header_size = max(60, int(height * 0.052))
    header_font = _get_font(header_size)
    header_text = "RANKING"
    try:
        hb = header_font.getbbox(header_text)
        hw = hb[2] - hb[0]
    except Exception:
        hw = header_size * len(header_text) // 2
    draw.text(((width - hw) // 2, int(height * 0.28)), header_text,
              font=header_font, fill=(220, 220, 220))

    # Main title — split across two lines if needed, gold color
    title_size = max(72, int(height * 0.062))
    title_font = _get_font(title_size)
    words = title.split()
    lines: List[str] = []
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        try:
            tb = title_font.getbbox(test)
            tw = tb[2] - tb[0]
        except Exception:
            tw = title_size * len(test) // 2
        if tw > width * 0.85 and current:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)

    total_text_h = len(lines) * int(title_size * 1.25)
    y_start = int(height * 0.44) - total_text_h // 2

    for li, line in enumerate(lines):
        try:
            lb = title_font.getbbox(line)
            lw = lb[2] - lb[0]
        except Exception:
            lw = title_size * len(line) // 2
        x = (width - lw) // 2
        y = y_start + li * int(title_size * 1.25)
        draw.text((x, y), line, font=title_font, fill=(255, 215, 0))  # gold

    # Trophy emoji stand-in: draw a small decorative line
    accent_y = int(height * 0.73)
    draw.rectangle([width // 2 - 60, accent_y, width // 2 + 60, accent_y + 4],
                   fill=(255, 215, 0))

    img.save(out_path, "PNG")
    return out_path


def make_title_card_video(
    png_path: str,
    duration: float,
    width: int,
    height: int,
    out_path: str,
) -> str:
    """Convert a PNG image into a short silent MP4."""
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-i", png_path,
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{duration:.2f}",
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
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
    """Reframe a clip to target dimensions.

    If already vertical (portrait), just scale/pad.
    Otherwise use smart MediaPipe face tracking, falling back to letterbox.
    """
    src_w, src_h, _ = _probe(in_path)
    target_ratio = width / height

    # If source is already portrait (within 10% of target ratio), just rescale
    src_ratio = src_w / src_h
    if src_ratio <= target_ratio * 1.10 and not letterbox:
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error", "-i", in_path,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k", out_path,
        ]
        subprocess.run(cmd, check=True, timeout=_TIMEOUT)
        return out_path

    if letterbox:
        # Simple letterbox: scale to fit, pad with black bars
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error", "-i", in_path,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k", out_path,
        ]
        subprocess.run(cmd, check=True, timeout=_TIMEOUT)
        return out_path

    # Try smart MediaPipe reframing
    try:
        from ..local.clipper import _smart_reframe
        aspect_ratio = f"{width}:{height}"
        _smart_reframe(in_path, out_path, aspect_ratio)
        # Verify output was created
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception as e:
        print(f"[ranking/reframe] smart reframe failed ({e}), using letterbox", flush=True)

    # Letterbox fallback
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error", "-i", in_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


# ── Per-clip assembly ─────────────────────────────────────────────────────────

def assemble_rank_clip(
    clip_path: str,
    rank: int,
    total_ranks: int,
    overlay_png: str,
    commentary_audio: Optional[str],
    commentary_duration: float,
    out_path: str,
) -> str:
    """Overlay ranking PNG + mix TTS commentary with audio ducking.

    Original clip audio ducks to 15% during commentary, returns to 100% after.
    """
    w, h, _ = _probe(clip_path)

    if commentary_audio and os.path.exists(commentary_audio) and commentary_duration > 0:
        duck_expr = f"if(lt(t,{commentary_duration:.2f}),0.15,1.0)"
        filter_complex = (
            # overlay the ranking PNG
            f"[0:v][1:v]overlay=0:0[vout];"
            # duck original audio during commentary
            f"[0:a]volume='{duck_expr}':eval=frame[ducked];"
            # boost commentary audio (1.8x so it's clearly audible)
            f"[2:a]volume=1.8[hook];"
            f"[ducked][hook]amix=inputs=2:duration=first:"
            f"dropout_transition=0:normalize=0[aout]"
        )
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error",
            "-i", clip_path,           # 0: video clip
            "-i", overlay_png,         # 1: ranking overlay PNG
            "-i", commentary_audio,    # 2: TTS commentary
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]
    else:
        # No commentary audio — just overlay the PNG
        filter_complex = "[0:v][1:v]overlay=0:0[vout]"
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error",
            "-i", clip_path,
            "-i", overlay_png,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]

    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path


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
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]
        subprocess.run(cmd, check=True, timeout=_TIMEOUT)
        return out_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
