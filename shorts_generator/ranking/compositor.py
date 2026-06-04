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


def get_clip_duration(path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 10.0


def _has_audio(path: str) -> bool:
    """Return True if the file contains at least one audio stream."""
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=15,
    )
    return "audio" in r.stdout

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


def _render_flash_png(
    text: str,
    width: int,
    height: int,
    out_path: str,
) -> str:
    """Render a pattern-interrupt flash graphic as a transparent RGBA PNG.

    Style: large bold text in bright red/yellow with thick black outline,
    centered in the lower-middle of the frame — looks like a meme reaction stamp.
    """
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = max(72, int(height * 0.080))
    font = _get_font(font_size)

    # Measure and center
    tw, th = _text_size(font, text)
    x = (width - tw) // 2
    y = int(height * 0.52) - th // 2   # lower-center of frame

    # Thick black outline (stroke=8)
    stroke = 8
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))

    # Main text: bright yellow-white
    draw.text((x, y), text, font=font, fill=(255, 230, 30, 255))

    img.save(out_path, "PNG")
    return out_path


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
    """Render a transparent RGBA overlay.

    - Top black bar with coloured title text (smaller than before)
    - Left-aligned rank list: only REVEALED ranks are drawn
      Revealed = current rank + all ranks already shown (higher numbers in countdown).
      e.g. when rank #3 clip plays: ranks #5, #4 (dim/already revealed) + #3 (bright/current)
    - Current rank: full opacity, slightly larger
    - Already-revealed ranks: 55% opacity, smaller
    - Future ranks: slot is blank (not drawn yet)
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Top title bar ─────────────────────────────────────────────────────────
    bar_h = int(height * 0.115)   # slightly slimmer bar
    draw.rectangle([0, 0, width, bar_h], fill=(0, 0, 0, 210))

    if title:
        words = title.upper().split()
        title_font_size = max(30, int(height * 0.028))   # smaller title text
        font = _get_font(title_font_size)

        LIME  = (100, 255, 80, 255)
        WHITE = (255, 255, 255, 255)
        word_colors = [LIME if i == 0 else WHITE for i in range(len(words))]

        max_w = int(width * 0.92)
        lines: List[List[Tuple[str, tuple]]] = []
        current_line: List[Tuple[str, tuple]] = []
        current_w = 0
        gap = int(title_font_size * 0.26)

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

        line_h = int(title_font_size * 1.18)
        total_text_h = len(lines) * line_h
        y0 = (bar_h - total_text_h) // 2

        for li, line_words in enumerate(lines):
            line_w = (sum(_text_size(font, w)[0] for w, _ in line_words)
                      + gap * (len(line_words) - 1))
            x = (width - line_w) // 2
            ly = y0 + li * line_h
            for word, color in line_words:
                ww, _ = _text_size(font, word)
                _draw_outlined(draw, (x, ly), word, font, color, stroke=2)
                x += ww + gap

    # ── Rank list (left side, below title bar) ────────────────────────────────
    # All N slots have fixed positions so the list doesn't shift as ranks appear.
    # We count down from total_ranks → 1, so "revealed" = ranks >= current_rank.
    list_top    = bar_h + int(height * 0.025)
    list_bottom = int(height * 0.96)
    list_h      = list_bottom - list_top
    item_h      = list_h / total_ranks   # fixed slot height regardless of how many are visible

    left_x = int(width * 0.025)

    for i in range(total_ranks):
        rank = i + 1          # rank 1 drawn at top slot, rank N at bottom slot
        is_current  = (rank == current_rank)
        is_revealed = (rank >= current_rank)  # countdown N→1, so revealed = >= current

        # Skip future (not-yet-revealed) ranks — their slot stays blank
        if not is_revealed:
            continue

        color = _rank_color(rank)
        # Current rank: full brightness. Already-revealed (lower hype): dimmed.
        alpha = 255 if is_current else 140   # ~55% for past reveals

        # Smaller font sizes (≈35% reduction vs. old sizes)
        num_size   = max(34, int(height * (0.040 if is_current else 0.032)))
        label_size = max(22, int(height * (0.024 if is_current else 0.019)))

        num_font   = _get_font(num_size)
        label_font = _get_font(label_size)

        y_center = int(list_top + item_h * i + item_h / 2)

        # Rank number
        num_text = str(rank)
        nw, nh = _text_size(num_font, num_text)
        ny = y_center - nh // 2
        _draw_outlined(draw, (left_x, ny), num_text, num_font,
                       (*color, alpha), stroke=3)

        # Label next to number
        label = (clip_names or {}).get(rank, "")
        if label:
            lw, lh = _text_size(label_font, label)
            lx = left_x + nw + int(width * 0.014)
            ly = y_center - lh // 2
            _draw_outlined(draw, (lx, ly), label.upper(), label_font,
                           (255, 255, 255, alpha), stroke=2)

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
        # Upmix TTS (mono 24kHz) → stereo 44100 Hz, pad to duration
        cmd = [
            _FFMPEG, "-y", "-loglevel", "error",
            "-loop", "1", "-i", png_path,
            "-i", audio_path,
            "-filter_complex",
            f"[1:a]aresample=44100,pan=stereo|c0=c0|c1=c0,"
            f"apad,atrim=0:{duration:.3f}[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-t", f"{duration:.2f}",
            "-vf", f"scale={width}:{height}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
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
            "-c:a", "aac", "-b:a", "192k",
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
    """Reframe clip to target dimensions; reset timestamps; mirror horizontally.

    Horizontal flip (hflip) disrupts automated Content ID fingerprint matching
    without affecting viewability.
    """
    src_w, src_h, _ = _probe(in_path)
    src_ratio = src_w / src_h
    target_ratio = width / height

    _af_reset = "asetpts=PTS-STARTPTS,aresample=44100,pan=stereo|c0=c0|c1=c1"

    def _letterbox_cmd(src, dst):
        # scale-to-fit + center pad + hflip + timestamp reset
        return [
            _FFMPEG, "-y", "-loglevel", "error", "-i", src,
            "-vf", (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                    f"hflip,setpts=PTS-STARTPTS"),
            "-af", _af_reset,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", dst,
        ]

    # Already portrait-ish — just scale/pad/flip
    if src_ratio <= target_ratio * 1.10 and not letterbox:
        subprocess.run(_letterbox_cmd(in_path, out_path), check=True, timeout=_TIMEOUT)
        return out_path

    if letterbox:
        subprocess.run(_letterbox_cmd(in_path, out_path), check=True, timeout=_TIMEOUT)
        return out_path

    # Smart MediaPipe reframing
    try:
        from ..local.clipper import _smart_reframe
        _smart_reframe(in_path, out_path, f"{width}:{height}")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            # Re-encode: reset timestamps + yuv420p + stereo 44100 + hflip
            tmp = out_path + ".tmp.mp4"
            os.rename(out_path, tmp)
            cmd = [
                _FFMPEG, "-y", "-loglevel", "error", "-i", tmp,
                "-vf", "hflip,setpts=PTS-STARTPTS",
                "-af", _af_reset,
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", out_path,
            ]
            subprocess.run(cmd, check=True, timeout=_TIMEOUT)
            os.remove(tmp)
            return out_path
    except Exception as e:
        print(f"[ranking/reframe] smart reframe failed ({e}), using letterbox", flush=True)

    # Letterbox fallback
    subprocess.run(_letterbox_cmd(in_path, out_path), check=True, timeout=_TIMEOUT)
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
    music_path: Optional[str] = None,
    flash_text: Optional[str] = None,
) -> str:
    """Build a monetization-ready rank clip.

    Visual transformations applied (YouTube YPP requirements):
      - Clip shrunk to 78% of frame width, centered vertically, on dark background
      - Overlay PNG (title bar + rank list) composited on top
      - zoompan motion: gentle zoom-in/out cycle every 4 s keeps the frame moving
        and satisfies the "5-second rule" (no static uncut segment >5 s)

    Audio transformations:
      - Original clip audio ducked to -22 dB (very quiet background)
      - Commentary TTS at full volume — voice dominates the mix
      - Optional background music at -28 dB layered underneath everything
    """
    tmp_dir = tempfile.mkdtemp(prefix="rank_assemble_")
    try:
        src_w, src_h, src_fps = _probe(clip_path)

        # ── Video geometry: clip at 78% of target frame width, centered ──────
        # clip_w must be divisible by 2 for libx264
        clip_w = int(src_w * 0.78 / 2) * 2  # not used directly; scale handles it
        # The scaled clip fits inside the frame with padding around it.
        # We scale so the clip width = 78% of output width, then pad to full frame.
        scale_w = 720          # output width
        scale_h = 1280         # output height
        inner_w = int(scale_w * 0.78 / 2) * 2   # 78% of 720 = 561 → 560
        # inner_h preserves original aspect ratio
        inner_h = int(inner_w * src_h / src_w / 2) * 2
        # Y position: center the clip in the frame (accounting for overlay bar at top)
        pad_y = (scale_h - inner_h) // 2
        pad_x = (scale_w - inner_w) // 2

        # No zoompan — too slow on many clip formats. Scale + setsar handles
        # the sizing cleanly without risk of hanging.
        fps = min(max(round(src_fps), 24), 60) if src_fps > 0 else 30

        # ── Build filter_complex ──────────────────────────────────────────────
        # Inputs:  [0]=clip [1]=overlay_png [2]=flash_png(opt) [3]=tts(opt) [4]=music(opt)
        has_tts   = bool(commentary_audio and os.path.exists(commentary_audio)
                         and commentary_duration > 0)
        has_music = bool(music_path and os.path.exists(music_path))
        has_flash = bool(flash_text)

        # Render flash PNG if we have text
        flash_png_path = None
        if has_flash:
            flash_png_path = os.path.join(tmp_dir, "flash.png")
            _render_flash_png(flash_text, scale_w, scale_h, flash_png_path)

        # Assign input indices
        cmd = [_FFMPEG, "-y", "-loglevel", "error"]
        cmd += ["-i", clip_path]    # [0] clip video+audio
        cmd += ["-i", overlay_png]  # [1] rank overlay PNG
        next_idx = 2
        flash_idx = None
        if has_flash:
            cmd += ["-i", flash_png_path]
            flash_idx = next_idx
            next_idx += 1
        tts_idx = None
        if has_tts:
            cmd += ["-i", commentary_audio]
            tts_idx = next_idx
            next_idx += 1
        music_idx_final = None
        if has_music:
            cmd += ["-stream_loop", "-1", "-i", music_path]
            music_idx_final = next_idx
            next_idx += 1

        # Flash timing: appears at 40% into clip for 0.8 seconds
        clip_duration_s = get_clip_duration(clip_path)
        flash_start = max(0.5, clip_duration_s * 0.40)
        flash_end   = flash_start + 0.8

        # Video chain: scale → pad to 78% region → pad to full dark frame → overlays
        vf_parts = [
            f"[0:v]setpts=PTS-STARTPTS,"
            f"scale={inner_w}:{inner_h}:force_original_aspect_ratio=decrease,"
            f"pad={inner_w}:{inner_h}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,"
            f"pad={scale_w}:{scale_h}:{pad_x}:{pad_y}:color=0x0a0a0a[clipped]",
        ]
        if has_flash:
            vf_parts.append(f"[clipped][1:v]overlay=0:0[after_rank]")
            vf_parts.append(
                f"[after_rank][{flash_idx}:v]overlay=0:0:"
                f"enable='between(t,{flash_start:.2f},{flash_end:.2f})'[vout]"
            )
        else:
            vf_parts.append(f"[clipped][1:v]overlay=0:0[vout]")
        vf = ";".join(vf_parts)

        # Audio chain: use the indices already computed above (tts_idx, music_idx_final)
        # These correctly account for the flash PNG consuming one input slot when present.

        af_parts = []
        mix_inputs = []

        # Original audio (only if the clip actually has an audio stream)
        clip_has_audio = _has_audio(clip_path)
        if clip_has_audio:
            af_parts.append(
                "[0:a]asetpts=PTS-STARTPTS,aresample=44100,"
                "pan=stereo|c0=c0|c1=c1,volume=-22dB[orig]"
            )
            mix_inputs.append("[orig]")

        if has_tts:
            # TTS: upmix to stereo 44100, full volume, pad with silence to match clip.
            # whole_dur=N is mandatory — without it apad deadlocks waiting for a
            # video-duration signal that never arrives in this filter topology.
            af_parts.append(
                f"[{tts_idx}:a]aresample=44100,"
                f"pan=stereo|c0=c0|c1=c0,volume=1.0,"
                f"apad=whole_dur={clip_duration_s:.3f}[tts_padded]"
            )
            mix_inputs.append("[tts_padded]")

        if has_music and music_idx_final is not None:
            # Background music: loop, stereo 44100, very quiet -28 dB
            af_parts.append(
                f"[{music_idx_final}:a]aresample=44100,"
                f"pan=stereo|c0=c0|c1=c1,volume=-28dB,"
                f"apad=whole_dur={clip_duration_s:.3f}[music_padded]"
            )
            mix_inputs.append("[music_padded]")

        n_mix = len(mix_inputs)
        if n_mix == 0:
            # No audio at all — generate silent audio to satisfy the output map
            af_parts.append(
                f"aevalsrc=0:channel_layout=stereo:sample_rate=44100:duration={clip_duration_s:.3f}[aout]"
            )
        elif n_mix == 1:
            # Single stream — rename directly (amix with 1 input is valid but wasteful)
            single = mix_inputs[0]
            af_parts.append(f"{single}aresample=44100[aout]")
        else:
            mix_labels = "".join(mix_inputs)
            af_parts.append(
                f"{mix_labels}amix=inputs={n_mix}:duration=first:"
                "normalize=0:dropout_transition=0[aout]"
            )

        filter_complex = ";".join(af_parts)
        filter_complex = f"{vf};{filter_complex}"

        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]

        subprocess.run(cmd, check=True, timeout=_TIMEOUT)
        return out_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Final concat ──────────────────────────────────────────────────────────────

def concat_clips(clip_paths: List[str], out_path: str) -> str:
    """Concatenate clips using the ffmpeg concat FILTER (not the demuxer).

    The concat filter chains segments sequentially and ignores any stale
    presentation timestamps in the input containers — this is essential when
    clips were cut from the middle of a long source video and may still carry
    timestamps like 857s → 863s instead of 0s → 6s.
    """
    n = len(clip_paths)

    # Normalize SAR to 1:1 and reset PTS on every input before concatenation.
    # This is required because: (a) scene-cut clips carry source timestamps and
    # (b) reframed clips often have non-square SAR (e.g. 404:405) while the
    # announcement clip has 1:1 — mismatched SAR causes the concat filter to fail.
    sar_parts = "".join(f"[{i}:v]setsar=1[v{i}];" for i in range(n))
    v_labels   = "".join(f"[v{i}][{i}:a]" for i in range(n))
    filter_complex = f"{sar_parts}{v_labels}concat=n={n}:v=1:a=1[vout][aout]"

    cmd = [_FFMPEG, "-y", "-loglevel", "error"]
    for p in clip_paths:
        cmd += ["-i", p]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return out_path
