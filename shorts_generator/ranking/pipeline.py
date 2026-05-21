"""Ranking video pipeline.

Flow:
  1. Parse title + clip paths (rank 5 → rank 1 order; clips[0] = least extreme)
  2. Render title card PNG → 3-second silent intro video
  3. For each clip (rank N down to 1):
       a. Reframe to 9:16
       b. Render ranking overlay PNG (stacked numbers, current rank glowing)
       c. Generate GPT commentary + TTS audio
       d. Assemble: overlay PNG + TTS ducked over original audio
  4. Concatenate title card + all rank clips → final output
"""
import os
import shutil
import tempfile
import time
from typing import List, Optional

from ..config import LOCAL_OUTPUT_DIR
from .commentator import create_rank_commentary
from .compositor import (
    render_ranking_overlay,
    render_title_card_png,
    make_title_card_video,
    reframe_clip,
    assemble_rank_clip,
    concat_clips,
)

# Output dimensions
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280


def generate_ranking_shorts(
    title: str,
    clip_paths: List[str],
    aspect_ratio: str = "9:16",
    output_dir: Optional[str] = None,
    letterbox: bool = False,
) -> dict:
    """Main entry point for --mode ranking.

    Args:
        title:       Title/topic of the ranking (e.g. "Ranking Craziest Construction Fails")
        clip_paths:  Ordered list of clip paths — clips[0] = rank N (least extreme),
                     clips[-1] = rank 1 (most extreme).
        aspect_ratio: Target aspect ratio string (default "9:16").
        output_dir:  Output directory override.
        letterbox:   Force letterbox reframing instead of smart face tracking.

    Returns dict with keys: title, shorts (list of per-clip dicts), output_path.
    """
    if not clip_paths:
        raise ValueError("No clips provided. Use --clips path1.mp4 path2.mp4 ...")

    total = len(clip_paths)

    # Parse output dimensions from aspect_ratio
    try:
        aw, ah = aspect_ratio.split(":")
        ar = float(aw) / float(ah)
        out_w = OUTPUT_WIDTH
        out_h = int(out_w / ar)
    except Exception:
        out_w, out_h = OUTPUT_WIDTH, OUTPUT_HEIGHT

    # Resolve output directory
    base_out = output_dir or LOCAL_OUTPUT_DIR
    timestamp = int(time.time())
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)[:40].strip()
    run_dir = os.path.join(base_out, f"ranking_{safe_title}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    tmp_root = tempfile.mkdtemp(prefix="ranking_pipeline_")
    try:
        assembled: List[str] = []  # final per-clip MP4 paths in order
        shorts_meta: List[dict] = []

        # ── Title card ────────────────────────────────────────────────────────
        print(f"[ranking] rendering title card for \"{title}\"", flush=True)
        title_png = os.path.join(tmp_root, "title_card.png")
        title_mp4 = os.path.join(tmp_root, "title_card.mp4")
        render_title_card_png(title, out_w, out_h, title_png)
        make_title_card_video(title_png, duration=3.0, width=out_w, height=out_h,
                              out_path=title_mp4)
        assembled.append(title_mp4)

        # ── Per-rank clip processing ──────────────────────────────────────────
        # clips[0] is rank N (least extreme), clips[-1] is rank 1
        # We reveal them in order: rank N first, rank 1 last
        for idx, clip_path in enumerate(clip_paths):
            rank = total - idx  # rank N down to 1
            print(f"\n[ranking] processing rank #{rank} — {os.path.basename(clip_path)}", flush=True)

            clip_tmp = os.path.join(tmp_root, f"rank{rank:02d}")
            os.makedirs(clip_tmp, exist_ok=True)

            # Step 1: Reframe to 9:16
            print(f"[ranking] reframing clip to {out_w}x{out_h}...", flush=True)
            reframed = os.path.join(clip_tmp, "reframed.mp4")
            reframe_clip(clip_path, reframed, out_w, out_h, letterbox=letterbox)

            # Step 2: Render ranking overlay PNG
            print(f"[ranking] rendering overlay (rank {rank}/{total})...", flush=True)
            overlay_png = os.path.join(clip_tmp, "overlay.png")
            render_ranking_overlay(rank, total, out_w, out_h, overlay_png)

            # Step 3: Generate commentary + TTS
            print(f"[ranking] generating commentary TTS...", flush=True)
            commentary = create_rank_commentary(rank, total, title, clip_tmp)

            # Step 4: Assemble
            print(f"[ranking] assembling clip...", flush=True)
            assembled_clip = os.path.join(run_dir, f"rank{rank:02d}.mp4")
            assemble_rank_clip(
                clip_path=reframed,
                rank=rank,
                total_ranks=total,
                overlay_png=overlay_png,
                commentary_audio=commentary["audio_path"] if commentary else None,
                commentary_duration=commentary["duration"] if commentary else 0.0,
                out_path=assembled_clip,
            )
            assembled.append(assembled_clip)

            meta = {
                "rank": rank,
                "source_clip": clip_path,
                "clip_url": assembled_clip,
                "commentary": commentary["text"] if commentary else "",
                "commentary_duration": commentary["duration"] if commentary else 0.0,
            }
            shorts_meta.append(meta)
            print(f"[ranking] ✓ rank #{rank} done → {assembled_clip}", flush=True)

        # ── Concatenate all clips ─────────────────────────────────────────────
        print(f"\n[ranking] concatenating {len(assembled)} segments...", flush=True)
        final_path = os.path.join(run_dir, "ranking_final.mp4")
        concat_clips(assembled, final_path)
        print(f"[ranking] ✓ final video → {final_path}", flush=True)

        return {
            "title": title,
            "mode": "ranking",
            "output_path": final_path,
            "output_dir": run_dir,
            "shorts": shorts_meta,
            # Compatibility with main.py display loop
            "highlights": [{"title": f"Rank #{m['rank']}", **m} for m in shorts_meta],
            "source_video_url": title,
        }

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
