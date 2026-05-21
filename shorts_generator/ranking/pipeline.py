"""Ranking video pipeline.

Flow:
  1. GPT generates a 2-4 word label for every rank
  2. Render title card PNG; TTS reads the title aloud over it
  3. For each clip (rank N down to 1):
       a. Reframe to 9:16
       b. Render overlay PNG  (top title bar + left rank list)
       c. TTS reads the clip's label (e.g. "Funny Goof")
       d. Assemble: overlay PNG + label TTS ducked over original audio
  4. Concatenate title card + all rank clips -> final output
"""
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional

from ..config import LOCAL_OUTPUT_DIR
from .commentator import generate_clip_names, synthesize_tts, get_audio_duration
from .compositor import (
    render_ranking_overlay,
    render_title_card_png,
    make_title_card_video,
    reframe_clip,
    assemble_rank_clip,
    concat_clips,
)

OUTPUT_WIDTH  = 720
OUTPUT_HEIGHT = 1280


def generate_ranking_shorts(
    title: str,
    clip_paths: List[str],
    aspect_ratio: str = "9:16",
    output_dir: Optional[str] = None,
    letterbox: bool = False,
) -> dict:
    """Main entry point for --mode ranking.

    clip_paths must be ordered rank-N (least extreme) to rank-1 (most extreme).
    The title string is also used as the spoken intro line.
    """
    if not clip_paths:
        raise ValueError("No clips provided. Use --clips path1.mp4 path2.mp4 ...")

    total = len(clip_paths)

    try:
        aw, ah = aspect_ratio.split(":")
        out_w = OUTPUT_WIDTH
        out_h = int(out_w * float(ah) / float(aw))
    except Exception:
        out_w, out_h = OUTPUT_WIDTH, OUTPUT_HEIGHT

    base_out = output_dir or LOCAL_OUTPUT_DIR
    timestamp = int(time.time())
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)[:40].strip()
    run_dir = os.path.join(base_out, f"ranking_{safe_title}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    tmp_root = tempfile.mkdtemp(prefix="ranking_pipeline_")
    try:
        assembled: List[str] = []
        shorts_meta: List[dict] = []

        # ── Step 1: GPT generates short labels for every rank ─────────────────
        print("[ranking] generating clip labels...", flush=True)
        clip_names: Dict[int, str] = generate_clip_names(title, total)
        for r in range(1, total + 1):
            if r not in clip_names:
                clip_names[r] = f"RANK {r}"
        for r, name in sorted(clip_names.items()):
            print(f"[ranking]   rank #{r}: {name}", flush=True)

        # ── Step 2: Title card — PNG + TTS of the title ───────────────────────
        print(f"[ranking] rendering title card...", flush=True)
        title_png = os.path.join(tmp_root, "title_card.png")
        render_title_card_png(title, out_w, out_h, title_png)

        print(f"[ranking] synthesizing title TTS: \"{title}\"", flush=True)
        title_mp3 = os.path.join(tmp_root, "title_tts.mp3")
        synthesize_tts(title, title_mp3)
        title_tts_dur = get_audio_duration(title_mp3)
        # Give ~0.5 s of silence after the voice finishes
        title_card_dur = max(3.0, title_tts_dur + 0.5)

        title_mp4 = os.path.join(tmp_root, "title_card.mp4")
        make_title_card_video(
            title_png, title_card_dur, out_w, out_h, title_mp4,
            audio_path=title_mp3,
        )
        assembled.append(title_mp4)

        # ── Step 3: Per-rank clips ────────────────────────────────────────────
        # clips[0] = rank N (least extreme), clips[-1] = rank 1 (most extreme)
        for idx, clip_path in enumerate(clip_paths):
            rank = total - idx          # counts down: N, N-1, …, 1
            label = clip_names.get(rank, f"RANK {rank}")
            print(f"\n[ranking] rank #{rank} ({label}) — {os.path.basename(clip_path)}", flush=True)

            clip_tmp = os.path.join(tmp_root, f"rank{rank:02d}")
            os.makedirs(clip_tmp, exist_ok=True)

            # 3a: Reframe to 9:16
            print(f"[ranking] reframing...", flush=True)
            reframed = os.path.join(clip_tmp, "reframed.mp4")
            reframe_clip(clip_path, reframed, out_w, out_h, letterbox=letterbox)

            # 3b: Render overlay PNG (title bar + rank list)
            print(f"[ranking] rendering overlay...", flush=True)
            overlay_png = os.path.join(clip_tmp, "overlay.png")
            render_ranking_overlay(
                rank, total, out_w, out_h, overlay_png,
                clip_names=clip_names,
                title=title,
            )

            # 3c: TTS reads the clip label (e.g. "Funny Goof")
            print(f"[ranking] synthesizing label TTS: \"{label}\"", flush=True)
            label_mp3 = os.path.join(clip_tmp, "label_tts.mp3")
            try:
                synthesize_tts(label, label_mp3)
                label_dur = get_audio_duration(label_mp3)
                print(f"[ranking]   -> {label_dur:.1f}s", flush=True)
            except Exception as e:
                print(f"[ranking] TTS failed: {e}", flush=True)
                label_mp3 = None
                label_dur = 0.0

            # 3d: Assemble
            print(f"[ranking] assembling...", flush=True)
            assembled_clip = os.path.join(run_dir, f"rank{rank:02d}.mp4")
            assemble_rank_clip(
                clip_path=reframed,
                rank=rank,
                total_ranks=total,
                overlay_png=overlay_png,
                commentary_audio=label_mp3,
                commentary_duration=label_dur,
                out_path=assembled_clip,
            )
            assembled.append(assembled_clip)

            shorts_meta.append({
                "rank": rank,
                "label": label,
                "source_clip": clip_path,
                "clip_url": assembled_clip,
                "label_duration": label_dur,
            })
            print(f"[ranking] rank #{rank} done -> {assembled_clip}", flush=True)

        # ── Step 4: Concat ────────────────────────────────────────────────────
        print(f"\n[ranking] concatenating {len(assembled)} segments...", flush=True)
        final_path = os.path.join(run_dir, "ranking_final.mp4")
        concat_clips(assembled, final_path)
        print(f"[ranking] final video -> {final_path}", flush=True)

        return {
            "title": title,
            "mode": "ranking",
            "output_path": final_path,
            "output_dir": run_dir,
            "shorts": shorts_meta,
            "highlights": [{"title": f"Rank #{m['rank']}: {m['label']}", **m}
                           for m in shorts_meta],
            "source_video_url": title,
        }

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
