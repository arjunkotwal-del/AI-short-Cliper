"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." --num-clips 5 --min-score 80
    python main.py "https://..." --mode gaming --num-clips 3
    python main.py "Ranking Craziest Construction Fails" --mode ranking --clips clip1.mp4 clip2.mp4 clip3.mp4
"""
import argparse
import json
import sys

from shorts_generator import generate_shorts, generate_gaming_shorts, generate_ranking_shorts


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("url",
                        help="YouTube video URL (or ranking title when --mode ranking)")
    parser.add_argument("--mode", default="default",
                        choices=["default", "gaming", "ranking"],
                        help="Pipeline mode (default: default)")
    parser.add_argument("--clips", nargs="+", metavar="CLIP",
                        help="[ranking mode] Local clip paths ordered rank-N to rank-1 "
                             "(first clip = least extreme, last = most extreme)")
    parser.add_argument("--num-clips", type=int, default=3,
                        help="Max shorts to render (default: 3)")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Drop clips below this score 0-100 (default: 0 = keep all)")
    parser.add_argument("--aspect-ratio", default="9:16",
                        help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720",
                        choices=["360", "480", "720", "1080"],
                        help="Source resolution (default: 720)")
    parser.add_argument("--language", default=None,
                        help="Force Whisper language code e.g. 'en' (default: auto)")
    parser.add_argument("--output-json", default=None,
                        help="Write full result JSON to this path")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: from .env or ./output)")
    parser.add_argument("--letterbox", action="store_true",
                        help="Disable smart speaker framing, use letterbox instead")
    parser.add_argument("--remove-silence", action="store_true",
                        help="Remove silent gaps from clips (off by default)")
    parser.add_argument("--clip-duration", type=float, default=22.0,
                        help="Gaming mode: clip length in seconds (default: 22)")
    parser.add_argument("--min-gap", type=float, default=60.0,
                        help="Gaming mode: minimum seconds between peaks (default: 60)")
    args = parser.parse_args()

    try:
        if args.mode == "ranking":
            if not args.clips:
                print("ERROR: --mode ranking requires --clips path1.mp4 path2.mp4 ...",
                      file=sys.stderr)
                return 1
            result = generate_ranking_shorts(
                title=args.url,
                clip_paths=args.clips,
                aspect_ratio=args.aspect_ratio,
                output_dir=args.output_dir,
                letterbox=args.letterbox,
            )
        elif args.mode == "gaming":
            result = generate_gaming_shorts(
                youtube_url=args.url,
                num_clips=args.num_clips,
                aspect_ratio=args.aspect_ratio,
                download_format=args.format,
                language=args.language,
                output_dir=args.output_dir,
                clip_duration=args.clip_duration,
                min_gap=args.min_gap,
            )
        else:
            result = generate_shorts(
                youtube_url=args.url,
                num_clips=args.num_clips,
                aspect_ratio=args.aspect_ratio,
                download_format=args.format,
                language=args.language,
                min_score=args.min_score,
                output_dir=args.output_dir,
                remove_silence=args.remove_silence,
                letterbox=args.letterbox,
            )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    mode = result.get("mode", "default")
    print("\n" + "=" * 72)

    if mode == "ranking":
        print(f"Ranking title:  {result['title']}")
        print(f"Clips:          {len(result['shorts'])} ranked clips")
        print(f"Output:         {result['output_path']}")
        print("=" * 72)
        for s in result["shorts"]:
            print(f"\n  Rank #{s['rank']}  ->  {s['clip_url']}")
            print(f"    commentary: {s.get('commentary', '')}")
    else:
        print(f"Source video:  {result['source_video_url']}")
        print(f"Mode:          {mode}")
        print(f"Highlights:    {len(result['highlights'])} candidates -> rendered {len(result['shorts'])}")
        print("=" * 72)
        for i, s in enumerate(result["shorts"], 1):
            if mode == "gaming":
                peak_info = ""
                if s.get("peak_db") is not None:
                    peak_info = f"  [{s['peak_db']:.1f} dB]"
                print(f"\n#{i}{peak_info}  {s.get('start_time'):.1f}s -> {s.get('end_time'):.1f}s")
                print(f"     hook:   {s.get('hook_text') or s.get('title')}")
            else:
                dims = ""
                if s.get("hook_score") is not None:
                    dims = f"  [H={s['hook_score']} F={s['flow_score']} V={s['value_score']} T={s['trend_score']}]"
                print(f"\n#{i}  score={s.get('score')}{dims}  "
                      f"{s.get('start_time'):.1f}s -> {s.get('end_time'):.1f}s")
                print(f"     title:  {s.get('title')}")
                print(f"     hook:   {s.get('hook_sentence')}")
            if s.get("clip_url"):
                print(f"     clip:   {s['clip_url']}")
            else:
                print(f"     clip:   FAILED ({s.get('error')})")

        # Print the output folder path so it's easy to find
        if result["shorts"]:
            first_clip = next((s["clip_url"] for s in result["shorts"] if s.get("clip_url")), None)
            if first_clip:
                import os
                print(f"\nOutput folder: {os.path.abspath(os.path.dirname(first_clip))}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
