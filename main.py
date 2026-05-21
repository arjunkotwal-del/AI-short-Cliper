"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." --num-clips 5 --min-score 80
"""
import argparse
import json
import sys

from shorts_generator import generate_shorts, generate_gaming_shorts, generate_ai_short


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("url", help="YouTube video URL (or text prompt for --mode ai)")
    parser.add_argument("--mode", default="default", choices=["default", "gaming", "ai"],
                        help="Pipeline mode: 'default' (podcast/interview), 'gaming' (streamer/gameplay), or 'ai' (fully AI-generated from prompt) (default: default)")
    parser.add_argument("--voice", default="onyx", choices=["onyx", "echo", "alloy", "fable", "nova", "shimmer"],
                        help="AI mode: TTS voice (default: onyx)")
    parser.add_argument("--num-clips", type=int, default=3, help="Max shorts to render (default: 3)")
    parser.add_argument("--min-score", type=int, default=0, help="Drop clips below this score 0-100 (default: 0 = keep all)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", choices=["360", "480", "720", "1080"],
                        help="Source resolution (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code e.g. 'en' (default: auto)")
    parser.add_argument("--output-json", default=None, help="Write full result JSON to this path")
    parser.add_argument("--output-dir", default=None, help="Override output directory (default: from .env or ./output)")
    parser.add_argument("--letterbox", action="store_true", help="Disable smart speaker framing, use letterbox instead")
    parser.add_argument("--remove-silence", action="store_true", help="Remove silent gaps from clips (off by default)")
    parser.add_argument("--clip-duration", type=float, default=22.0, help="Gaming mode: clip length in seconds (default: 22)")
    parser.add_argument("--min-gap", type=float, default=60.0, help="Gaming mode: minimum seconds between peaks (default: 60)")
    args = parser.parse_args()

    try:
        if args.mode == "ai":
            result = generate_ai_short(
                prompt=args.url,  # In AI mode, "url" arg is the text prompt
                output_dir=args.output_dir,
                voice=args.voice,
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

    if mode == "ai":
        # AI mode output
        print("\n" + "=" * 72)
        print(f"Mode:    AI-generated short")
        print(f"Prompt:  {result.get('prompt', '')[:80]}")
        print(f"Title:   {result.get('title', '')}")
        print("=" * 72)
        if result.get("clip_url"):
            print(f"\nOutput:  {result['clip_url']}")
            import os
            print(f"Folder:  {os.path.abspath(os.path.dirname(result['clip_url']))}")
        else:
            print(f"\nFAILED:  {result.get('error')}")
    else:
        print("\n" + "=" * 72)
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
                print(f"\n#{i}  score={s.get('score')}{dims}  {s.get('start_time'):.1f}s -> {s.get('end_time'):.1f}s")
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
