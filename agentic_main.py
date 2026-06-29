import argparse
import sys
import os

from agent import Agent, run_agent_loop
from shorts_generator.local.downloader import download_youtube_local
from shorts_generator.local.transcriber import transcribe_local
from shorts_generator.local.clipper import crop_clip_local, crop_highlights_local
from shorts_generator.highlights import get_highlights, call_openai_llm
from shorts_generator.cancel_token import check_cancelled

# Pre-declare agents to allow circular references in handoffs
orchestrator_agent = Agent("Orchestrator", "")
downloader_agent = Agent("Downloader", "")
transcriber_agent = Agent("Transcriber", "")
clipper_agent = Agent("Clipper", "")

# ---------------------------------------------------------------------------
# Downloader Tools
# ---------------------------------------------------------------------------
def download_video(url: str, format: str = "720") -> str:
    """Download a YouTube video. Returns the local file path to the mp4."""
    try:
        return download_youtube_local(url, fmt=format)
    except Exception as e:
        return f"Download failed: {str(e)}"

def transfer_to_orchestrator(**kwargs) -> Agent:
    """Return control back to the orchestrator when finished downloading."""
    return orchestrator_agent

downloader_agent.instructions = (
    "You are the Downloader Agent. Your only job is to download YouTube videos.\n"
    "When a video needs to be downloaded, call download_video(url). Once you have the path, "
    "transfer control back to the Orchestrator with the path so it can decide the next step."
)
downloader_agent.tools = [download_video, transfer_to_orchestrator]
downloader_agent.tool_map = {t.__name__: t for t in downloader_agent.tools}

# ---------------------------------------------------------------------------
# Transcriber Tools
# ---------------------------------------------------------------------------
def transcribe_video(file_path: str, language: str = None) -> str:
    """Transcribe a local video file. Returns a summary of the transcript and saves it to disk."""
    try:
        # returns dict with duration, segments, words
        res = transcribe_local(file_path, language=language)
        return f"Success! Transcribed {len(res['segments'])} segments. Transcript cached on disk."
    except Exception as e:
        return f"Transcription failed: {str(e)}"

transcriber_agent.instructions = (
    "You are the Transcriber Agent. Your job is to extract transcripts from downloaded videos.\n"
    "When asked, call transcribe_video(file_path). It will cache the result to disk.\n"
    "When finished, transfer control back to the Orchestrator."
)
transcriber_agent.tools = [transcribe_video, transfer_to_orchestrator]
transcriber_agent.tool_map = {t.__name__: t for t in transcriber_agent.tools}

# ---------------------------------------------------------------------------
def _adjust_times(start: float, end: float, min_dur: float, max_dur: float, video_dur: float):
    dur = end - start
    if dur < min_dur:
        pad = min_dur - dur
        start = max(0.0, start - pad / 2.0)
        end = min(video_dur, start + min_dur)
        if end == video_dur:
            start = max(0.0, end - min_dur)
    elif dur > max_dur:
        end = start + max_dur
    return round(start, 3), round(end, 3)

# Clipper Tools
# ---------------------------------------------------------------------------
def clip_specific_times(
    source_path: str,
    start_time: float,
    end_time: float,
    min_duration: float = 10.0,
    max_duration: float = 60.0,
    voiceover: bool = False,
) -> str:
    """Clip a specific time range from a video file."""
    try:
        check_cancelled()
        import re
        import json
        filename = os.path.splitext(os.path.basename(source_path))[0]
        match = re.search(r"source_([A-Za-z0-9_-]{11})", filename)
        if match:
            vid_id = match.group(1)
        else:
            vid_id = filename[7:] if filename.startswith("source_") else filename

        cache_path = os.path.splitext(source_path)[0] + "_transcript.json"
        video_duration = 999999.0
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                t = json.load(f)
                video_duration = float(t.get("duration", 999999.0))

        start_time, end_time = _adjust_times(start_time, end_time, min_duration, max_duration, video_duration)

        out_dir = os.path.join("output", vid_id, "clips")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"clip_{int(start_time)}_{int(end_time)}.mp4")
        
        crop_clip_local(
            source_path=source_path,
            start_time=start_time,
            end_time=end_time,
            aspect_ratio="9:16",
            out_path=out_path,
            voiceover=voiceover
        )
        return f"Successfully clipped! Saved to {out_path}"
    except Exception as e:
        return f"Clipping failed: {str(e)}"

def extract_viral_highlights(
    source_path: str,
    num_clips: int = 3,
    min_duration: float = 10.0,
    max_duration: float = 60.0,
    voiceover: bool = False,
) -> str:
    """Extract viral highlights from a video using its cached transcript."""
    try:
        check_cancelled()
        # Load transcript from cache
        cache_path = os.path.splitext(source_path)[0] + "_transcript.json"
        if not os.path.exists(cache_path):
            return "Error: Transcript not found. The video must be transcribed first."
            
        import json
        with open(cache_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
            
        print("[Clipper] Finding viral moments...", flush=True)
        h_res = get_highlights(transcript, num_clips=num_clips, llm_fn=call_openai_llm)
        highlights = h_res.get("highlights", [])
        
        if not highlights:
            return "No highlights found."
            
        # Sort and cap highlights to requested num_clips
        highlights = sorted(highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
            
        import re
        filename = os.path.splitext(os.path.basename(source_path))[0]
        match = re.search(r"source_([A-Za-z0-9_-]{11})", filename)
        if match:
            vid_id = match.group(1)
        else:
            vid_id = filename[7:] if filename.startswith("source_") else filename

        # Adjust start and end times to fit within min_duration/max_duration
        adjusted_highlights = []
        video_duration = float(transcript.get("duration", 999999.0))
        for h in highlights:
            s, e = _adjust_times(float(h["start_time"]), float(h["end_time"]), min_duration, max_duration, video_duration)
            adjusted_highlights.append({**h, "start_time": s, "end_time": e})

        out_dir = os.path.join("output", vid_id, "highlights")
        results = crop_highlights_local(
            source_path=source_path,
            highlights=adjusted_highlights,
            words=transcript.get("words"),
            out_dir=out_dir,
            voiceover=voiceover
        )
        
        successes = [r for r in results if r.get("clip_url")]
        return f"Successfully extracted {len(successes)} highlights to {out_dir}"
    except Exception as e:
        return f"Highlight extraction failed: {str(e)}"

clipper_agent.instructions = (
    "You are the Clipper Agent. Your job is to clip videos.\n"
    "You have two modes:\n"
    "1. clip_specific_times: Use this if the user asks for specific timestamps (e.g., 'last 30 seconds').\n"
    "2. extract_viral_highlights: Use this if the user asks for 'the funniest part', 'viral moments', etc.\n"
    "Once done, transfer control back to the Orchestrator."
)
clipper_agent.tools = [clip_specific_times, extract_viral_highlights, transfer_to_orchestrator]
clipper_agent.tool_map = {t.__name__: t for t in clipper_agent.tools}

# ---------------------------------------------------------------------------
# Orchestrator Tools
# ---------------------------------------------------------------------------
def transfer_to_downloader(**kwargs) -> Agent:
    """Transfer control to the Downloader Agent."""
    return downloader_agent

def transfer_to_transcriber(**kwargs) -> Agent:
    """Transfer control to the Transcriber Agent."""
    return transcriber_agent

def transfer_to_clipper(**kwargs) -> Agent:
    """Transfer control to the Clipper Agent."""
    return clipper_agent

def finish_task(message: str) -> str:
    """Call this when the user's request has been completely fulfilled."""
    return f"TASK_COMPLETE: {message}"

orchestrator_agent.instructions = (
    "You are the Orchestrator Agent for an AI Shorts Generator pipeline.\n"
    "You communicate with the user, understand their request, and delegate tasks to sub-agents.\n"
    "Available sub-agents:\n"
    "- Downloader Agent: Downloads YouTube videos. (Needs a URL).\n"
    "- Transcriber Agent: Transcribes local video files. (Needs a local file path).\n"
    "- Clipper Agent: Clips videos by specific times or finds viral highlights. (Needs a local file path).\n"
    "Coordinate the workflow. If a URL is provided, send it to the Downloader first.\n"
    "If they ask for viral highlights, you must ensure the video is downloaded AND transcribed before sending it to the Clipper.\n"
    "If they ask for specific timestamps (e.g. 'last 30 seconds'), transcription is optional, just download and pass to Clipper.\n"
    "When the workflow is entirely done, call finish_task to inform the user."
)
orchestrator_agent.tools = [
    transfer_to_downloader, 
    transfer_to_transcriber, 
    transfer_to_clipper, 
    finish_task
]
orchestrator_agent.tool_map = {t.__name__: t for t in orchestrator_agent.tools}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Agentic Shorts Generator")
    parser.add_argument("prompt", nargs="*", help="Natural language request")
    args = parser.parse_args()

    # Load environment / configure output
    from shorts_generator.config import require_openai_key
    require_openai_key() # ensure API key is set

    # Start loop
    messages = []
    
    # If prompt passed in CLI args
    if args.prompt:
        user_input = " ".join(args.prompt)
        print(f"\nUser: {user_input}")
        messages.append({"role": "user", "content": user_input})
        
        result = run_agent_loop(orchestrator_agent, messages)
        print(f"\nOrchestrator: {result['response']}")
        sys.exit(0)

    # Otherwise interactive mode
    print("Welcome to the Agentic Shorts Generator. Type 'exit' to quit.")
    current_agent = orchestrator_agent
    
    while True:
        try:
            user_input = input(f"\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            messages.append({"role": "user", "content": user_input})
            result = run_agent_loop(current_agent, messages)
            
            # The agent might have switched, keep track
            current_agent = result["agent"]
            messages = result["messages"]
            
            print(f"\n{current_agent.name}: {result['response']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
