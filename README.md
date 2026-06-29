# AI YouTube Shorts Generator

Automatically turn any YouTube video into viral-ready vertical clips. 100% local, no watermarks, no subscriptions.

---

## Modes

### 1. `default` — Auto-clip any YouTube video

Downloads, transcribes, and scores every segment for virality. Picks the best moments and crops them to 9:16 with smart speaker tracking.

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Pipeline:**
```
YouTube URL
    │
    ▼
[yt-dlp]          download source video (720p mp4, cached)
    │
    ▼
[faster-whisper]  transcribe with word-level timestamps (cached)
    │
    ▼
[GPT-4o-mini]     score every segment on 8 virality signals → pick top N
    │
    ▼
[MediaPipe]       detect faces per second → build crop keyframe timeline
    │
    ▼
[ffmpeg]          cut + dynamic speaker crop + encode 9:16 MP4
    │
    ▼
output/{video_id}/01_title.mp4
```

**Virality scoring:**

| Signal | What it measures |
|--------|-----------------|
| Hook | Does the first line stop the scroll within 3 s? |
| Emotional peak | Anger, awe, laughter, shock |
| Opinion bomb | Controversial take that triggers replies |
| Revelation | Surprising fact or plot twist |
| Conflict | Confrontation or tension |
| Quotable | Single sentence you'd screenshot |
| Story peak | Narrative climax |
| Practical value | Tip, hack, or advice people will save |

**Smart speaker framing** (on by default): MediaPipe samples 1 frame/second, detects faces, builds a smooth crop keyframe timeline, and dynamically pans the 9:16 window to follow whoever is talking. Pass `--letterbox` to use static scale-to-fit instead.

---

### 2. `gaming` — Gaming highlight clips

Finds the loudest, most intense moments in gaming footage by audio peak detection.

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode gaming --num-clips 5
```

**Pipeline:**
```
YouTube URL  →  [yt-dlp]  →  [ffmpeg audio analysis]  →  detect dB peaks
    →  cut N-second clips around each peak  →  AI hook text  →  9:16 MP4s
```

- Clips are centered on each audio spike (kills, clutch moments, crowd reactions)
- Minimum gap between peaks prevents overlapping clips
- GPT writes a hook text label for each clip

---

### 3. `agentic` — Natural language agentic pipeline

Uses an Orchestrator Agent to interpret natural language requests and dynamically coordinate sub-agents:

```
                  [User Request]
                        │
                        ▼
                 [Orchestrator]
            ┌───────────┼───────────┐
            ▼           ▼           ▼
      [Downloader] [Transcriber] [Clipper]
```

- **Downloader Agent**: Downloads YouTube videos.
- **Transcriber Agent**: Transcribes local video files.
- **Clipper Agent**: Clips specific timestamps or extracts viral highlights.

#### Modes of operation

* **Interactive Chat Mode**:
  Start a terminal conversation with the Orchestrator to step-by-step coordinate actions:
  ```bash
  python agentic_main.py
  ```

* **Direct Prompt Execution**:
  Pass a single command prompt directly via arguments:
  ```bash
  python agentic_main.py "make 3 clips for https://www.youtube.com/watch?v=VIDEO_ID"
  ```

* **Interactive Web Dashboard**:
  Run a visual web interface to stream agent logs live and easily configure clip parameters:
  ```bash
  python web_app.py
  ```

#### Advanced Capabilities

* **Natural Language Configuration**:
  The orchestrator and sub-agents parse custom settings directly from your prompt (e.g., `"extract 3 clips with min duration 15s, max duration 40s, and voiceover disabled from https://youtube.com/watch?v=VIDEO_ID"`).
* **Local Video Support**:
  You can bypass downloading completely by passing local file paths directly to the agent:
  ```bash
  # Transcribe and find viral highlights locally
  python agentic_main.py "extract 3 highlights from C:\path\to\video.mp4 with voiceover disabled"

  # Clip specific timestamps from a local file
  python agentic_main.py "clip from 30 to 60 seconds of C:\path\to\video.mp4"
  ```
* **Cancellation**:
  Running pipeline processes can be instantly stopped via the **Stop Pipeline** button on the web dashboard or by standard interrupt signals, safely halting all sub-agents and ffmpeg subprocesses.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.9+ | [python.org](https://www.python.org/) |
| ffmpeg | `winget install ffmpeg` / `brew install ffmpeg` / `apt install ffmpeg` |
| Git | for cloning |

---

## Installation

```bash
git clone https://github.com/arjunkotwal-del/AI-short-generator.git
cd AI-short-generator

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key:

```env
OPENAI_API_KEY=sk-proj-...
LOCAL_OUTPUT_DIR=/path/to/your/output/folder
```

---

## Flask Web Dashboard

You can control the agentic clipping pipeline through a Flask-based web dashboard featuring a dark-glass UI and live log streaming.

### Running the Dashboard

1. **Activate your virtual environment** (if not already active):
   ```bash
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```
2. **Start the Flask server**:
   ```bash
   python web_app.py
   ```
3. **Access the interface**:
   Open your browser and navigate to:
   👉 **http://localhost:5000**

### Interactive Options
- **Clips Count**: Specify the maximum number of viral shorts to render.
- **Min Duration (s)**: Set the minimum length of each clip (defaults to `10` seconds). Clips shorter than this are padded.
- **Max Duration (s)**: Set the maximum length of each clip (defaults to `60` seconds). Clips longer than this are truncated.
- **Enable TTS Voiceover**: Toggle to enable or disable the OpenAI Text-to-Speech audio overlay.
- **Stop Pipeline**: Terminate a running pipeline execution instantly.

---

## CLI reference

```bash
python main.py URL [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `default` | `default` or `gaming` |
| `--num-clips N` | `3` | Max shorts to render |
| `--min-score N` | `0` | Drop clips scoring below N (0–100) |
| `--format 360/480/720/1080` | `720` | Source download resolution |
| `--aspect-ratio W:H` | `9:16` | Output aspect ratio |
| `--language CODE` | auto | Force Whisper language (e.g. `en`, `es`) |
| `--letterbox` | off | Static scale-to-fit instead of smart speaker crop |
| `--remove-silence` | off | Strip silent gaps from clips |
| `--output-dir PATH` | from `.env` | Override output folder |
| `--output-json PATH` | — | Dump full result metadata to JSON |

### Gaming mode extras

| Flag | Default | Description |
|------|---------|-------------|
| `--clip-duration N` | `22` | Seconds per gaming clip |
| `--min-gap N` | `60` | Minimum seconds between audio peaks |

---

## Examples

```bash
# Auto-clip a podcast — top 5 moments, drop anything under score 75
python main.py "https://youtu.be/abc123" --num-clips 5 --min-score 75

# Gaming clips — 30s each, at least 90s apart
python main.py "https://youtu.be/abc123" --mode gaming --clip-duration 30 --min-gap 90

# Force English transcription
python main.py "https://youtu.be/abc123" --language en

# Letterbox (no face tracking) for screen recordings
python main.py "https://youtu.be/abc123" --letterbox
```

---

## Output structure

```
shorts-output/
  source_{id}.mp4                  ← cached (not re-downloaded)
  source_{id}_transcript.json      ← cached (not re-transcribed)
  {id}/
    01_hook_title.mp4
    02_another_moment.mp4
    ...
  {id}_gaming/
    01_watch_this_insane_clip.mp4
    ...
```

---

## Environment variables

Set in `.env` (never commit this file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for virality scoring |
| `LOCAL_WHISPER_MODEL` | `base` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `LOCAL_WHISPER_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `LOCAL_OUTPUT_DIR` | `output` | Root folder for all generated files |

---

## GPU acceleration (faster transcription)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

```env
LOCAL_WHISPER_MODEL=large-v3
LOCAL_WHISPER_DEVICE=cuda
```

`large-v3` on a GPU runs ~10× faster than `base` on CPU with noticeably better accuracy on accents and fast speech.

---

## Project structure

```
main.py                            Default/gaming CLI entry point
agentic_main.py                    Agentic pipeline CLI / interactive entry point
web_app.py                         Flask web dashboard server
agent.py                           Swarm agentic loop definition
requirements.txt                   Python dependencies
.env.example                       Example environment configuration
static/                            Dashboard UI static assets (HTML/CSS/JS)
shorts_generator/
  __init__.py                      Public API exports
  pipeline.py                      Default mode orchestrator
  highlights.py                    GPT virality scoring + social copy
  config.py                        Env-var loading
  cancel_token.py                  Pipeline cancellation state utility
  local/
    downloader.py                  yt-dlp wrapper with caching
    transcriber.py                 faster-whisper wrapper with caching
    clipper.py                     ffmpeg cutting, smart reframe (MediaPipe)
    llm.py                         OpenAI client wrapper
  gaming/
    pipeline.py                    Gaming mode orchestrator
    audio_peaks.py                 dB peak detection
    narrator.py                    GPT hook text generator
    assembler.py                   ffmpeg clip assembler
```

---

## License

MIT License
