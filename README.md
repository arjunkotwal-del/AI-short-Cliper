# AI YouTube Shorts Generator


Automatically cut any YouTube video into viral-ready short clips. 100% free, no watermarks.  
Give it a URL — it downloads, transcribes, scores every moment for virality, and renders polished vertical videos with animated word-level captions.

---

## How it works

```
YouTube URL
    |
    v
[yt-dlp]  download source video (720p mp4, cached)
    |
    v
[faster-whisper]  transcribe with word-level timestamps (cached)
    |
    v
[GPT-4o-mini]  score every segment on 4 dimensions -> pick top N
    |
    v
[ffmpeg]  cut + silence removal + letterbox + burn ASS captions
    |
    v
output/{video_id}/01_title.mp4  +  .txt social copy
```

### Virality scoring

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Hook      | 35 %   | Does the opening line stop the scroll within 3 s? |
| Flow      | 20 %   | No mid-sentence cuts, no dead air |
| Value     | 25 %   | Entertainment, info, or emotional payoff |
| Trend     | 20 %   | Cultural relevance, memes, current events |

`final score = hook*0.35 + flow*0.20 + value*0.25 + trend*0.20`

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.9+ | [python.org](https://www.python.org/) |
| ffmpeg | `winget install ffmpeg` / `brew install ffmpeg` / apt |
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

Copy `.env.example` to `.env` and add your OpenAI key:

```
OPENAI_API_KEY=sk-proj-...
```

---

## Quick start

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

That produces 3 shorts (default) in `output/{video_id}/`.

---

## CLI reference

```
python main.py URL [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num-clips N` | `3` | Max shorts to render |
| `--min-score N` | `0` | Drop clips below this score (0 = keep all) |
| `--format 360/480/720/1080` | `720` | Source download resolution |
| `--aspect-ratio W:H` | `9:16` | Output aspect ratio |
| `--language CODE` | auto | Force Whisper language (e.g. `en`, `es`) |
| `--output-json PATH` | — | Write full result JSON to this file |

### Examples

```bash
# 5 clips from a commentary video, drop anything under 80
python main.py "https://youtu.be/abc123" --num-clips 5 --min-score 80

# 1080p source for higher quality
python main.py "https://youtu.be/abc123" --format 1080 --num-clips 3

# Force English transcription
python main.py "https://youtu.be/abc123" --language en
```

---

## Output structure

```
output/
  source_lWYcK2YBAh4.mp4              <- cached source (not re-downloaded)
  source_lWYcK2YBAh4_transcript.json  <- cached transcript
  lWYcK2YBAh4/
    01_cars_a_weird_franchise.mp4     <- vertical short with captions
    01_cars_a_weird_franchise.txt     <- TikTok/IG caption + hashtags
    02_mater_the_reality_warper.mp4
    02_mater_the_reality_warper.txt
    ...
```

Each `.txt` sidecar contains a punchy 150-char caption and 8 hashtags ready to paste.

---

## Environment variables

Set in `.env` (never commit this file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for highlight scoring |
| `LOCAL_WHISPER_MODEL` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `LOCAL_WHISPER_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `LOCAL_OUTPUT_DIR` | `output` | Root folder for all output |

---

## GPU acceleration (faster transcription)

Install PyTorch with CUDA, then use a larger Whisper model:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

```
# .env
LOCAL_WHISPER_MODEL=large-v3
LOCAL_WHISPER_DEVICE=cuda
```

`large-v3` on GPU runs ~10x faster than `base` on CPU and noticeably more accurate on accents and fast speech.

---

## Project structure

```
main.py                          CLI entry point
shorts_generator/
  pipeline.py                    End-to-end orchestrator
  highlights.py                  LLM virality scoring & social copy
  config.py                      Env-var loading
  local/
    downloader.py                yt-dlp wrapper with caching
    transcriber.py               faster-whisper wrapper with caching
    clipper.py                   ffmpeg cutting, letterbox, captions
    llm.py                       OpenAI client wrapper
requirements.txt
.env.example
```

---

## License

MIT License
