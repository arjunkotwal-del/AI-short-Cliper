import os

from dotenv import load_dotenv

load_dotenv(override=True)  # .env always wins over system environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto")  # auto, cpu, or cuda


LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "output")

# JavaScript runtimes yt-dlp may use to get past YouTube's download protection.
# At least one must be installed (deno, node, bun or quickjs); without it YouTube answers 403.
YTDLP_JS_RUNTIMES = [r.strip().lower() for r in os.getenv("YTDLP_JS_RUNTIMES", "deno,node").split(",") if r.strip()]


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file:\n"
            "    OPENAI_API_KEY=sk-proj-..."
        )
    return OPENAI_API_KEY
