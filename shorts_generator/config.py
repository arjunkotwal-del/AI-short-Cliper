import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto")  # auto, cpu, or cuda


def _default_output_dir() -> str:
    """Pick a safe default output directory.

    If the project lives under a path with non-ASCII characters (e.g. Windows
    OneDrive "Документы"), ffmpeg and some tools can choke on it.  Fall back to
    ~/shorts-output which is always ASCII-safe.
    """
    cwd = os.getcwd()
    try:
        cwd.encode("ascii")
        return os.path.join(cwd, "output")
    except UnicodeEncodeError:
        return os.path.join(os.path.expanduser("~"), "shorts-output")


LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR") or _default_output_dir()


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file:\n"
            "    OPENAI_API_KEY=sk-proj-..."
        )
    return OPENAI_API_KEY
