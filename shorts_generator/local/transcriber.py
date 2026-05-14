"""Local transcription via faster-whisper.

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text]}.
"""
from typing import Dict, Optional

from ..config import LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Run faster-whisper on a local file path."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[transcribe/local] faster-whisper model={LOCAL_WHISPER_MODEL} device={device}", flush=True)

    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        media_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    segments = []
    all_words = []
    for s in segments_iter:
        words = []
        if s.words:
            for w in s.words:
                entry = {"word": w.word, "start": float(w.start), "end": float(w.end)}
                words.append(entry)
                all_words.append(entry)
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
            "words": words,
        })

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/local] {len(segments)} segments, {len(all_words)} words, {duration:.0f}s of audio", flush=True)
    return {"duration": duration, "segments": segments, "words": all_words}
