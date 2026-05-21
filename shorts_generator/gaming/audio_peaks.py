"""Module 1: Audio peak detection for gaming/streamer clips.

Extracts raw audio from a video, computes per-window dB levels,
finds the top-N loudest peaks (with minimum gap dedup), and returns
clip boundaries centered on each peak.
"""
import os
import shutil
import struct
import subprocess
import tempfile
from typing import List, Tuple

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_TIMEOUT = 300


def _get_duration(path: str) -> float:
    """Get video duration in seconds."""
    r = subprocess.run(
        [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True, timeout=_TIMEOUT,
    )
    return float(r.stdout.strip())


def _extract_raw_audio(video_path: str, wav_path: str) -> str:
    """Extract mono 16kHz s16le WAV from video."""
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        "-f", "wav", wav_path,
    ]
    subprocess.run(cmd, check=True, timeout=_TIMEOUT)
    return wav_path


def _compute_rms_windows(wav_path: str, window_sec: float = 0.5) -> List[Tuple[float, float]]:
    """Compute RMS dB for each window.

    Returns list of (time_center, rms_db).
    Uses raw struct parsing — no numpy dependency.
    """
    import math

    with open(wav_path, "rb") as f:
        # Skip WAV header (44 bytes standard)
        f.read(44)
        raw = f.read()

    sample_rate = 16000
    samples_per_window = int(sample_rate * window_sec)
    num_samples = len(raw) // 2  # 16-bit = 2 bytes
    fmt = f"<{num_samples}h"
    samples = struct.unpack(fmt, raw[:num_samples * 2])

    results = []
    for i in range(0, num_samples, samples_per_window):
        chunk = samples[i:i + samples_per_window]
        if len(chunk) < samples_per_window // 2:
            break
        # RMS
        sum_sq = sum(s * s for s in chunk)
        rms = math.sqrt(sum_sq / len(chunk))
        db = 20 * math.log10(max(rms, 1)) - 90.3  # normalize so ~0 dB = loud
        time_center = (i + len(chunk) / 2) / sample_rate
        results.append((time_center, db))

    return results


def _find_peaks(
    rms_data: List[Tuple[float, float]],
    num_peaks: int,
    min_gap: float = 60.0,
    edge_margin: float = 15.0,
    duration: float = 0.0,
) -> List[float]:
    """Find top-N loudest peaks with minimum gap dedup.

    Args:
        rms_data: list of (time, db) tuples
        num_peaks: how many peaks to return
        min_gap: minimum seconds between peaks
        edge_margin: skip peaks too close to start/end
        duration: total video duration
    """
    # Sort by dB descending
    sorted_data = sorted(rms_data, key=lambda x: x[1], reverse=True)

    peaks: List[float] = []
    for time, db in sorted_data:
        if len(peaks) >= num_peaks:
            break
        # Skip edges
        if time < edge_margin or (duration > 0 and time > duration - edge_margin):
            continue
        # Check gap from existing peaks
        if any(abs(time - p) < min_gap for p in peaks):
            continue
        peaks.append(time)

    return sorted(peaks)


def detect_audio_peaks(
    video_path: str,
    num_clips: int = 3,
    clip_duration: float = 22.0,
    min_gap: float = 60.0,
) -> List[dict]:
    """Detect top-N loudest moments and return clip boundaries.

    Returns list of dicts: [{start_time, end_time, peak_time, peak_db}, ...]
    """
    duration = _get_duration(video_path)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        _extract_raw_audio(video_path, wav_path)
        rms_data = _compute_rms_windows(wav_path, window_sec=0.5)

    peaks = _find_peaks(
        rms_data, num_peaks=num_clips,
        min_gap=min_gap, edge_margin=clip_duration / 2,
        duration=duration,
    )

    clips = []
    for peak_time in peaks:
        # Center the clip around the peak, bias slightly before (40/60 split)
        half = clip_duration / 2
        start = max(0, peak_time - half * 0.8)
        end = min(duration, start + clip_duration)
        # Adjust start if we hit the end
        if end - start < clip_duration:
            start = max(0, end - clip_duration)

        # Find the actual dB at this peak
        peak_db = 0.0
        for t, db in rms_data:
            if abs(t - peak_time) < 0.5:
                peak_db = db
                break

        clips.append({
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "peak_time": round(peak_time, 3),
            "peak_db": round(peak_db, 1),
        })

    return clips
