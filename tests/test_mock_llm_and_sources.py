import json

from shorts_generator import highlights, pipeline
from shorts_generator.local.mock_llm import MATH_SOLVER_SCRIPT, mock_response


def _transcript(seconds=300):
    segments = [{"start": float(t), "end": float(t + 4), "text": f"sentence number {t} about lions"} for t in range(0, seconds, 5)]
    return {"duration": float(seconds), "segments": segments, "words": []}


def test_mock_llm_drives_the_highlight_pipeline(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    result = highlights.get_highlights(_transcript(), num_clips=3)

    picks = result["highlights"]
    assert 1 <= len(picks) <= 3
    for h in picks:
        assert 0 <= h["start_time"] < h["end_time"] <= 310
        assert h["title"] and h["hook_sentence"]


def test_mock_llm_answers_social_copy_with_json(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    copy = highlights.generate_social_copy({"title": "why lions love pitch black"})
    assert copy["caption"].startswith("Why lions love pitch black")
    assert copy["hashtags"]


def test_mock_llm_keeps_the_math_solver_script_for_other_prompts():
    assert mock_response("Solve y' + 2y = x with y(0) = 5") == MATH_SOLVER_SCRIPT
    assert json.loads(mock_response("Please classify the content type of this video"))["density"] == "medium"


def test_local_files_skip_the_download(tmp_path, monkeypatch):
    video = tmp_path / "My Podcast (ep 3).mp4"
    video.write_bytes(b"not really a video")
    monkeypatch.setattr(pipeline, "download_youtube_local", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))

    path, folder = pipeline._resolve_source(str(video), "720")

    assert path == str(video.resolve())
    assert folder == "My_Podcast_ep_3"


def test_youtube_urls_are_downloaded_into_a_folder_named_by_id(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "download_youtube_local", lambda url, fmt: calls.append((url, fmt)) or "/tmp/source.mp4")

    path, folder = pipeline._resolve_source("https://www.youtube.com/watch?v=X5_JUc592Es", "1080")

    assert (path, folder) == ("/tmp/source.mp4", "X5_JUc592Es")
    assert calls == [("https://www.youtube.com/watch?v=X5_JUc592Es", "1080")]
