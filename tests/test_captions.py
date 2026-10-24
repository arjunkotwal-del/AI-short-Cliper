import re

from shorts_generator.local.clipper import _chunk_words, _generate_ass, _hook_font_size


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_chunks_hold_at_most_three_words():
    words = [_w(f"W{i}", i * 0.3, i * 0.3 + 0.25) for i in range(7)]
    assert [len(c) for c in _chunk_words(words)] == [3, 3, 1]


def test_chunks_break_at_sentence_ends():
    words = [_w("THRILLED,", 0.0, 0.4), _w("THOUGH.", 0.45, 0.8), _w("IN", 0.85, 1.0), _w("FACT", 1.05, 1.3)]
    assert [[w["word"] for w in c] for c in _chunk_words(words)] == [["THRILLED,", "THOUGH."], ["IN", "FACT"]]


def test_chunks_break_at_long_pauses():
    words = [_w("ONE", 0.0, 0.3), _w("TWO", 1.2, 1.5), _w("THREE", 1.55, 1.8)]
    assert [[w["word"] for w in c] for c in _chunk_words(words)] == [["ONE"], ["TWO", "THREE"]]


def test_short_hooks_keep_full_size_and_long_ones_shrink():
    short = _hook_font_size("why lions love pitch black", 720, 1280)
    long = _hook_font_size("Exploring why nocturnal predators would thrive while ancient animal breeding cycles break down", 720, 1280)
    assert short == max(72, int(1280 * 0.065))
    assert 44 <= long < short


def test_ass_places_captions_above_platform_ui_and_shows_hook_text():
    words = [_w("Hello", 10.0, 10.4), _w("world.", 10.5, 10.9)]
    ass = _generate_ass(words, 10.0, 20.0, 720, 1280, hook_text="why lions love pitch black")

    pop_style = next(line for line in ass.splitlines() if line.startswith("Style: Pop"))
    margin_v = int(pop_style.split(",")[-2])
    assert margin_v >= 1280 * 0.25
    assert "WHY LIONS LOVE PITCH BLACK" in ass
    # caption times are relative to the clip start
    assert re.search(r"Dialogue: 0,0:00:00\.00,0:00:00\.90,Pop", ass)


def test_ass_is_none_without_words_or_hook():
    assert _generate_ass([], 0.0, 10.0, 720, 1280) is None
