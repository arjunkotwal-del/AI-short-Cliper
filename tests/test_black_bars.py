from shorts_generator.local.clipper import _choose_bar_crop

W, H = 1920, 1080


def test_symmetric_bars_seen_in_every_sample_are_cropped():
    boxes = [(1920, 956, 0, 62)] * 5
    assert _choose_bar_crop(boxes, W, H) == (1920, 956, 0, 62)


def test_dark_scene_cannot_shrink_the_crop():
    # one dark sample makes cropdetect report a much smaller box; the union keeps the picture
    boxes = [(1920, 956, 0, 62)] * 4 + [(1920, 600, 0, 240)]
    assert _choose_bar_crop(boxes, W, H) == (1920, 956, 0, 62)


def test_a_bright_full_frame_sample_means_no_bars():
    boxes = [(1920, 956, 0, 62)] * 4 + [(1920, 1080, 0, 0)]
    assert _choose_bar_crop(boxes, W, H) is None


def test_thin_or_lopsided_bars_are_left_alone():
    assert _choose_bar_crop([(1920, 1060, 0, 10)] * 5, W, H) is None  # under 3% each
    assert _choose_bar_crop([(1920, 900, 0, 160)] * 5, W, H) is None  # 160 top vs 20 bottom


def test_mostly_black_footage_is_not_cropped_to_a_sliver():
    assert _choose_bar_crop([(1920, 400, 0, 340)] * 5, W, H) is None


def test_too_few_samples_means_no_decision():
    assert _choose_bar_crop([(1920, 956, 0, 62)] * 2, W, H) is None
