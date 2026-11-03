import numpy as np
from test_common import needs_dataset

from audio import cut, load
from common import SAMPLE_RATE, SEGMENT_SAMPLES, find_track


def test_30s_gives_9_pieces():
    assert cut(np.zeros(30 * SAMPLE_RATE, np.float32)).shape == (9, SEGMENT_SAMPLES)


def test_short_clip_gives_fewer():
    assert cut(np.zeros(2 * SEGMENT_SAMPLES + 100, np.float32)).shape == (
        2,
        SEGMENT_SAMPLES,
    )
    assert cut(np.zeros(100, np.float32)).shape == (0, SEGMENT_SAMPLES)


def test_pieces_keep_order():
    pieces = cut(np.arange(3 * SEGMENT_SAMPLES, dtype=np.float32))
    assert pieces[1][0] == SEGMENT_SAMPLES


@needs_dataset
def test_real_mp3():
    audio = load(find_track(2))
    assert audio.dtype == np.float32
    assert cut(audio).shape == (9, SEGMENT_SAMPLES)
