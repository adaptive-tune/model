import pytest

from common import DATASET, GENRES, find_track, tracks

needs_dataset = pytest.mark.skipif(
    not (DATASET / "fma_metadata" / "tracks.csv").exists(),
    reason="dataset not downloaded, cd dataset, run download.sh",
)


def test_find_track_path():
    assert find_track(2) == DATASET / "fma_small" / "000" / "000002.mp3"
    assert find_track(123456) == DATASET / "fma_small" / "123" / "123456.mp3"


@needs_dataset
def test_tracks_are_balanced():
    t = tracks()
    assert len(t) == 8000
    assert t["genre"].value_counts().to_dict() == {g: 1000 for g in range(len(GENRES))}
    assert t["split"].value_counts().to_dict() == {
        "training": 6400,
        "validation": 800,
        "test": 800,
    }


@needs_dataset
def test_tracks_point_to_real_files():
    t = tracks()
    assert all(find_track(i).exists() for i in t.index[:20])
