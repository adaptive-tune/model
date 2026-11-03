from pathlib import Path

import pandas

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]


SAMPLE_RATE = 16000
N_FFT = 1024
HOP = 512
N_MELS = 64
N_FRAMES = 96

SEGMENT_SAMPLES = N_FFT + (N_FRAMES - 1) * HOP


def tracks():
    t = pandas.read_csv(
        DATASET / "fma_metadata" / "tracks.csv", index_col=0, header=[0, 1]
    )
    t = t[t[("set", "subset")] == "small"]
    return pandas.DataFrame(
        {
            "split": t[("set", "split")],
            "genre": t[("track", "genre_top")].map(GENRES.index),
            "artist": t[("artist", "id")],
        }
    )


def find_track(track_id):
    s = f"{track_id:06d}"
    return DATASET / "fma_small" / s[:3] / f"{s}.mp3"
