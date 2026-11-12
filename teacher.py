import essentia
import numpy as np
from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs

from audio import cut, load
from classifier import genre_prob
from common import MOODS, POSITIVE, ROOT, find_track, tracks

essentia.log.warningActive = False


TEACHER = ROOT / "out" / "teacher"
MODELS = ROOT / "models"
EFFNET_FRAME_HOP = 256
EFFNET_PATCH_HOP = 62
PATCHES_PER_SEGMENT = 2
PATCHES_PER_SLOT = 4
SLOT_SAMPLES = PATCHES_PER_SLOT * EFFNET_PATCH_HOP * EFFNET_FRAME_HOP

effnet = TensorflowPredictEffnetDiscogs(
    graphFilename=str(MODELS / "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1"
)
moods = {
    m: TensorflowPredict2D(
        graphFilename=str(MODELS / f"mood_{m}-discogs-effnet-1.pb"),
        output="model/Softmax",
    )
    for m in MOODS
}


def spaced(pieces):
    audio = np.zeros(len(pieces) * SLOT_SAMPLES, dtype=np.float32)
    for k, piece in enumerate(pieces):
        audio[k * SLOT_SAMPLES : k * SLOT_SAMPLES + len(piece)] = piece
    return audio


def process(track_id):
    pieces = cut(load(find_track(track_id)))
    n = len(pieces)
    patches = np.arange(n)[:, None] * PATCHES_PER_SLOT + np.arange(PATCHES_PER_SEGMENT)
    emb = effnet(spaced(pieces))[patches.reshape(-1)]
    mood = np.stack([moods[m](emb)[:, POSITIVE[m]] for m in MOODS], axis=1)
    return {
        "emb": emb.reshape(n, PATCHES_PER_SEGMENT, -1).mean(axis=1).astype(np.float16),
        "mood": mood.reshape(n, PATCHES_PER_SEGMENT, -1).mean(axis=1).astype(np.float32),
    }


def generate_npz():
    TEACHER.mkdir(parents=True, exist_ok=True)
    t = tracks()
    embs, mood_list, track_ids, pieces = [], [], [], []
    for track_id in t.index:
        path = TEACHER / f"{track_id}.npz"
        if not path.exists():
            try:
                np.savez(path, **process(track_id))
            except Exception as e:
                print("skip", track_id, e)
                continue
        d = np.load(path)
        embs.append(d["emb"])
        mood_list.append(d["mood"])
        track_ids.append(np.full(len(d["emb"]), track_id))
        pieces.append(np.arange(len(d["emb"])))
    track_id = np.concatenate(track_ids)
    split = t.loc[track_id, "split"].to_numpy().astype(str)
    genre = t.loc[track_id, "genre"].to_numpy()
    artist = t.loc[track_id, "artist"].to_numpy()
    emb = np.concatenate(embs).astype(np.float32)
    np.savez(
        ROOT / "out" / "teacher.npz",
        tid=track_id,
        k=np.concatenate(pieces),
        split=split,
        genre=genre,
        genre_prob=genre_prob(emb, genre, split, artist),
        mood=np.concatenate(mood_list),
    )


if __name__ == "__main__":
    generate_npz()
