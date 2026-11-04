import numpy as np

from audio import cut, load
from common import ROOT, find_track, tracks
from mel import logmel

MEL = ROOT / "out" / "mel"


def process(track_id):
    pieces = cut(load(find_track(track_id)))
    return np.stack([logmel(p) for p in pieces]).astype(np.float16)


def generate_npz():
    MEL.mkdir(parents=True, exist_ok=True)
    t = tracks()
    mels, track_ids, pieces = [], [], []
    for track_id in t.index:
        path = MEL / f"{track_id}.npy"
        if not path.exists():
            try:
                np.save(path, process(track_id))
            except Exception as e:
                print("skip", track_id, e)
                continue
        m = np.load(path)
        mels.append(m)
        track_ids.append(np.full(len(m), track_id))
        pieces.append(np.arange(len(m)))
    track_id = np.concatenate(track_ids)
    np.savez(
        ROOT / "out" / "mel.npz",
        track_id=track_id,
        piece=np.concatenate(pieces),
        split=t.loc[track_id, "split"].to_numpy().astype(str),
        genre=t.loc[track_id, "genre"].to_numpy(),
        mel=np.concatenate(mels),
    )


if __name__ == "__main__":
    generate_npz()
