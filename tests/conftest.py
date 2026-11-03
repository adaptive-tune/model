import hashlib
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path(__file__).parent / "data"
MUSIC_URL = "https://librosa.org/data/audio/admiralbob77_-_Choice_-_Drum-bass.ogg"
MUSIC_SHA256 = "ac644f9645e7c15174e4a4f8561e4d1448d7f6e59ff6b0556b310ebbced879bc"


@pytest.fixture(scope="session")
def music_path():
    path = DATA / "choice.ogg"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MUSIC_URL, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MUSIC_SHA256
    return path
