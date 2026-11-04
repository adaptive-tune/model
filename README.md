# model

### Get started

Requirements:

- Python 3.13

Initialize `.venv`

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Run tests

Requirements:

- Python 3.13

To run tests:

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

---

### Download dataset

**Requirements**:

- Linux
- wget
- sha1sum
- unzip
- 20 GB of free space

To download dataset:

```
cd dataset
./download.sh
```

Which will download 2 zip archives, and unzip them into `./dataset/fma_metadata/` and `./dataset/fma_small/`
Remove zip files yourself when you've made sure that everything is ok

---

### Prepare dataset

**Requirements**:

- Downloaded dataset
- 2 GB of free space
- .venv setup

```
python export.py
```

Which will prepare dataset into `./out/mel.npz`:

---

### Download models

**Requirements**:

- Linux
- wget
- sha1sum
- 25 MB of free space

To download models:

```
cd models
./download.sh
```

Which will download Essentia models into `./models/`:

- `discogs-effnet-bs64-1.pb`: Discogs-EffNet embeddings
- `deeptemp-k16-3.pb`: TempoCNN for BPM
- `mood_*-discogs-effnet-1.pb` and `.json`: mood classifiers (aggressive, happy, party, relaxed, sad)

Models are licensed under CC BY-NC-SA 4.0, see <https://essentia.upf.edu/models.html>
