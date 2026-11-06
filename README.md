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

---

### Run teacher

**Requirements**:

- Downloaded dataset
- Downloaded models
- 500 MB of free space
- .venv setup

```
python teacher.py
```

Which will run Essentia models on every track and save results into `./out/teacher.npz`:

- `genre_prob`: genre probabilities from a classifier trained on Discogs-EffNet embeddings
- `mood`: mood probabilities (happy, sad, relaxed, aggressive, party)
- `bpm`: tempo from TempoCNN

Takes a few hours, safe to stop and run again, finished tracks are cached in `./out/teacher/`

---

### Train student

**Requirements**:

- `./out/mel.npz` and `./out/teacher.npz`
- 4 GB of free RAM
- .venv setup

```
python student.py small
python student.py large
```

Which will train a small CNN on mel spectrograms to copy the teacher, and save it into `./out/student_small.keras` or `./out/student_large.keras`

- `small`: for ESP32 without PSRAM
- `large`: for ESP32 with PSRAM

Takes 1-2 hours per model, prints test accuracy at the end

---

### Quantize student

**Requirements**:

- `./out/student_small.keras` or `./out/student_large.keras`
- .venv setup

```
python quantize.py small
python quantize.py large
```

Which will convert the model to int8 and save it into `./out/student_small.tflite` or `./out/student_large.tflite`

Prints int8 test accuracy, difference against TensorFlow Lite Micro, and memory needed on the ESP32 (tensor arena)
