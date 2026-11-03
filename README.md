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
