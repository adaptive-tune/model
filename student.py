import sys

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from scipy.special import expit, softmax

from common import GENRES, MOODS, N_FRAMES, N_MELS, ROOT

TEACHER_WEIGHT = 0.7
TEMPERATURE = 2
EPSILON = 1e-7

BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 512
EARLY_STOP_PATIENCE = 7
LEARNING_RATE = 2e-3
FINAL_LEARNING_RATE_FRACTION = 0.01
DROPOUT = 0.2
MAX_LOUDNESS_SHIFT = 1.0
SEED = 0

KERNEL_SIZE = 3
BPM_HIDDEN_UNITS = 32
BPM_CENTER = 120
BPM_SCALE = 40

SIZES = {
    "small": {
        "stem": (24, 2),
        "blocks": [(40, 2), (64, 1), (96, 2), (128, 1), (192, 2), (256, 1)],
        "epochs": 40,
    },
    "large": {
        "stem": (32, 1),
        "blocks": [
            (64, 2),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (256, 1),
            (384, 2),
            (512, 1),
        ],
        "epochs": 18,
    },
}


def conv(x, filters, kernel, stride):
    x = layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False)(x)
    return layers.ReLU()(layers.BatchNormalization()(x))


def separable(x, filters, stride):
    x = layers.DepthwiseConv2D(
        KERNEL_SIZE, strides=stride, padding="same", use_bias=False
    )(x)
    x = layers.ReLU()(layers.BatchNormalization()(x))
    return conv(x, filters, 1, 1)


def build(size, mean, std):
    inputs = keras.Input((N_FRAMES, N_MELS, 1))
    x = layers.Rescaling(1 / std, -mean / std)(inputs)
    filters, stride = SIZES[size]["stem"]
    x = conv(x, filters, KERNEL_SIZE, stride)
    for filters, stride in SIZES[size]["blocks"]:
        x = separable(x, filters, stride)
    x = layers.Dropout(DROPOUT)(layers.GlobalAveragePooling2D()(x))
    genre = layers.Dense(len(GENRES), name="genre")(x)
    mood = layers.Dense(len(MOODS), name="mood")(x)
    bpm = layers.Dense(BPM_HIDDEN_UNITS, activation="relu")(x)
    bpm = layers.Dense(1, name="bpm")(bpm)
    return keras.Model(inputs, [genre, mood, bpm])


def genre_loss(y, logits):
    n = len(GENRES)
    teacher = tf.nn.softmax(tf.math.log(y[:, :n] + EPSILON) / TEMPERATURE)
    student = tf.nn.log_softmax(logits / TEMPERATURE)
    kl = tf.reduce_sum(teacher * (tf.math.log(teacher + EPSILON) - student), axis=-1)
    ce = -tf.reduce_sum(y[:, n:] * tf.nn.log_softmax(logits), axis=-1)
    return TEACHER_WEIGHT * TEMPERATURE**2 * kl + (1 - TEACHER_WEIGHT) * ce


def targets(d, rows):
    onehot = np.eye(len(GENRES), dtype=np.float32)[d["genre"][rows]]
    return {
        "genre": np.concatenate([d["genre_prob"][rows], onehot], axis=1),
        "mood": d["mood"][rows],
        "bpm": ((d["bpm"][rows] - BPM_CENTER) / BPM_SCALE)[:, None],
    }


class Batches(keras.utils.PyDataset):
    def __init__(self, x, d, rows, augment):
        super().__init__()
        self.x = x
        self.d = d
        self.rows = np.flatnonzero(rows)
        self.augment = augment
        self.rng = np.random.default_rng(SEED)
        self.on_epoch_end()

    def __len__(self):
        return -(-len(self.rows) // BATCH_SIZE)

    def on_epoch_end(self):
        if self.augment:
            self.rng.shuffle(self.rows)

    def __getitem__(self, i):
        rows = np.sort(self.rows[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])
        x = self.x[rows].astype(np.float32)
        if self.augment:
            shift = self.rng.uniform(
                -MAX_LOUDNESS_SHIFT, MAX_LOUDNESS_SHIFT, (len(rows), 1, 1, 1)
            )
            x += shift.astype(np.float32)
            for j, frames in enumerate(self.rng.integers(0, N_FRAMES, len(rows))):
                x[j] = np.roll(x[j], frames, axis=0)
        return x, targets(self.d, rows)


def track_accuracy(track_id, genre, prob):
    ids, inverse = np.unique(track_id, return_inverse=True)
    total = np.zeros((len(ids), prob.shape[1]))
    np.add.at(total, inverse, prob)
    true = np.zeros(len(ids), dtype=int)
    true[inverse] = genre
    return (total.argmax(axis=1) == true).mean()


def evaluate(model, x, d, rows):
    genre, mood, bpm = model.predict(x[rows], batch_size=PREDICT_BATCH_SIZE, verbose=0)
    prob = softmax(genre, axis=1)
    print("segment accuracy", (prob.argmax(axis=1) == d["genre"][rows]).mean())
    print("track accuracy", track_accuracy(d["tid"][rows], d["genre"][rows], prob))
    print("mood mae", np.abs(expit(mood) - d["mood"][rows]).mean())
    print("bpm mae", np.abs(bpm[:, 0] * BPM_SCALE + BPM_CENTER - d["bpm"][rows]).mean())


if __name__ == "__main__":
    size = sys.argv[1]
    keras.utils.set_random_seed(SEED)
    mel = np.load(ROOT / "out" / "mel.npz")
    d = dict(np.load(ROOT / "out" / "teacher.npz"))
    assert np.array_equal(mel["tid"], d["tid"])
    x = mel["mel"][..., None]
    split = d["split"]
    train, val, test = (split == s for s in ["training", "validation", "test"])
    x_train = x[train]
    mean = float(x_train.mean(dtype=np.float64))
    std = float(x_train.std(dtype=np.float64))
    del x_train
    model = build(size, mean, std)
    epochs = SIZES[size]["epochs"]
    train_batches = Batches(x, d, train, augment=True)
    steps = epochs * len(train_batches)
    schedule = keras.optimizers.schedules.CosineDecay(
        LEARNING_RATE, steps, alpha=FINAL_LEARNING_RATE_FRACTION
    )
    model.compile(
        optimizer=keras.optimizers.Adam(schedule),
        loss={
            "genre": genre_loss,
            "mood": keras.losses.BinaryCrossentropy(from_logits=True),
            "bpm": keras.losses.Huber(),
        },
    )
    model.fit(
        train_batches,
        validation_data=Batches(x, d, val, augment=False),
        epochs=epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=EARLY_STOP_PATIENCE, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    model.save(ROOT / "out" / f"student_{size}.keras")
    evaluate(model, x, d, test)
