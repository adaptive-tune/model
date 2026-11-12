import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEMPERATURE_BOUNDS = (0.05, 20)
ECE_BINS = 15


def scale(p, T):
    return softmax(np.log(np.clip(p, 1e-12, 1)) / T, axis=1).astype(np.float32)


def nll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()


def fit_temperature(logits, y):
    return minimize_scalar(
        lambda t: nll(softmax(logits / t, axis=1), y), bounds=TEMPERATURE_BOUNDS, method="bounded"
    ).x


def ece(p, y):
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    bins = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    return sum(
        (bins == b).mean() * abs(correct[bins == b].mean() - confidence[bins == b].mean())
        for b in range(ECE_BINS)
        if (bins == b).any()
    )


def genre_prob(emb, genre, split, artist):
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier((256,), alpha=1e-2, early_stopping=True, random_state=0),
    )
    train, val = split == "training", split == "validation"
    clf.fit(emb[train], genre[train])
    raw = clf.predict_proba(emb)
    T = minimize_scalar(
        lambda t: nll(scale(raw[val], t), genre[val]), bounds=TEMPERATURE_BOUNDS, method="bounded"
    ).x
    p = scale(raw, T)
    idx = np.flatnonzero(train)
    for a, b in GroupKFold(5).split(idx, groups=artist[idx]):
        fold = clone(clf).fit(emb[idx[a]], genre[idx[a]])
        p[idx[b]] = scale(fold.predict_proba(emb[idx[b]]), T)
    test = split == "test"
    print("T", T, "test acc", (p[test].argmax(1) == genre[test]).mean())
    return p
