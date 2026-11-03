import numpy as np

from common import HOP, N_FFT, N_FRAMES, N_MELS, SAMPLE_RATE


def hz2mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel2hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def filterbank():
    bins = np.arange(N_FFT // 2 + 1) * SAMPLE_RATE / N_FFT
    p = mel2hz(np.linspace(0, hz2mel(SAMPLE_RATE / 2), N_MELS + 2))
    lo, c, hi = p[:-2, None], p[1:-1, None], p[2:, None]
    return np.maximum(
        0, np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c))
    ).astype(np.float32)


WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
FB = filterbank()


def logmel(x):
    idx = np.arange(N_FRAMES)[:, None] * HOP + np.arange(N_FFT)
    spec = np.abs(np.fft.rfft(x[idx] * WINDOW)) ** 2
    return np.log(spec @ FB.T + 1e-6).astype(np.float32)
