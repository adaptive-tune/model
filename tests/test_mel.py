import librosa
import numpy as np
import pytest

from common import HOP, N_FFT, N_FRAMES, N_MELS, SAMPLE_RATE
from mel import FB, logmel

SEG = N_FFT + (N_FRAMES - 1) * HOP


def tone(freq):
    t = np.arange(SEG) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_shape_and_type():
    m = logmel(np.random.default_rng(0).standard_normal(SEG).astype(np.float32))
    assert m.shape == (N_FRAMES, N_MELS)
    assert m.dtype == np.float32


def test_too_short_input_fails():
    with pytest.raises(IndexError):
        logmel(np.zeros(SEG - 1, np.float32))


def test_silence_is_floor():
    assert np.allclose(logmel(np.zeros(SEG, np.float32)), np.log(1e-6))


@pytest.mark.parametrize("freq", [100, 440, 1000, 4000])
def test_tone_lands_in_its_band(freq):
    expected = FB[:, round(freq * N_FFT / SAMPLE_RATE)].argmax()
    assert logmel(tone(freq)).mean(axis=0).argmax() == expected


def test_filterbank_matches_librosa():
    ref = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=SAMPLE_RATE / 2, htk=True, norm=None)
    assert np.allclose(FB, ref, atol=1e-5)


def test_real_music_matches_librosa(music_path):
    x, _ = librosa.load(music_path, sr=SAMPLE_RATE, mono=True, duration=SEG / SAMPLE_RATE + 1)
    x = x[:SEG]
    ref = librosa.feature.melspectrogram(
        y=x, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, window="hann", center=False,
        power=2.0, n_mels=N_MELS, fmin=0, fmax=SAMPLE_RATE / 2, htk=True, norm=None,
    )
    ref = np.log(ref.T + 1e-6)
    assert np.abs(logmel(x) - ref).max() < 1e-3
