from essentia.standard import MonoLoader

from common import SAMPLE_RATE, SEGMENT_SAMPLES


def load(path):
    return MonoLoader(filename=str(path), sampleRate=SAMPLE_RATE, resampleQuality=4)()


def cut(audio):
    n = len(audio) // SEGMENT_SAMPLES
    return audio[: n * SEGMENT_SAMPLES].reshape(n, SEGMENT_SAMPLES)
