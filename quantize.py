import sys

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow.lite.python import schema_py_generated as schema
from tflite_micro import runtime

from common import GENRES, MOODS, N_FRAMES, N_MELS, ROOT
from student import BPM_CENTER, BPM_SCALE, track_accuracy

REPRESENTATIVE_SAMPLES = 500
TFLM_CHECK_SAMPLES = 200
MAX_ARENA_BYTES = 8 << 20
ARENA_PRECISION_BYTES = 64
OUTPUT_SIZES = [len(GENRES), len(MOODS), 1]
INT8_MIN, INT8_MAX = -128, 127


def with_activations(model):
    inputs = keras.Input((N_FRAMES, N_MELS, 1), batch_size=1)
    genre, mood, bpm = model(inputs)
    return keras.Model(
        inputs,
        [
            layers.Softmax()(genre),
            layers.Activation("sigmoid")(mood),
            layers.Rescaling(BPM_SCALE, BPM_CENTER)(bpm),
        ],
    )


def convert(model, samples):
    def representative():
        for x in samples:
            yield [x[None].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = bytearray(converter.convert())
    graph = schema.Model.GetRootAsModel(blob, 0).Subgraphs(0)
    outputs = graph.OutputsAsNumpy()
    outputs[:] = sorted(
        outputs, key=lambda i: OUTPUT_SIZES.index(graph.Tensors(i).ShapeAsNumpy()[1])
    )
    return bytes(blob)


def quantize(x, details):
    scale, zero_point = details["quantization"]
    return np.clip(np.round(x / scale) + zero_point, INT8_MIN, INT8_MAX).astype(np.int8)


def dequantize(q, details):
    scale, zero_point = details["quantization"]
    return (q.astype(np.float32) - zero_point) * scale


def run_tflite(blob, x):
    interpreter = tf.lite.Interpreter(
        model_content=blob,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    results = [[] for _ in output_details]
    for sample in x:
        interpreter.set_tensor(input_details["index"], quantize(sample, input_details)[None])
        interpreter.invoke()
        for result, details in zip(results, output_details):
            result.append(interpreter.get_tensor(details["index"])[0])
    return [np.stack(r) for r in results], input_details, output_details


def run_tflm(blob, x, arena=MAX_ARENA_BYTES):
    interpreter = runtime.Interpreter.from_bytes(blob, arena_size=arena)
    results = [[] for _ in OUTPUT_SIZES]
    for sample in x:
        interpreter.set_input(sample[None], 0)
        interpreter.invoke()
        for i, result in enumerate(results):
            result.append(interpreter.get_output(i).reshape(-1))
    return [np.stack(r) for r in results]


def fits(blob, x, arena):
    try:
        run_tflm(blob, x, arena)
        return True
    except Exception:
        return False


def arena_size(blob, x):
    low, high = 0, MAX_ARENA_BYTES
    while high - low > ARENA_PRECISION_BYTES:
        middle = (low + high) // 2
        if fits(blob, x, middle):
            high = middle
        else:
            low = middle
    return high


if __name__ == "__main__":
    size = sys.argv[1]
    mel = np.load(ROOT / "out" / "mel.npz")
    d = np.load(ROOT / "out" / "teacher.npz")
    x = mel["mel"][..., None]
    train = np.flatnonzero(d["split"] == "training")
    test = d["split"] == "test"
    rng = np.random.default_rng(0)
    samples = x[np.sort(rng.choice(train, REPRESENTATIVE_SAMPLES, replace=False))]
    model = keras.models.load_model(ROOT / "out" / f"student_{size}.keras", compile=False)
    blob = convert(with_activations(model), samples)
    (ROOT / "out" / f"student_{size}.tflite").write_bytes(blob)

    x_test = x[test].astype(np.float32)
    outputs, input_details, output_details = run_tflite(blob, x_test)
    genre, mood, bpm = (dequantize(o, od) for o, od in zip(outputs, output_details))
    print("tflite bytes", len(blob))
    print("input scale, zero point", input_details["quantization"])
    print("segment accuracy", (genre.argmax(axis=1) == d["genre"][test]).mean())
    print("track accuracy", track_accuracy(d["tid"][test], d["genre"][test], genre))
    print("mood mae", np.abs(mood - d["mood"][test]).mean())
    print("bpm mae", np.abs(bpm[:, 0] - d["bpm"][test]).mean())

    x_check = quantize(x_test[:TFLM_CHECK_SAMPLES], input_details)
    tflm = run_tflm(blob, x_check)
    diff = [np.abs(a.astype(int) - b[:TFLM_CHECK_SAMPLES].astype(int)).max() for a, b in zip(tflm, outputs)]
    print("tflm vs tflite max int8 diff (genre, mood, bpm)", diff)
    print("tflm arena bytes", arena_size(blob, x_check[:1]))
