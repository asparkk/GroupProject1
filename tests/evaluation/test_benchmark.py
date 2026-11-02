import time
import numpy as np
from src.evaluation.benchmark import time_inference, time_single_image, time_training


def test_inference_timing_close_to_known_sleep():
    X = np.zeros((10, 784))
    #creating a fake model that sleeps 10 ms per call so about 1 ms for 10 images
    res = time_inference(lambda x: time.sleep(0.01), X, n_runs=3)
    assert 0.8 < res["ms_per_image"] < 3.0


def test_single_image_and_training():
    X = np.zeros((5, 784))
    assert time_single_image(lambda x: x.sum(), X, 5)["ms_per_image"] >= 0
    out, secs = time_training(lambda a: a * 2, 21)
    assert out == 42 and secs >= 0