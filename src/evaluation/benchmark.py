import time
import numpy as np

#This file contains functions to measure the time taken for inference and training of a model


def time_inference(predict_fn, X, n_runs=5, warmup=1):
    """Average milliseconds per image when predicting a whole array at once.

    predict_fn is any function mapping (N, 784) images to predictions, such as A's
    model.predict. A warm-up run is discarded because the first call is often slower."""
    for _ in range(warmup):
        predict_fn(X)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()  # perf_counter is the right clock for short intervals
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    return {"ms_per_image": 1000 * float(np.mean(times)) / len(X),
            "total_seconds_mean": float(np.mean(times)), "n_images": len(X), "n_runs": n_runs}


def time_single_image(predict_fn, X, n_images=200):
    """Latency when classifying one image at a time (closer to real deployment)."""
    n = min(n_images, len(X))
    predict_fn(X[:1])  # warm-up
    t0 = time.perf_counter()
    for i in range(n):
        predict_fn(X[i:i + 1])  # slice keeps the 2-D shape (1, 784)
    return {"ms_per_image": 1000 * (time.perf_counter() - t0) / n}


def time_training(train_fn, *args, **kwargs):
    """Run train_fn and return (its result, seconds taken)."""
    t0 = time.perf_counter()
    out = train_fn(*args, **kwargs)
    return out, time.perf_counter() - t0