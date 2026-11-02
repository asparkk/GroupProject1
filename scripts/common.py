import numpy as np
from sklearn.datasets import fetch_openml
from preprocess import normalize, split_train_val  # Person B's functions


def load_data(seed=42):
    """Download MNIST and return a dict shaped like B's prepare_data output:
    X_train, y_train, X_val, y_val, X_test, y_test (48000 / 12000 / 10000 images).

    Swap this body for prepare_data(folder, seed) if B's data files ever become available."""
    mn = fetch_openml('mnist_784', version=1, as_frame=False)   # 70,000 images; the first 60,000 are the original training set
    X_raw, y_all = mn.data.astype(np.uint8), mn.target.astype(int)
    X_tv, y_tv = normalize(X_raw[:60000]), y_all[:60000]         # B: scale pixels from 0-255 to 0-1
    X_test, y_test = normalize(X_raw[60000:]), y_all[60000:]     # test set: scaled only
    X_train, y_train, X_val, y_val = split_train_val(X_tv, y_tv, seed)   # B: 80/20 split with a fixed seed
    data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test}
    for k in data:
        if k.startswith('X_'):
            data[k] = data[k].astype(np.float32)                  # halves memory; A's model uses float32 anyway
    return data