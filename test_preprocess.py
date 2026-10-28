import numpy as np

from preprocess import normalize, split_train_val


def test_norm_range():
    X = np.array([[0] * 392 + [255] * 392], dtype=np.uint8)
    out = normalize(X)
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert out.dtype == np.float64


def test_norm_shape():
    X = np.zeros((5, 784), dtype=np.uint8)
    out = normalize(X)
    assert out.shape == (5, 784)


def test_split_sizes():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train, y_train, X_val, y_val = split_train_val(X, y, 42)
    assert len(y_train) == 8
    assert len(y_val) == 2
    for label in y_train:
        assert label not in y_val


def test_split_seed():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X_train1, y_train1, X_val1, y_val1 = split_train_val(X, y, 42)
    X_train2, y_train2, X_val2, y_val2 = split_train_val(X, y, 42)
    assert list(y_train1) == list(y_train2)
    for i in range(len(y_train1)):
        assert X_train1[i][0] == y_train1[i]
