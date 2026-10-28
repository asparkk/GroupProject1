import numpy as np

from preprocess import make_batches


def test_sizes():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    batches = make_batches(X, y, 4, 42)
    assert len(batches) == 3
    assert len(batches[0][1]) == 4
    assert len(batches[1][1]) == 4
    assert len(batches[2][1]) == 2


def test_all_pic_used():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    batches = make_batches(X, y, 4, 42)
    seen = []
    for X_batch, y_batch in batches:
        for label in y_batch:
            seen.append(int(label))
    assert sorted(seen) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_labels():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    batches = make_batches(X, y, 4, 42)
    for X_batch, y_batch in batches:
        for i in range(len(y_batch)):
            assert X_batch[i][0] == y_batch[i]


def test_seed():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    a = make_batches(X, y, 4, 42)
    b = make_batches(X, y, 4, 42)
    c = make_batches(X, y, 4, 7)
    assert list(a[0][1]) == list(b[0][1])
    assert list(a[0][1]) != list(c[0][1])
