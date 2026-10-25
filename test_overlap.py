import numpy as np

from preprocess import count_overlap


def test_shared_picture():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [1, 2]])
    assert count_overlap(A, B) == 1


def test_gives_zero():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    assert count_overlap(A, B) == 0


def test_twice():
    A = np.array([[1, 2], [1, 2], [3, 4]])
    B = np.array([[1, 2]])
    assert count_overlap(A, B) == 2
