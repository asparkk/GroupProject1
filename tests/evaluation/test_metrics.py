import numpy as np
from sklearn.metrics import precision_recall_fscore_support  # used ONLY to cross-check
from src.evaluation.metrics import (accuracy, confusion_matrix, precision_recall_f1, macro_f1, per_class_report, most_confused_pairs)

# 6 samples, 3 classes. Hand-computed confusion matrix:
# true 0 -> predicted 0 once, 1 once   | true 1 -> predicted 1 twice
# true 2 -> predicted 2 once, 0 once
Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])


def test_accuracy_hand_computed():
    assert np.isclose(accuracy(Y_TRUE, Y_PRED), 4 / 6)  # 4 of 6 correct


def test_confusion_matrix_hand_computed():
    expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert np.array_equal(confusion_matrix(Y_TRUE, Y_PRED, 3), expected)


def test_precision_recall_f1_hand_computed():
    p, r, f = precision_recall_f1(Y_TRUE, Y_PRED, 3)
    assert np.allclose(p, [0.5, 2 / 3, 1.0])   # column-wise: 1/2, 2/3, 1/1
    assert np.allclose(r, [0.5, 1.0, 0.5])     # row-wise: 1/2, 2/2, 1/2
    assert np.allclose(f, [0.5, 0.8, 2 / 3])   # 2pr/(p+r) for each class


def test_perfect_predictions():
    y = np.arange(10).repeat(5)  # 0,0,0,0,0,1,1,...
    assert accuracy(y, y) == 1.0
    assert macro_f1(y, y) == 1.0


def test_class_never_predicted_gives_zero_not_nan():
    y_true = np.array([0, 1, 2]); y_pred = np.array([0, 1, 1])  # class 2 never predicted
    p, r, f = precision_recall_f1(y_true, y_pred, 3)
    assert not np.isnan(p).any() and p[2] == 0 and f[2] == 0


def test_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)  # Making a fixed seed so it is reproducible, like the project requires
    y_true = rng.integers(0, 10, 1000)
    y_pred = np.where(rng.random(1000) < 0.8, y_true, rng.integers(0, 10, 1000))
    p, r, f = precision_recall_f1(y_true, y_pred)
    sp, sr, sf, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(10), zero_division=0)
    assert np.allclose(p, sp) and np.allclose(r, sr) and np.allclose(f, sf)


def test_confusion_matrix_invariants():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 10, 500); y_pred = rng.integers(0, 10, 500)
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (10, 10) and cm.sum() == 500 # every sample counted once
    assert np.array_equal(cm.sum(axis=1), np.bincount(y_true, minlength=10)) # rows = class counts
    assert np.isclose(np.trace(cm) / 500, accuracy(y_true, y_pred))# diagonal = accuracy


def test_most_confused_pairs():
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert most_confused_pairs(cm, 1)[0][2] == 1  # every error here happens exactly once


def test_per_class_report_structure():
    report = per_class_report(Y_TRUE, Y_PRED, 3)
    assert np.isclose(report["accuracy"], 4 / 6)
    assert report["per_class"][1]["support"] == 2  # two true 1s