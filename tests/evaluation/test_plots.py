import numpy as np
from src.evaluation.plots import plot_learning_curves, plot_confusion_matrix, plot_misclassified
from src.evaluation.metrics import confusion_matrix


def test_learning_curves_save(tmp_path):  # tmp_path is a throwaway folder pytest provides
    e = np.arange(10)
    h = {"train_loss": list(np.exp(-e / 3)), "val_loss": list(np.exp(-e / 4)),
         "train_acc": list(1 - np.exp(-e / 3)), "val_acc": list(1 - np.exp(-e / 4))}
    plot_learning_curves(h, tmp_path / "curves.png")
    assert (tmp_path / "curves.png").exists()


def test_confusion_and_misclassified(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 100); pred = rng.integers(0, 10, 100)
    X = rng.random((100, 784))  # fake images, so no MNIST needed
    plot_confusion_matrix(confusion_matrix(y, pred), save_path=tmp_path / "cm.png")
    plot_misclassified(X, y, pred, save_path=tmp_path / "wrong.png")
    assert (tmp_path / "cm.png").exists() and (tmp_path / "wrong.png").exists()


def test_misclassified_handles_no_errors():
    y = np.arange(10); X = np.random.rand(10, 784)
    plot_misclassified(X, y, y)  # a perfect model has no mistakes to show so it should not crash