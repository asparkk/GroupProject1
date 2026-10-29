import numpy as np
from pathlib import Path
from src.evaluation.metrics import confusion_matrix, per_class_report, format_report
from src.evaluation.plots import plot_confusion_matrix, plot_learning_curves, plot_misclassified

Path('results').mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# SAMPLE DATA ONLY: simulated predictions that are right about 90% of the time
y_true = rng.integers(0, 10, 2000)
y_pred = np.where(rng.random(2000) < 0.9, y_true, rng.integers(0, 10, 2000))
print(format_report(per_class_report(y_true, y_pred)))
plot_confusion_matrix(confusion_matrix(y_true, y_pred), save_path='results/demo_confusion.png')

# SAMPLE DATA ONLY: made-up training history
e = np.arange(1, 21)
history = {'train_loss': list(2 * np.exp(-e / 5)), 'val_loss': list(2 * np.exp(-e / 6) + 0.01 * e),
           'train_acc': list(1 - np.exp(-e / 5)), 'val_acc': list(1 - np.exp(-e / 6))}
plot_learning_curves(history, 'results/demo_curves.png')

# Layout demo only: random pixels. Swap in Person B's real MNIST images if you have them loaded.
X = rng.random((2000, 784))
plot_misclassified(X, y_true, y_pred, save_path='results/demo_misclassified.png')