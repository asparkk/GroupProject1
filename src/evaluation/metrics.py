"""Evaluation metrics written from scratch with NumPy.
Every function takes y_true and y_pred, both shape (N,)
with integers 0-9 (see docs/INTERFACES.md), so it works on any model's output.
"""
import numpy as np

NUM_CLASSES = 10  # digits 0-9


def accuracy(y_true, y_pred):
    """Fraction of predictions that match the true labels (the headline number)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # Catch shape bugs early: (N,1) vs (N,) would otherwise broadcast silently
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    # (y_true == y_pred) is a boolean array; the mean of booleans is the fraction True
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    """10x10 table where cm[i, j] = images of true digit i predicted as digit j.

    The diagonal is correct predictions. Large off-diagonal entries show which
    digits get mixed up (the "commonly misclassified pairs" analysis in the plan).
    Every other metric below is derived from this matrix.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # np.add.at adds 1 at each (true, pred) position and handles repeated pairs;
    # plain cm[y_true, y_pred] += 1 would count a repeated pair only once
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_divide(num, den):
    """Elementwise num/den, returning 0 where den is 0 (avoids NaN and warnings).

    Needed because a digit the model never predicts has precision 0/0."""
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def precision_recall_f1(y_true, y_pred, num_classes=NUM_CLASSES):
    """Per-digit precision, recall and F1, each an array of length 10."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)  # true positives: correct predictions per digit
    # Precision: of everything predicted as digit j, how much was right (column sums)
    precision = _safe_divide(tp, cm.sum(axis=0).astype(np.float64))
    # Recall: of all true images of digit i, how many were found (row sums)
    recall = _safe_divide(tp, cm.sum(axis=1).astype(np.float64))
    # F1: harmonic mean of the two; low if either one is low
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(y_true, y_pred, num_classes=NUM_CLASSES):
    """Unweighted mean of the per-digit F1 scores (each digit counts equally)."""
    return float(np.mean(precision_recall_f1(y_true, y_pred, num_classes)[2]))


def per_class_report(y_true, y_pred, num_classes=NUM_CLASSES):
    """Bundle everything into one dict; this is what goes into the results JSON."""
    p, r, f = precision_recall_f1(y_true, y_pred, num_classes)
    support = confusion_matrix(y_true, y_pred, num_classes).sum(axis=1)  # images per digit
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": float(f.mean()),
        "per_class": {
            int(c): {"precision": float(p[c]), "recall": float(r[c]),
                     "f1": float(f[c]), "support": int(support[c])}
            for c in range(num_classes)
        },
    }


def format_report(report):
    """Turn a per_class_report dict into a printable table for the final report."""
    lines = [f"{'digit':>5} {'prec':>7} {'recall':>7} {'f1':>7} {'n':>6}"]
    for c, m in report["per_class"].items():
        lines.append(f"{c:>5} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} {m['support']:6d}")
    lines.append(f"accuracy={report['accuracy']:.4f}  macro_f1={report['macro_f1']:.4f}")
    return "\n".join(lines)


def most_confused_pairs(cm, top_k=5):
    """Largest off-diagonal entries as [(true, pred, count), ...].

    Answers 'which digit pairs does the model confuse most' for the error analysis."""
    off = cm.copy()
    np.fill_diagonal(off, 0)  # ignore correct predictions
    order = np.argsort(off, axis=None)[::-1][:top_k]  # flat indices, biggest first
    return [(int(i // cm.shape[1]), int(i % cm.shape[1]), int(off.flat[i]))
            for i in order if off.flat[i] > 0]