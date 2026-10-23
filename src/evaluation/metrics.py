import numpy as np

NUM_CLASSES = 10


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    """cm[i, j] = number of samples with true class i predicted as j."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_divide(num, den):
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def precision_recall_f1(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    precision = _safe_divide(tp, cm.sum(axis=0).astype(np.float64))  # per predicted class
    recall = _safe_divide(tp, cm.sum(axis=1).astype(np.float64))     # per true class
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(y_true, y_pred, num_classes=NUM_CLASSES):
    return float(np.mean(precision_recall_f1(y_true, y_pred, num_classes)[2]))


def per_class_report(y_true, y_pred, num_classes=NUM_CLASSES):
    p, r, f = precision_recall_f1(y_true, y_pred, num_classes)
    support = confusion_matrix(y_true, y_pred, num_classes).sum(axis=1)
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
    lines = [f"{'digit':>5} {'prec':>7} {'recall':>7} {'f1':>7} {'n':>6}"]
    for c, m in report["per_class"].items():
        lines.append(f"{c:>5} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} {m['support']:6d}")
    lines.append(f"accuracy={report['accuracy']:.4f}  macro_f1={report['macro_f1']:.4f}")
    return "\n".join(lines)


def most_confused_pairs(cm, top_k=5):
    """Return [(true, pred, count), ...] for the largest off-diagonal entries."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:top_k]
    return [(int(i // cm.shape[1]), int(i % cm.shape[1]), int(off.flat[i])) for i in order if off.flat[i] > 0]