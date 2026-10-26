import numpy as np
import matplotlib.pyplot as plt


def _finish(fig, save_path):
    #a shared ending for all of the plotting functions, making sure the layout is tidy and saving the figure if a path was given
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_learning_curves(history, save_path=None):
    """Train vs validation loss and accuracy per epoch.

    history is A's dict (train_loss, val_loss, train_acc, val_acc; one entry per epoch).
    If validation loss rises while training loss keeps falling, the model is overfitting."""
    #This function is used to visualize the training and validation loss and accuracy over a specfic number of epochs.
    # we take the history dictionary and plot the training and validation loss and accuracy on two subplots.
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy"); ax2.legend()
    return _finish(fig, save_path)


def plot_confusion_matrix(cm, normalize=True, save_path=None):
    #Fucntion to plot the confusion matrix. It takes in the confusion matrix, a boolean to indicate whether to normalize the values,
    # as well as an optional save path to save the figure. it returns the figure object.
    """Heatmap of the confusion matrix with a number in every cell.

    normalize=True divides each row by its total, so cells show the fraction of each
    true digit sent to each prediction (fair across digits with different counts)."""
    data = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(n), yticks=range(n), xlabel="Predicted", ylabel="True",
           title="Confusion matrix" + (" (row-normalized)" if normalize else ""))
    for i in range(n):
        for j in range(n):
            val = f"{data[i, j]:.2f}" if normalize else str(int(data[i, j]))
            ax.text(j, i, val, ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > data.max() / 2 else "black")  # readable on dark cells
    return _finish(fig, save_path)


def plot_examples(X, y_true, y_pred, indices, title="Examples", save_path=None, cols=5):
    """Grid of digit images labeled 'true / pred'. X is (N, 784), so reshape to 28x28."""
    indices = list(indices)
    rows = max(1, int(np.ceil(len(indices) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")  # hide empty slots in the grid
    for ax, i in zip(axes.ravel(), indices):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"true {y_true[i]} / pred {y_pred[i]}", fontsize=9)
    fig.suptitle(title)
    return _finish(fig, save_path)


def plot_misclassified(X, y_true, y_pred, n=20, seed=0, save_path=None):
    """Random sample of the model's mistakes (the 'visualize incorrect images' item)."""
    wrong = np.flatnonzero(y_true != y_pred)  # indices where the prediction was wrong
    rng = np.random.default_rng(seed)         # fixed seed => same sample every run
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False) if len(wrong) else []
    return plot_examples(X, y_true, y_pred, pick, "Misclassified digits", save_path)