from preprocess import make_batches
from src.evaluation.metrics import accuracy


def train(model, data, epochs=20, lr=0.1, batch_size=64, seed=42, verbose=True):
    """Mini-batch SGD. Returns (history, best_val_acc).

    history has one entry per epoch (train_loss, val_loss, train_acc, val_acc).
    When training ends, the model holds the checkpoint with the best VALIDATION
    accuracy, which is how we should select the models and never with the test set."""
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc, best_params = -1.0, None
    for epoch in range(epochs):
        # Since person B's portions make_batches shuffles with the seed; a new seed each epoch gives a new order each pass
        for X_b, y_b in make_batches(X_train, y_train, batch_size, seed + epoch):
            model.step(model.gradients(X_b, y_b), lr)             # Saylens backprop, then one SGD update
        sub = slice(0, 10000)                                     # score a 10,000-image slice of train to keep epochs fast
        hist['train_loss'].append(model.loss(X_train[sub], y_train[sub]))
        hist['train_acc'].append(accuracy(y_train[sub], model.predict(X_train[sub])))
        hist['val_loss'].append(model.loss(X_val, y_val))
        hist['val_acc'].append(accuracy(y_val, model.predict(X_val)))
        if hist['val_acc'][-1] > best_acc:                        # remember the best checkpoint so far
            best_acc = hist['val_acc'][-1]
            best_params = {k: v.copy() for k, v in model.params.items()}
        if verbose:
            va, vl = hist['val_acc'][-1], hist['val_loss'][-1]
            print(f'  epoch {epoch + 1}/{epochs}: val acc {va:.4f}, val loss {vl:.4f}')
    model.params = best_params                                    # restore the best checkpoint
    return hist, best_acc