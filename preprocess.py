import numpy as np

def load_mnist(folder):
    X_train = np.fromfile(folder + "/train-images.idx3-ubyte", dtype= np.uint8, offset = 16).reshape(-1,784)
    y_train = np.fromfile(folder + "/train-labels.idx1-ubyte", dtype = np.uint8, offset =8)
    X_test = np.fromfile(folder + "/t10k-images.idx3-ubyte", dtype=np.uint8, offset=16).reshape(-1, 784)
    y_test = np.fromfile(folder + "/t10k-labels.idx1-ubyte", dtype=np.uint8, offset=8)
    return X_train,y_train, X_test, y_test
def normalize(X):
    X = np.asarray(X)
    if len (X.shape) != 2 or X.shape[1] != 784:
        raise ValueError("Value Error")
    if X.min() < 0 or X.max() > 255:
        raise ValueError("Value Error")
    return X.astype(float) / 255
def split_train_val(X,y,seed):
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X.shape) !=2:
        raise ValueError("Value Error")
    if len(y.shape) != 1:
        raise ValueError("Value Error")
    if len(X) != len(y):
        raise ValueError("Value Error")
    n = len(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(0.8 * n)
    train_rows = order[:cut]
    val_rows = order[cut:]
    return X[train_rows], y[train_rows], X[val_rows], y[val_rows]
