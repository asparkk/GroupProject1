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
def make_batches(X,y, batch_size,seed):
    X= np.asarray(X)
    y= np.asarray(y)
    if len(X.shape) != 2:
        raise ValueError("Value Error")
    if len(y.shape) != 1:
        raise ValueError("Value Error")
    if len(X)!= len(y): 
        raise ValueError("Value Error")
    if isinstance(batch_size, bool):
        raise ValueError("Value Error")
    if not isinstance(batch_size, (int,np.integer)):
        raise ValueError("Value Error")
    if batch_size < 1:
        raise ValueError("Value Error")
    n = len(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    batches = []
    for start in range(0, n, batch_size):
        rows = order[start:start + batch_size]
        batches.append((X[rows],y[rows]))
    return batches
def count_overlap(A, B):
    A = np.asarray(A)
    B = np.asarray(B)

    if len(A.shape) != 2:
        raise ValueError("Value Error")
    if len(B.shape) != 2:
        raise ValueError("Value Error")
    if A.shape[1] != B.shape[1]:
        raise ValueError("Value Error")
    seen = set()
    for i in range(len(B)):
        seen.add(B[i].tobytes())
    count = 0
    for i in range(len(A)):
        if A[i].tobytes() in seen:
            count = count + 1
    return count
def prepare_data(folder, seed):
    X_train_full, y_train_full, X_test, y_test = load_mnist(folder)
    X_train_full = normalize(X_train_full)
    X_test = normalize(X_test)
    X_train, y_train, X_val, y_val = split_train_val(X_train_full,y_train_full, seed)
    y_train - y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
