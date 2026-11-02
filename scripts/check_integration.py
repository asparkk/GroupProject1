import numpy as np
from preprocess import normalize, split_train_val, make_batches, count_overlap
# also need to import B's loader here

X_raw, y_raw, X_test_raw, y_test = load_mnist()   # replace with B's real loader call
print('raw max:', X_raw.max())   # 255 means you still need normalize; 1.0 means the loader already did it

X_trainval = normalize(X_raw)
X_test = normalize(X_test_raw)
X_train, y_train, X_val, y_val = split_train_val(X_trainval, y_raw, 42)

for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
    print(name, X.shape, X.dtype, float(X.min()), float(X.max()), y.shape, y.dtype)
    assert X.ndim == 2 and X.shape[1] == 784
    assert y.shape == (X.shape[0],)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(np.unique(y)) <= set(range(10))

print('sizes:', len(X_train), len(X_val), len(X_test))   # expect 48000 12000 10000
print('overlap train/val:', count_overlap(X_train, X_val))
print('overlap train/test:', count_overlap(X_train, X_test))
print('overlap val/test:', count_overlap(X_val, X_test))

batches = make_batches(X_train, y_train, 64, 42)
print('batches:', len(batches), batches[0][0].shape, batches[0][1].shape)   # expect 750 (64, 784) (64,)