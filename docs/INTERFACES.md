# Shared Interfaces

| Name | Type | Shape / format |
|---|---|---|
| X (images) | float64 NumPy array | (N, 784), values in [0, 1] |
| y (labels) | int NumPy array | (N,), values 0-9 |
| y_proba | float NumPy array | (N, 10), each row sums to 1 |
| y_pred | int NumPy array | (N,), values 0-9 |
| history | dict of lists | keys: train_loss, val_loss, train_acc, val_acc; one entry per epoch |

## Gradient checking contract
- params: dict[str, float64 ndarray], e.g. W1, b1, W2, b2
- grads: dict with the same keys and shapes as params
- loss_fn(): takes no arguments, returns a float computed from the current contents of params
