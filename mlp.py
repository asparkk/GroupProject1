"""From-scratch NumPy MLP for MNIST: 784 -> 256 -> 128 -> 10.

Shared interface
----------------
Inputs:  X  float array (N, 784), pixel values scaled to [0, 1]
         y  int array   (N,),     digit labels 0-9
Outputs: model.forward(X)   -> probabilities (N, 10), each row sums to 1
         model.predict(X)   -> predicted digits (N,)
         model.loss(X, y)   -> mean cross-entropy (float)
         model.gradients(X, y) -> dict of gradients, same keys/shapes as model.params
         model.step(grads, lr) -> one SGD update
         model.params       -> {"W1", "b1", "W2", "b2", "W3", "b3"}
         model.save(path) / MLP.load(path)

Example training step:
    model = MLP(seed=0)
    grads = model.gradients(X_batch, y_batch)
    model.step(grads, lr=0.1)
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    # Subtract each row's max so exp() can't overflow; the result is unchanged.
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z):
    # log(softmax(z)) computed directly, so it never takes log(0).
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


class MLP:
    def __init__(self, layer_sizes=(784, 256, 128, 10), seed=0, dtype=np.float32):
        self.layer_sizes = list(layer_sizes)
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        self.params = {}
        for l in range(1, len(self.layer_sizes)):
            n_in, n_out = self.layer_sizes[l - 1], self.layer_sizes[l]
            # He initialization: spread sqrt(2 / inputs), suited to ReLU.
            self.params[f"W{l}"] = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)).astype(dtype)
            self.params[f"b{l}"] = np.zeros(n_out, dtype=dtype)
        self.cache = {}

    @property
    def num_layers(self):
        return len(self.layer_sizes) - 1

    def forward(self, X):
        """Return class probabilities (N, 10). Saves intermediate values for backprop."""
        a = np.asarray(X, dtype=self.dtype)
        if a.ndim != 2 or a.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected X of shape (N, {self.layer_sizes[0]}), got {a.shape}")
        self.cache = {"a0": a}
        L = self.num_layers
        for l in range(1, L + 1):
            z = a @ self.params[f"W{l}"] + self.params[f"b{l}"]
            self.cache[f"z{l}"] = z
            if l < L:                      # hidden layers use ReLU
                a = relu(z)
                self.cache[f"a{l}"] = a
        self.cache["probs"] = softmax(z)   # output layer uses softmax
        return self.cache["probs"]

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def loss(self, X, y):
        """Mean cross-entropy, computed from the raw scores for stability."""
        self.forward(X)
        logits = self.cache[f"z{self.num_layers}"]
        return float(-log_softmax(logits)[np.arange(len(y)), y].mean())

    def gradients(self, X, y):
        """Backprop. Returns a dict with the same keys and shapes as self.params."""
        probs = self.forward(X)
        c, p, n = self.cache, self.params, len(y)
        dz = (probs - one_hot(y, self.layer_sizes[-1])) / n   # softmax + cross-entropy
        grads = {}
        for l in range(self.num_layers, 0, -1):
            grads[f"W{l}"] = (c[f"a{l-1}"].T @ dz).astype(self.dtype)
            grads[f"b{l}"] = dz.sum(axis=0).astype(self.dtype)
            if l > 1:
                dz = (dz @ p[f"W{l}"].T) * (c[f"z{l-1}"] > 0)  # back through ReLU
        return grads

    def step(self, grads, lr):
        """Plain SGD update: param -= lr * gradient."""
        for k in self.params:
            self.params[k] -= (lr * grads[k]).astype(self.dtype)

    def save(self, path):
        np.savez(path, layer_sizes=np.array(self.layer_sizes), **self.params)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            model = cls(layer_sizes=data["layer_sizes"].tolist())
            model.params = {k: data[k].copy() for k in data.files if k != "layer_sizes"}
        return model
