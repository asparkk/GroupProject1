"""Quick checks for mlp.py using random data. Run: pytest -q"""

import numpy as np

from mlp import MLP, softmax


def batch(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 784)).astype(np.float32), rng.integers(0, 10, n)


def test_shapes():
    model = MLP()
    X, _ = batch()
    assert model.params["W1"].shape == (784, 256)
    assert model.params["W2"].shape == (256, 128)
    assert model.params["W3"].shape == (128, 10)
    assert model.forward(X).shape == (16, 10)
    assert model.predict(X).shape == (16,)


def test_softmax_rows_sum_to_one_and_is_stable():
    p = softmax(np.array([[1000.0, 1001.0, 1002.0], [0.0, 0.0, 0.0]]))
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_initial_loss_near_ln10():
    X, y = batch()
    assert 1.8 < MLP().loss(X, y) < 3.0


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    model = MLP(layer_sizes=(6, 5, 4, 3), seed=1, dtype=np.float64)
    X, y = rng.normal(size=(8, 6)), rng.integers(0, 3, 8)
    grads = model.gradients(X, y)
    eps = 1e-5
    for k, param in model.params.items():
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + eps; up = model.loss(X, y)
            param[idx] = old - eps; down = model.loss(X, y)
            param[idx] = old
            numeric = (up - down) / (2 * eps)
            assert abs(grads[k][idx] - numeric) <= 1e-6 * max(1.0, abs(numeric)), k


def test_step_reduces_loss():
    model = MLP()
    X, y = batch(64)
    before = model.loss(X, y)
    model.step(model.gradients(X, y), lr=0.05)
    assert model.loss(X, y) < before


def test_save_load(tmp_path):
    model = MLP(seed=3)
    X, _ = batch()
    model.save(tmp_path / "m.npz")
    np.testing.assert_array_equal(MLP.load(tmp_path / "m.npz").forward(X), model.forward(X))
