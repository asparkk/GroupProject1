import numpy as np
from src.evaluation.gradientcheck import numerical_gradient, check_gradients


def test_quadratic():
    x = np.array([1.0, -2.0, 3.0])
    num = numerical_gradient(lambda: np.sum(x ** 2), x)   # true gradient is 2x
    assert np.allclose(num, 2 * x, atol=1e-6)
    assert np.array_equal(x, [1.0, -2.0, 3.0])            # x must be restored


def test_2d_function():
    A = np.arange(6.0).reshape(2, 3)
    num = numerical_gradient(lambda: np.sum(np.sin(A)), A)  # true gradient is cos(A)
    assert np.allclose(num, np.cos(A), atol=1e-6)


def test_detects_wrong_gradient():
    # The checker must fail when the gradient is wrong (3x instead of 2x)
    x = np.array([1.0, 2.0])
    assert not check_gradients(lambda: np.sum(x ** 2), {"x": x}, {"x": 3 * x})["passed"]


def test_softmax_cross_entropy_gradients():
    # Stand-in for A's output layer: linear layer + softmax + cross-entropy.
    # The known gradient is (probabilities - one_hot_labels) / N.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4)); y = rng.integers(0, 3, 5)
    params = {"W": rng.normal(size=(4, 3)), "b": rng.normal(size=3)}

    def probs():
        z = X @ params["W"] + params["b"]
        z = z - z.max(axis=1, keepdims=True)              # numerical stability
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    loss = lambda: float(-np.mean(np.log(probs()[np.arange(5), y])))
    d = probs(); d[np.arange(5), y] -= 1; d /= 5
    grads = {"W": X.T @ d, "b": d.sum(axis=0)}
    assert check_gradients(loss, params, grads)["passed"]