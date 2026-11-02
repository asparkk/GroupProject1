import numpy as np
from mlp import MLP, relu


def test_relu_values():
    z = np.array([-2.0, 0.0, 3.5])
    assert np.array_equal(relu(z), np.array([0.0, 0.0, 3.5])) # negatives and zero become 0


def test_cross_entropy_hand_computed():
    # one layer with zero weights, so the scores are just the biases [0, ln 2, 0]
    model = MLP(layer_sizes=(2, 3), seed=0, dtype=np.float64)
    model.params['W1'] = np.zeros((2, 3))
    model.params['b1'] = np.array([0.0, np.log(2.0), 0.0])
    X = np.zeros((1, 2))
    # probabilities are [1, 2, 1] / 4 = [0.25, 0.5, 0.25]
    assert np.isclose(model.loss(X, np.array([1])), -np.log(0.5))# true class 1: ln 2 = 0.6931
    assert np.isclose(model.loss(X, np.array([0])), -np.log(0.25))# true class 0: ln 4 = 1.3863


def test_step_applies_exact_update():
    model = MLP(layer_sizes=(2, 3), seed=0, dtype=np.float64)
    before_W = model.params['W1'].copy()
    before_b = model.params['b1'].copy()
    grads = {'W1': np.ones((2, 3)), 'b1': np.ones(3)}
    model.step(grads, lr=0.5)# each weight should drop by 0.5
    assert np.allclose(model.params['W1'], before_W - 0.5)
    assert np.allclose(model.params['b1'], before_b - 0.5)