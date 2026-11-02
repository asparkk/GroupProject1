import numpy as np
from mlp import MLP
from src.evaluation.gradientcheck import check_gradients
from scripts.common import load_data

# The tiny network, every weight is checked (float64)
rng = np.random.default_rng(1)
tiny = MLP(layer_sizes=(6, 5, 4, 3), seed=1, dtype=np.float64)
X = rng.normal(size=(8, 6))
y = rng.integers(0, 3, 8)


def tiny_loss():
    return tiny.loss(X, y)


result = check_gradients(tiny_loss, tiny.params, tiny.gradients(X, y))
print('tiny network passed:', result['passed'])
for name in result['errors']:
    print(name, 'relative error:', result['errors'][name])

# The real 784-256-128-10 network on 8 real digits.
# Because checking all ~235,000 weights would take hours, here we check 25 random ones per parameter.
data = load_data()
X = data['X_val'][:8].astype(np.float64)
y = data['y_val'][:8]
big = MLP(seed=0, dtype=np.float64)
grads = big.gradients(X, y)#Using person A's backprop gradients
picker = np.random.default_rng(0)
eps = 1e-5

print('real network:')
for name in big.params:
    p = big.params[name]
    worst = 0.0
    for trial in range(25):
        row = picker.integers(0, p.shape[0])
        if p.ndim == 2:
            col = picker.integers(0, p.shape[1])
            index = (row, col)
        else:
            index = (row,)

        old = p[index]
        p[index] = old + eps
        up = big.loss(X, y)
        p[index] = old - eps
        down = big.loss(X, y)
        p[index] = old #put the weight back

        numeric = (up - down) / (2 * eps) # slope measured by nudging the weight
        analytic = grads[name][index] # slope from backprop
        error = abs(analytic - numeric) / max(abs(analytic) + abs(numeric), 1e-8)
        if error > worst:
            worst = error
    print(name, 'worst relative error:', worst)