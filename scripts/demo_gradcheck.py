import numpy as np
from src.evaluation.gradientcheck import check_gradients

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 4)); y = rng.integers(0, 3, 5)
params = {'W': rng.normal(size=(4, 3)), 'b': rng.normal(size=3)}

def probs():
    z = X @ params['W'] + params['b']
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

loss = lambda: float(-np.mean(np.log(probs()[np.arange(5), y])))
d = probs(); d[np.arange(5), y] -= 1; d /= 5  # analytical gradient
good = {'W': X.T @ d, 'b': d.sum(axis=0)}
bad = {'W': 1.1 * good['W'], 'b': good['b']}   # deliberately wrong

for label, grads in [('correct gradient', good), ('wrong gradient', bad)]:
    res = check_gradients(loss, params, grads)
    print(label, '->', 'PASSED' if res['passed'] else 'FAILED')
    for name, err in res['errors'].items():
        print(f'    {name}: relative error {err:.2e}')