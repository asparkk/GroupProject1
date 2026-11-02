import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from src.evaluation.metrics import precision_recall_f1

rng = np.random.default_rng(0)
y_true = rng.integers(0, 10, 1000)
y_pred = np.where(rng.random(1000) < 0.8, y_true, rng.integers(0, 10, 1000))   # simulated, about 80% right

p, r, f = precision_recall_f1(y_true, y_pred)                                   # my code
sp, sr, sf, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(10), zero_division=0)   # scikit-learn

print('digit   my F1     scikit-learn F1')
for c in range(10):
    print(f'{c:>5}   {f[c]:.4f}    {sf[c]:.4f}') #using printing formatting to make results print nicely
print('largest difference (precision, recall, F1):', np.abs(p - sp).max(), np.abs(r - sr).max(), np.abs(f - sf).max())
print('all match:', np.allclose(p, sp) and np.allclose(r, sr) and np.allclose(f, sf))