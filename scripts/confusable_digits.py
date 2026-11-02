import numpy as np
import matplotlib
matplotlib.use('Agg')# draw to files instead of opening windows
import matplotlib.pyplot as plt
from pathlib import Path
from mlp import MLP
from scripts.common import load_data
from src.evaluation.metrics import confusion_matrix, most_confused_pairs
from src.evaluation.results import load_result

Path('docs/figures').mkdir(parents=True, exist_ok=True)

# Load the model that was selected earlier by validation accuracy, and the validation data
results = []
for path in Path('results').glob('lr*.json'):
    results.append(load_result(path))
best = results[0]
for r in results:
    if r['metrics']['best_val_acc'] > best['metrics']['best_val_acc']:
        best = r
model = MLP.load('results/models/' + best['name'] + '.npz')
data = load_data(best['hyperparams']['seed'])
X = data['X_val']
y = data['y_val']

probs = model.forward(X) # shape (12000, 10) => the network's probability for every digit
pred = probs.argmax(axis=1)# the digit it picks

#Finding the most confused pair in the way that true digit a gets mistaken for digit b
cm = confusion_matrix(y, pred)
pairs = most_confused_pairs(cm, 5)
print('most confused (true, predicted, count):', pairs)
a, b, count = pairs[0]
print('focusing on the true', a, 'predicted as', b, 'pair')

# Average image of each digit, and where they differ. This will show us why they look alike
mean_a = X[y == a].mean(axis=0).reshape(28, 28)
mean_b = X[y == b].mean(axis=0).reshape(28, 28)
fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
axes[0].imshow(mean_a, cmap='gray')
axes[0].set_title('average ' + str(a))
axes[1].imshow(mean_b, cmap='gray')
axes[1].set_title('average ' + str(b))
axes[2].imshow(mean_a - mean_b, cmap='bwr', vmin=-0.6, vmax=0.6)
axes[2].set_title('red: more ink in ' + str(a) + ', blue: more ink in ' + str(b))
for ax in axes:
    ax.axis('off')
fig.tight_layout()
fig.savefig('docs/figures/similar_pair_average.png', dpi=150)

#Real mistakes with the full probability output, this will show us how the network reacts
wrong = np.flatnonzero((y == a) & (pred == b))
show = wrong[:5]
fig, axes = plt.subplots(2, len(show), figsize=(3 * len(show), 5), squeeze=False)
for k in range(len(show)):
    i = show[k]
    axes[0][k].imshow(X[i].reshape(28, 28), cmap='gray')
    axes[0][k].set_title('true ' + str(y[i]) + ', predicted ' + str(pred[i]))
    axes[0][k].axis('off')
    axes[1][k].bar(range(10), probs[i])
    axes[1][k].set_xticks(range(10))
    axes[1][k].set_ylim(0, 1)
    axes[1][k].set_xlabel('digit')
    axes[1][k].set_ylabel('probability')
    print('example', k + 1, ': probability of true digit', a, '=', round(float(probs[i][a]), 3),
          ', probability of predicted digit', b, '=', round(float(probs[i][b]), 3))
fig.tight_layout()
fig.savefig('docs/figures/similar_pair_examples.png', dpi=150)

#Soft confusion: average probability given to each digit, for each true digit
avg = np.zeros((10, 10))
for d in range(10):
    avg[d] = probs[y == d].mean(axis=0)
items = []
for i in range(10):
    for j in range(10):
        if i != j:
            items.append((avg[i][j], i, j))
items.sort(reverse=True)
print('digits the network treats as closest, even when it is right:')
for value, i, j in items[:5]:
    print('  on true', i, 'images it gives digit', j, 'an average probability of', round(float(value), 4))

off = avg.copy()
for d in range(10):
    off[d][d] = 0 # hide the diagonal so the small values are visible
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(off, cmap='Blues')
fig.colorbar(im, ax=ax)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('digit the network gives probability to')
ax.set_ylabel('true digit')
ax.set_title('Average probability on a wrong digit')
fig.tight_layout()
fig.savefig('docs/figures/soft_confusion.png', dpi=150)

# Seeing if the network is less sure when it is wrong
confidence = probs.max(axis=1)
print('average confidence when right:', round(float(confidence[pred == y].mean()), 3))
print('average confidence when wrong: ', round(float(confidence[pred != y].mean()), 3))