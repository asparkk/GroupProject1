from mlp import MLP
from scripts.common import load_data
from scripts.training import train

data = load_data()
accuracies = []
for run in range(2):
    model = MLP(layer_sizes=(784, 128, 10), seed=42)
    history, best_acc = train(model, data, epochs=2, seed=42, verbose=False)
    accuracies.append(best_acc)
print(accuracies)
print('difference between the two runs:', abs(accuracies[0] - accuracies[1]))   # expect 0.0