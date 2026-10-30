import numpy as np
from mlp import MLP
from scripts.common import load_data
from scripts.training import train
from src.evaluation.benchmark import time_training
from src.evaluation.metrics import per_class_report
from src.evaluation.results import make_result, save_result, compare_results

SEED = 42
EPOCHS = 10


class MomentumMLP(MLP):
    """Using the same network, but SGD with momentum so each update remembers the previous direction."""

    def __init__(self, layer_sizes=(784, 128, 10), seed=0, dtype=np.float32, beta=0.9):
        MLP.__init__(self, layer_sizes, seed, dtype)
        self.beta = beta
        self.velocity = {}
        for name in self.params:
            self.velocity[name] = np.zeros_like(self.params[name])

    def step(self, grads, lr):
        for name in self.params:
            self.velocity[name] = self.beta * self.velocity[name] - lr * grads[name]
            self.params[name] += self.velocity[name].astype(self.dtype)


# Defining each entry in the format: name, model, learning rate. Momentum uses a smaller rate because it builds up speed.
CONFIGS = [
    ('stretch_baseline_1_hidden', MLP(layer_sizes=(784, 128, 10), seed=SEED), 0.1),
    ('stretch_2_hidden_layers', MLP(layer_sizes=(784, 256, 128, 10), seed=SEED), 0.1),
    ('stretch_momentum', MomentumMLP(layer_sizes=(784, 128, 10), seed=SEED), 0.02),
]

data = load_data(SEED)
results = []
for name, model, lr in CONFIGS:
    print('running', name)
    out, seconds = time_training(train, model, data, epochs=EPOCHS, lr=lr, batch_size=64, seed=SEED)
    history, best_acc = out
    report = per_class_report(data['y_val'], model.predict(data['X_val']))   # validation data only
    hyperparams = {'lr': lr, 'batch_size': 64, 'epochs': EPOCHS, 'seed': SEED, 'layers': model.layer_sizes}
    metrics = {'val_accuracy': report['accuracy'], 'val_macro_f1': report['macro_f1'], 'best_val_acc': best_acc}
    result = make_result(name, hyperparams, history, metrics, {'train_seconds': seconds})
    save_result(result)
    results.append(result)

print(compare_results(results))