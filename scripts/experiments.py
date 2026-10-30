from pathlib import Path
from mlp import MLP
from scripts.common import load_data
from scripts.training import train
from src.evaluation.benchmark import time_training, time_inference
from src.evaluation.metrics import per_class_report
from src.evaluation.results import make_result, save_result, load_result, compare_results

SEED = 42
EPOCHS = 10 # set to 1 first as a smoke test

# each run is learning rate, hidden size, batch size, the first one is the baseline
RUNS = [
    (0.1, 128, 64),
    (0.01, 128, 64),
    (0.05, 128, 64),
    (0.2, 128, 64),
    (0.1, 64, 64),
    (0.1, 256, 64),
    (0.1, 128, 32),
    (0.1, 128, 128),
]

data = load_data(SEED)
Path('results/models').mkdir(parents=True, exist_ok=True)

for lr, hidden, batch_size in RUNS:
    name = 'lr' + str(lr) + '_h' + str(hidden) + '_bs' + str(batch_size)
    print('running', name)

    model = MLP(layer_sizes=(784, hidden, 10), seed=SEED)
    out, seconds = time_training(train, model, data, epochs=EPOCHS, lr=lr, batch_size=batch_size, seed=SEED)
    history, best_acc = out

    predictions = model.predict(data['X_val']) # validation data only
    report = per_class_report(data['y_val'], predictions)
    speed = time_inference(model.predict, data['X_val'], n_runs=3)

    model.save('results/models/' + name + '.npz') # saves the best checkpoint

    hyperparams = {'lr': lr, 'hidden': hidden, 'batch_size': batch_size, 'epochs': EPOCHS, 'seed': SEED}
    metrics = {'val_accuracy': report['accuracy'], 'val_macro_f1': report['macro_f1'], 'best_val_acc': best_acc}
    timing = {'train_seconds': seconds, 'inference_ms_per_image': speed['ms_per_image']}
    save_result(make_result(name, hyperparams, history, metrics, timing))

results = []
for path in Path('results').glob('lr*.json'):
    results.append(load_result(path))
print(compare_results(results)) # creating the ranked table, best validation accuracy first