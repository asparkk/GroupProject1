import numpy as np
from pathlib import Path
from mlp import MLP
from scripts.common import load_data
from src.evaluation.metrics import confusion_matrix, per_class_report, format_report, most_confused_pairs
from src.evaluation.plots import plot_learning_curves, plot_confusion_matrix, plot_misclassified
from src.evaluation.benchmark import time_inference, time_single_image
from src.evaluation.results import load_result, compare_results, make_result, save_result

# Choosing the best model using the validation accuracy only
results = []
for path in Path('results').glob('lr*.json'):
    results.append(load_result(path))

best = results[0]
for r in results:
    if r['metrics']['best_val_acc'] > best['metrics']['best_val_acc']:
        best = r

name = best['name']
print('selected:', name)
print(compare_results(results))

#Loading the saved model, and checking that loading it twice gives identical predictions
model_path = 'results/models/' + name + '.npz'
model = MLP.load(model_path)
model_again = MLP.load(model_path)
data = load_data(best['hyperparams']['seed'])
same = np.array_equal(model.predict(data['X_val']), model_again.predict(data['X_val']))
print('save/load gives identical predictions:', same)

# This is the final evaluation using the test set, used exactly once
X_test = data['X_test']
y_test = data['y_test']
y_pred = model.predict(X_test)
report = per_class_report(y_test, y_pred)
print(format_report(report))
cm = confusion_matrix(y_test, y_pred)
print('most confused (true, predicted, count):', most_confused_pairs(cm, 5))

# Creating the figures
plot_learning_curves(best['history'], 'results/final_curves.png')
plot_confusion_matrix(cm, save_path='results/final_confusion.png')
plot_misclassified(X_test, y_test, y_pred, n=20, save_path='results/final_misclassified.png')

# Preforming timing metrics both for the whole test set at once and for one image at a time
batch_speed = time_inference(model.predict, X_test)
single_speed = time_single_image(model.predict, X_test)
print('inference, whole test set at once:', batch_speed)
print('inference, one image at a time:', single_speed)
print('training seconds:', best['timing']['train_seconds'])

# Saving the final result
metrics = {'test_accuracy': report['accuracy'], 'test_macro_f1': report['macro_f1']}
timing = {'train_seconds': best['timing']['train_seconds'], 'inference_ms_per_image': batch_speed['ms_per_image']}
save_result(make_result('FINAL_' + name, best['hyperparams'], best['history'], metrics, timing))