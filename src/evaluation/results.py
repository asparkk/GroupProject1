import json
from pathlib import Path
import numpy as np


def _to_jsonable(o):
    #since json can't store NumPy types, here I am converting arrays and Numpy numbers to 
    #plain Python types so that they can be serialized to JSON format
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"Not serializable: {type(o)}")


def make_result(name, hyperparams, history, metrics, timing):
    """One experiment as a dict. Example name: 'lr0.1_h128_bs64'."""
    #This function will create a dictionary that contains the results of an experiment, 
    #including the name of the experiment, hyperparameters used, training history, evaluation metrics, and timing information
    return {"name": name, "hyperparams": hyperparams, "history": history,
            "metrics": metrics, "timing": timing}


def save_result(result, directory="results"):
    #Save the result dictionary as a JSON file in the specified directory. The filename will be based on the expiriment name
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / f"{result['name']}.json"
    path.write_text(json.dumps(result, indent=2, default=_to_jsonable))
    return path


def load_result(path):
    return json.loads(Path(path).read_text())


def compare_results(results):
    #This function takes a list of result dictionaries and returns a string representation of a table that compares 
    #the experiments based on their best validation accuracy and final validation loss. The table is then sorted in descending order of best validation accuracy
    #with the experiment name, best validation accuracy, and final validation loss displayed in each row. 
    rows = sorted(results, key=lambda r: -max(r["history"]["val_acc"]))
    lines = [f"{'name':<28} {'best val acc':>12} {'final val loss':>14}"]
    for r in rows:
        lines.append(f"{r['name']:<28} {max(r['history']['val_acc']):>12.4f} {r['history']['val_loss'][-1]:>14.4f}")
    return "\n".join(lines)