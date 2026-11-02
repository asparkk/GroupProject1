import numpy as np
from src.evaluation.results import make_result, save_result, load_result, compare_results


def _res(name, acc):
    h = {"train_loss": [1.0], "val_loss": [0.9], "train_acc": [0.7], "val_acc": [acc]}
    return make_result(name, {"lr": np.float64(0.1)}, h, {"accuracy": np.float64(acc)}, {})


def test_round_trip(tmp_path):
    loaded = load_result(save_result(_res("a", 0.8), tmp_path))  # save, then read back
    assert loaded["name"] == "a" and loaded["hyperparams"]["lr"] == 0.1


def test_compare_orders_best_first():
    table = compare_results([_res("low", 0.5), _res("high", 0.9)])
    assert table.index("high") < table.index("low")