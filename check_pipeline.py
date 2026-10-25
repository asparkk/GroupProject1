from preprocess import prepare_data, count_overlap

folder = "../project/data/mnist"
seed = 42

data = prepare_data(folder, seed)

assert data["X_train"].shape == (48000, 784)
assert data["X_val"].shape == (12000, 784)
assert data["X_test"].shape == (10000, 784)
assert data["y_train"].shape == (48000,)
assert data["y_val"].shape == (12000,)
assert data["y_test"].shape == (10000,)

for name in ["X_train", "X_val", "X_test"]:
    assert data[name].min() >= 0.0
    assert data[name].max() <= 1.0

for name in ["y_train", "y_val", "y_test"]:
    assert data[name].min() >= 0
    assert data[name].max() <= 9

assert count_overlap(data["X_train"], data["X_val"]) == 0
assert count_overlap(data["X_train"], data["X_test"]) == 0
assert count_overlap(data["X_val"], data["X_test"]) == 0

again = prepare_data(folder, seed)
for name in data:
    assert (data[name] == again[name]).all()

print("All pipeline checks passed")
