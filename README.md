# GroupProject1

# Members:
- Aaron, Saylen, and Bernisha

# Handwritten Digit Recognition with a From-Scratch NumPy Neural Network

A multilayer perceptron that recognizes handwritten digits (0 to 9) from the MNIST dataset, built with NumPy only: ReLU hidden layers, softmax output, cross-entropy loss, hand-derived backpropagation and mini-batch SGD. No PyTorch or TensorFlow.


## Results

| Metric | Value |
| --- | --- |
| Test accuracy | 97.54% |
| Test macro F1 | 0.9753 |
| Selected model | one hidden layer of 128 units, learning rate 0.1, batch size 32 |
| Validation accuracy of selected model | 97.62% |
| Gradient check (largest relative error) | below 3e-7 |
| Unit tests | 40 passing |


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt
```

`requirements.txt` should list numpy, matplotlib, pytest, scikit-learn and `pandas<3`. MNIST downloads automatically through scikit-learn the first time a script runs, and is cached afterwards.

**Troubleshooting**

- *SSL certificate error on macOS:* run `Install Certificates.command` from your Python folder in Applications, then retry.
- *Error mentioning pandas while loading MNIST:* use `pip install "pandas<3"`.
- *`No module named 'scripts'`:* run commands from the repository root with `python -m scripts.<name>`.

## Run

Run everything from the repository root with the virtual environment active.

```bash
python -m pytest -q                       # all unit tests
python -m scripts.gradientcheck_mlp       # gradient check on the network
python -m scripts.experiments             # hyperparameter experiments (slowest step)
python -m scripts.final_evaluation        # picks the best model on validation data, then evaluates on the test set once
python -m scripts.confusable_digits       # analysis of digits that look alike
python -m scripts.stretch_comparison      # extra hidden layer and momentum comparison
python -m scripts.check_reproducible      # same seed gives the same result
```

Results are written to `results/`, which is not tracked by git. Figures used in the report are in `docs/figures/`.

## Repository layout

```
mlp.py                      neural network (Person A)
preprocess.py               data preparation and batching (Person B)
check_pipeline.py           data pipeline checks (Person B)
preproccessing.md           data pipeline documentation (Person B)
test_mlp.py                 network tests (Person A)
test_preprocess.py          data tests (Person B)
test_batches.py             mini-batch tests (Person B)
test_overlap.py             data leakage tests (Person B)
src/evaluation/             metrics, gradient checking, plots, benchmarks, results format (Person C)
tests/evaluation/           tests for the evaluation toolkit (Person C)
tests/test_nn_extra.py      extra network tests: ReLU, cross-entropy, SGD update (Person C)
scripts/                    data loading, training loop, experiments, final evaluation, analyses (Person C)
docs/INTERFACES.md          shared data formats agreed by the team
docs/figures/               figures used in the report
report.md                   full written report
```

## Who did what

| Person | Contribution |
| --- | --- |
| Person A | Neural network: initialization, forward pass, loss, backpropagation, SGD step, saving and loading |
| Person B | Data pipeline: normalization, flattening, train/validation split, mini-batches, overlap checks |
| Person C | Evaluation and testing toolkit, plus the scripts that connect the parts: training loop, experiments, final evaluation, and analyses |

