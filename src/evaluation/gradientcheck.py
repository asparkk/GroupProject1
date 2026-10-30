import numpy as np


def numerical_gradient(f, x, eps=1e-5):
    #In this function we are estimating the gradient of a function f with 
    #with respect to a variable x using central differences. The function f is a zero-argument function that 
    #reads x from its closure, so editing x in place changes f's results. The function returns the estimated gradient as a NumPy array
    #of the same shape as x. The function uses np.nditer to iterate over every element of x and compute the gradient for each element by nudging 
    #it up and down by eps and computing the differences in f's output. The original value of x is restored after each computation.
    """Estimate df/dx with central differences: (f(x+eps) - f(x-eps)) / (2*eps).

    f is a zero-argument function that reads x from its closure, so editing x in
    place changes f's result. x is restored afterwards. In the context of this project
     f is A's loss, x is one weight matrix such as W1.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])  # visit every element
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps; f_plus = f() # loss with this one weight nudged up
        x[idx] = orig - eps; f_minus = f() # loss with it nudged down
        x[idx] = orig # always restore the original value
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b, floor=1e-8):
    """Worst-case relative difference between two gradient arrays.

    Relative error, so tiny and huge gradients are judged fairly.
    When we are intepretting results, below 1e-7 is great, above 1e-4 means a bug."""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), floor)))


def check_gradients(loss_fn, params, grads, eps=1e-5, tol=1e-6):
    """Function to compare analytical gradients from backpropogation done in the nueral network
     with numerical ones, per parameter. Params and gradients are dicts with the same keys (W1, b1, W2, b2), as agreed in
    INTERFACES.md. Returns the error for each key, so if only W1 fails you know
    the hidden-layer backprop is the problem."""
    errors = {}
    for name, p in params.items():
        # float32 rounding noise would swamp the tiny eps nudges and give false alarms
        assert p.dtype == np.float64, f"{name} must be float64 for gradient checking"
        num = numerical_gradient(loss_fn, p, eps)
        errors[name] = relative_error(grads[name], num)
    return {"errors": errors, "passed": all(e < tol for e in errors.values())}