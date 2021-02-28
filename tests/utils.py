import numpy as np
import unittest
from dezero import Function, Variable, as_array, as_variable


def gradient_check(f, x, *args, rtol=1e-4, atol=1e-5, **kwards):
    """Test backward procedure of a given function.
    This automatically checks the backward-process of a given function. For
    checking the correctness, this function compares gradients by
    backprop and ones by numerical derivation. If the result is within a
    tolerance this function return True, otherwise False.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A traget `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        bool: Return True if the result is within a tolerance, otherwise False.
    """
    x = as_variable(x)
    x.data = x.data.astype(np.float64)

    y = f(x, *args, **kwards)
    y.backward()
    bp_grad = x.grad.data

    num_grad = numerical_grad(f, x, *args, **kwards)

    assert bp_grad.shape == num_grad.shape
    return np.allclose(bp_grad, num_grad, atol=atol, rtol=rtol)


def numerical_grad(f, x, *args, **kwargs):
    """
    Computes numerical gradient by finite differences.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (np.ndarray or dezero.Variable): A target `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        `ndarray`: Gradient.

    """
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    grad = np.zeros_like(x)

    # Calculate all partial deviation with all elements one-by-one.
    # Below np.nditer iterates all element in multi-dimensional array.
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_x = x[idx].copy()

        x[idx] = tmp_x + eps
        y_r = f(x, *args, **kwargs)
        if isinstance(y_r, Variable):
            y_r = y_r.data
        y_r = y_r.copy()

        x[idx] = tmp_x - eps
        y_l = f(x, *args, **kwargs)
        if isinstance(y_l, Variable):
            y_l = y_l.data
        y_l = y_l.copy()

        diff = (y_r - y_l).sum()
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_x
        it.iternext()
    return grad


class FunctionTestCase(unittest.TestCase):

    def validate_forward(self, func, inputs, expected=None, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        y = func(*inputs, **kwargs)

        self.assertTrue(np.allclose(y.data, expected))

    def validate_backward(self, f, inputs, expected_grads=None, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
            expected_grads = [expected_grads]
        y = f(*inputs, **kwargs)
        y.backward()

        for x, expected_grad in zip(inputs, expected_grads):
            if (isinstance(x, Variable)) and (expected_grad is not None):
                bp_grad = x.grad
                self.assertTrue(np.allclose(bp_grad.data, expected_grad.data))

    def validate_numerically(self, f, *args, **kwargs):
        np.random.seed(0)
        self.assertTrue(gradient_check(f, np.random.rand(100), *args, **kwargs))
        self.assertTrue(gradient_check(f, np.random.rand(10, 10) * 100, *args, **kwargs))
        self.assertTrue(gradient_check(f, np.random.rand(10, 10, 10) * 100, *args, **kwargs))