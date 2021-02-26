import numpy as np
import unittest
from steps.step10 import Variable, square, numerical_diff


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        self.assertEqual(y.data, np.array(4.0))

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        self.assertEqual(x.grad, np.array(6.0))

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        self.assertTrue(np.allclose(x.grad, num_grad))