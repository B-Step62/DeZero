import numpy as np
import unittest
from dezero.core import *
from tests.utils import FunctionTestCase, gradient_check


class CastTest(unittest.TestCase):
    def test_as_array_scalar(self):
        x = 1.0
        self.assertEqual(as_array(x), np.array(x))

    def test_as_array_array(self):
        x = np.array(1.0)
        self.assertEqual(as_array(x), x)

    def test_as_variable_array(self):
        x = np.array(1.0)
        v = as_variable(x)
        self.assertTrue(isinstance(v, Variable))
        self.assertEqual(v.data, x)

    def test_as_variable_variable(self):
        x = Variable(np.array(1.0))
        self.assertEqual(as_variable(x), x)


class ConfigTest(unittest.TestCase):
    def no_grad_test(self):
        Config.enable_backprop = True
        with no_grad():
            self.assertTrue(Config.enable_backprop)
        self.assertFalse(Config.enable_backprop)


class TestVariable(unittest.TestCase):
    def test_init_from_array(self):
        x = np.array(1.0)
        v = Variable(x, "x")
        self.assertEqual(v.data, x)
        self.assertEqual(v.name, "x")

    def test_init_from_scalar_exception(self):
        x = 1.0
        with self.assertRaises(TypeError):
            v = Variable(x, "x")


class TestCoreFunctions(FunctionTestCase):

    def test_neg_forward(self):
        self.validate_forward(neg,
                              inputs=Variable(np.array([1., 2.])),
                              expected=np.array([-1., -2.]))

    def test_neg_backward(self):
        self.validate_backward(neg,
                               inputs=Variable(np.array([1., 2.])),
                               expected_grads=Variable(np.array([-1., -1.])))
        self.validate_numerically(neg)

    def test_add_forward(self):
        self.validate_forward(add,
                              inputs=[Variable(np.array([1., -1.])), Variable(np.array([2., 1.]))],
                              expected=np.array([3., 0.]))

    def test_add_backward(self):
        self.validate_backward(add,
                               inputs=[np.array([1., -1.]), np.array([2., 1.])],
                               expected_grads=[Variable(np.array([1., 1.])), Variable(np.array([1., 1.]))])
        self.validate_numerically(lambda x: add(x, Variable(np.array(1.0))))
        self.validate_numerically(lambda x: add(Variable(np.array(1.0)), x))

    def test_sub_forward(self):
        self.validate_forward(sub,
                              inputs=[Variable(np.array([1., -1.])), Variable(np.array([2., 1.]))],
                              expected=np.array([-1., -2.]))

    def test_sub_backward(self):
        self.validate_backward(sub,
                               inputs=[Variable(np.array([1., 2.])), Variable(np.array([2., -1.]))],
                               expected_grads=[Variable(np.array([1., 1.])), Variable(np.array([-1., -1.]))])
        self.validate_numerically(lambda x: sub(x, Variable(np.array(1.0))))
        self.validate_numerically(lambda x: sub(Variable(np.array(1.0)), x))

    def test_rsub_forward(self):
        self.validate_forward(rsub,
                              inputs=[Variable(np.array([1., -1.])), Variable(np.array([2., 1.]))],
                              expected=np.array([1., 2.]))

    def test_rsub_backward(self):
        self.validate_backward(rsub,
                               inputs=[Variable(np.array([1., 2.])), Variable(np.array([2., -1.]))],
                               expected_grads=[Variable(np.array([-1., -1.])), Variable(np.array([1., 1.]))])
        self.validate_numerically(lambda x: rsub(x, Variable(np.array(1.0))))
        self.validate_numerically(lambda x: rsub(Variable(np.array(1.0)), x))

    def test_mul_forward(self):
        self.validate_forward(mul,
                              inputs=[Variable(np.array([1., -1.])), Variable(np.array([2., 3.]))],
                              expected=np.array([2., -3.]))

    def test_mul_backward(self):
        self.validate_backward(mul,
                               inputs=[Variable(np.array([1., -1.])), Variable(np.array([2., 3.]))],
                               expected_grads=[Variable(np.array([2., 3.])), Variable(np.array([1., -1.]))])
        self.validate_numerically(lambda x: mul(x, Variable(np.array(2.0))))
        self.validate_numerically(lambda x: mul(Variable(np.array(2.0)), x))

    def test_div_forward(self):
        self.validate_forward(div,
                              inputs=[Variable(np.array([1., -6.])), Variable(np.array([2., 4.]))],
                              expected=np.array([0.5, -1.5]))

    def test_div_backward(self):
        self.validate_backward(div,
                               inputs=[Variable(np.array([1., -6.])), Variable(np.array([2., 4.]))],
                               expected_grads=[Variable(np.array([0.5, 0.25])), Variable(np.array([-0.25, 0.375]))])

    def test_rdiv_forward(self):
        self.validate_forward(rdiv,
                              inputs=[Variable(np.array([2., 4.])), Variable(np.array([1., -6.]))],
                              expected=np.array([0.5, -1.5]))

    def test_rdiv_backward(self):
        self.validate_backward(rdiv,
                               inputs=[Variable(np.array([2., 4.])), Variable(np.array([1., -6.]))],
                               expected_grads=[Variable(np.array([-0.25, 0.375])), Variable(np.array([0.5, 0.25]))])

    def test_pow_forward(self):
        self.validate_forward(pow,
                              inputs=Variable(np.array([2., 3.])),
                              c=4,
                              expected=np.array([16., 81.]))

    def test_pow_backward(self):
        self.validate_backward(pow,
                               inputs=Variable(np.array([2., 3.])),
                               c=4,
                               expected_grads=Variable(np.array([32., 108])))
        self.validate_numerically(pow, c=4)

    def test_composittion_sphere(self):
        def _sphere(x, y):
            return x**2 + y**2
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = _sphere(x, y)
        self.assertTrue(np.allclose(z.data, np.array(13.0)))
        z.backward()
        self.assertTrue(np.allclose(x.grad.data, np.array(4.0)))
        self.assertTrue(np.allclose(y.grad.data, np.array(6.0)))

    def test_composittion_matyas(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y + 1.0
        self.assertTrue(np.allclose(z.data, np.array(1.5)))
        z.backward()
        self.assertTrue(np.allclose(x.grad.data, np.array(-0.4)))
        self.assertTrue(np.allclose(y.grad.data, np.array(0.6)))