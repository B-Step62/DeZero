import numpy as np
from dezero.core import add
from dezero.functions import *
from tests.utils import FunctionTestCase


class TestFunctions(FunctionTestCase):

    def test_sin_forward(self):
        self.validate_forward(sin,
                              inputs=Variable(np.array([1., 2.])),
                              expected=np.array(np.sin([1., 2.])))

    def test_sin_backward(self):
        self.validate_backward(sin,
                               inputs=Variable(np.array([1., 2.])),
                               expected_grads=Variable(np.array(np.cos([1., 2.]))))

    def test_cos_forward(self):
        self.validate_forward(cos,
                              inputs=Variable(np.array([1., 2.])),
                              expected=np.array(np.cos([1., 2.])))

    def test_cos_backward(self):
        self.validate_backward(cos,
                               inputs=Variable(np.array([1., 2.])),
                               expected_grads=Variable(np.array(-np.sin([1., 2.]))))

    def test_tanh_forward(self):
        self.validate_forward(tanh,
                              inputs=Variable(np.array([1., 2.])),
                              expected=np.array(np.tanh([1., 2.])))

    def test_tanh_backward(self):
        x = np.array([1., 2.])
        y = np.tanh(x)
        expected_grad = 1 - y * y

        self.validate_backward(tanh,
                               inputs=Variable(x),
                               expected_grads=Variable(expected_grad))

    def test_reshape_forward(self):
        self.validate_forward(lambda x: reshape(x, (4, 1)),
                              inputs=Variable(np.array([[1., 2.], [3., 4.]])),
                              expected=np.array([[1.], [2.], [3.], [4.]]))

    def test_reshape_backward(self):
        self.validate_backward(lambda x: reshape(x, (4, 1)),
                               inputs=Variable(np.array([[1., 2.], [3., 4.]])),
                               expected_grads=np.array([[1., 1.], [1., 1.]]))

    def test_transpose_forward(self):
        self.validate_forward(transpose,
                              inputs=Variable(np.array([[1., 2.], [3., 4.]])),
                              expected=np.array([[1., 3.], [2., 4.]]))

    def test_transpose_backward(self):
        self.validate_backward(transpose,
                               inputs=Variable(np.array([[1., 2.], [3., 4.]])),
                               expected_grads=np.array([[1., 1.], [1., 1.]]))

    def test_sum_forward(self):
        self.validate_forward(sum,
                              inputs=Variable(np.array([1., 2., 3.])),
                              expected=np.array([6.]))

    def test_sum_backward(self):
        self.validate_backward(sum,
                               inputs=Variable(np.array([1., 2., 3.])),
                               expected_grads=np.array([1., 1., 1.]))

    def test_add_broadcast_forward(self):
        self.validate_forward(add,
                              inputs=[Variable(np.array([1., 2., 3.])), Variable(np.array([10.]))],
                              expected=np.array([11., 12., 13.]))

    def test_add_broadcast_backward(self):
        self.validate_backward(add,
                               inputs=[Variable(np.array([1., 2., 3.])), Variable(np.array([10.]))],
                               expected_grads=[Variable(np.array([1., 1., 1.])), Variable(np.array([3.]))])

