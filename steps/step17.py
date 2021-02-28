from memory_profiler import profile
import weakref
import numpy as np
from typing import List, Union


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen = set()

        def add_func(f):
            if f not in seen:
                funcs.append(f)
                seen.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, list):
                gxs = [gxs]

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs: Variable) -> Union[Variable, List[Variable]]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = [ys]
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max(x.generation for x in inputs)
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def __lt__(self, other):
        return self.generation < other.generation

    def forward(self, xs: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> Variable:
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> List[np.ndarray]:
        return [gy, gy]


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x: np.ndarray):
        y = x**2
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)

@profile
def run():
    for i in range(100):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()
    print(y.data, x.grad)


run()