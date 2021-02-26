import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x**2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


v = Variable(np.array(0.5))
A, B, C = Square(), Exp(), Square()
y = A(B(C(v)))

print(type(y), y.data)