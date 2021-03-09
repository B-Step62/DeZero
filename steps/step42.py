import matplotlib.pyplot as plt
import numpy as np
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import dezero.functions as F


np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)


x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y


lr = 0.1
iters = 100
xaxis = np.arange(min(x.data), max(x.data), 0.01)

fig = plt.figure(figsize=(8, 8))
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 9:
        plt.plot(xaxis, W.data[0, 0] * xaxis + b.data[0], label=f"iter:{i+1}, W:{W.data[0,0]:.3f}, b:{b.data[0]:.3f}, loss:{loss.data:.3f}")

plt.scatter(x.data, y.data)
plt.legend()
plt.show()

