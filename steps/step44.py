import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import dezero.functions as F
import dezero.layers as L

# Dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000
xaxis = np.arange(min(x), max(x), 0.01).reshape(-1, 1)

fig = plt.figure(figsize=(8, 8))
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if (i+1) % 1000 == 0:
        plt.plot(xaxis, predict(xaxis).data,
                 label=f"iter:{i+1}, loss:{loss.data:.3f}",
                 color=cm.bone(1.0 - i/iters))

plt.scatter(x, y)
plt.legend()
plt.show()

