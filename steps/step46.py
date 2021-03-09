import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


# Dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# Model
lr = 0.2
hidden_size = 10
model = MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

iters = 10000
xaxis = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
fig = plt.figure(figsize=(8, 8))

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if (i+1) % 1000 == 0:
        plt.plot(xaxis, model(xaxis).data,
                 label=f"iter:{i+1}, loss:{loss.data:.3f}",
                 color=cm.bone(1.0 - i/iters))

plt.scatter(x, y)
plt.legend()
plt.savefig("MomemtumSGD.png")

