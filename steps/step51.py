import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import dezero
from dezero import DataLoader
from dezero.datasets import MNIST
import dezero.functions as F
import dezero.transforms as T
from dezero.models import MLP
from dezero.optimizers import SGD

# Train config
batch_size = 100
max_epoch = 5

# Dataset
def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x
train_set = MNIST(train=True, transform=f)
test_set = MNIST(train=False, transform=f)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# Model
hidden_size = 1000
lr = 0.01
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = SGD(lr).setup(model)

train_losses, train_accs, test_losses, test_accs = [], [], [], []
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print(f"epoch: {epoch+1}")
    print(f"train loss: {sum_loss / len(train_set):.4f}, accuracy: {sum_acc / len(train_set):.4f}")
    train_losses.append(sum_loss / len(train_set))
    train_accs.append(sum_acc / len(train_set))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        print(f"test loss: {sum_loss / len(test_set):.4f}, accuracy: {sum_acc / len(test_set):.4f}")
        test_losses.append(sum_loss / len(test_set))
        test_accs.append(sum_acc / len(test_set))

fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot(1, 2, 1)
xs = np.arange(0, max_epoch, 1)
ax1.plot(xs, train_losses, label="train")
ax1.plot(xs, test_losses, label="test")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2 = plt.subplot(1, 2, 2)
ax2.plot(xs, train_accs, label="train")
ax2.plot(xs, test_accs, label="test")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.set_ylim([0., 1.])
ax2.legend()

plt.show()