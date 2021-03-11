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
from dezero.models import MLP
from dezero.optimizers import SGD

# Train config
batch_size = 100
max_epoch = 3

# Dataset
train_set = MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)

# Model
hidden_size = 1000
lr = 0.01
model = MLP((hidden_size, 10), activation=F.relu)
optimizer = SGD(lr).setup(model)

if os.path.exists("ml_mlp.npz"):
    model.load_weight("ml_mlp.npz")

train_losses, train_accs = [], []
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

model.save_weight("ml_mlp.npz")