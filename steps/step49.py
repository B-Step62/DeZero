import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import datasets
from dezero import optimizers
import dezero.functions as F
import dezero.transforms as T
from dezero.models import MLP

# Dataset
f = T.Compose([T.Normalize(mean=0.0, std=1.0), T.AsType(np.float64)])
train_set = datasets.Spiral()

# Model
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print(f"epoch {epoch + 1}, loss {avg_loss:.2f}")
