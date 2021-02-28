from memory_profiler import profile
import numpy as np

if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable


x = Variable(np.array(2.0))
y = x ** 3
y.backward()

print(y, x.grad)

