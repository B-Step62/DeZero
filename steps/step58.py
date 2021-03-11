import numpy as np
from PIL import Image
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import dezero
from dezero.models import VGG16
from dezero.utils import get_file

model = VGG16(pretrained=True)
img_path = get_file("https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg")
img = Image.open(img_path)
x = VGG16.preprocess(img)
x = x[np.newaxis]

with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)
print(predict_id)