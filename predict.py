from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

IMG_SIZE = 50
path = sys.argv[1]
try:
	img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
except Exception as e:
	pass
plt.imshow(img_array, cmap="gray")
plt.show()
X = np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
MODEL_NAME = '2-CONV-0-DENSE-64-NODES-1543693503.h5'
model = load_model(MODEL_NAME)

y = model.predict(X)

if y==0:
	print('Cat')
else:
	print('Dog')