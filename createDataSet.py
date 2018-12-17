import numpy as np
import os
import cv2
import random
import pickle
data=[]
DIR = '/home/priyansh/code/NeuralNetworks/classification/data/PetImages' 
CATEGORIES = ["Cat", "Dog"]   #cats will be labeled as 0 and dogs will be labeled as 1  
IMG_SIZE = 50

for category in CATEGORIES: #iterating through Cat and Dog folders 
    label = CATEGORIES.index(category)

    path = os.path.join(DIR, category)
    for img in os.listdir(path):    #iterating through all the images
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append([new_array, label])
        except Exception as e :
            pass
random.shuffle(data)
X=[]
y=[]

for features, lbl in data:
    X.append(features)
    y.append(lbl)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#pickling
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()