from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout, Activation
from keras.callbacks import TensorBoard
import pickle
import time

NAME = "cats-vs-dogs-cnn-6x24-{}".format(time.time())

tb = TensorBoard(log_dir="Logs/{}".format(NAME))

pickle_in = open("data/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("data/y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tb])

MODEL_NAME = '2-CONV-0-DENSE-64-NODES-{}.h5'.format(int(time.time()))
model.save(MODEL_NAME)