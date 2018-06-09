import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Dense, Input

sess = tf.InteractiveSession()

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/TRAINING_DATA.csv"

model = tf.keras.Sequential()
model.add(Dense(9,input_dim=9))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

labels = np.random.randint(2, size=(1000, 1))

data = list()
labels = list()
with open(filename) as inf:
    next(inf)
    for line in inf:
        values = line.strip().split(',')
        features = values[0:9]
        label = list(values[10])
        data.append(features)
        labels.append(label)

data = np.asarray(data,dtype=int)
labels = np.asarray(labels, dtype=int)
model.fit(x=data, y=labels, batch_size=32, epochs=200)


filename = dir_path + "/TESTING_DATA.csv"

data = list()
labels = list()

with open(filename) as inf:
    next(inf)
    for line in inf:
        values = line.strip().split(',')
        features = values[0:9]
        label = list(values[10])
        data.append(features)
        labels.append(label)

data = np.asarray(data,dtype=int)
labels = np.asarray(labels, dtype=int)

score = model.evaluate(data, labels, batch_size=128)
print("Accuracy on Testing Data:",str(score[1]*100)+"%")
