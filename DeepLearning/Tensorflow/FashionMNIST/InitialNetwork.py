# pylint: disable=E0611

import tensorflow as tf
import numpy as np
import mnist_reader
import os
import sys
from tensorflow.python.keras.layers import Activation, Dense, Input, Dropout, Conv2D
from tensorflow.python.keras.optimizers import Adam

filepath = sys.argv[0][:sys.argv[0].rfind("/")]+"/data/fashion"
x_train, y_train = mnist_reader.load_mnist(filepath, kind='train')
x_test, y_test = mnist_reader.load_mnist(filepath, kind='t10k')

model = None
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = tf.keras.Sequential()
model.add(Conv2D())


optimizer = Adam(lr=params['learningRate'])

model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

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
model.fit(x=data, y=labels, batch_size=32, epochs=int(params['epochs']))



score = model.evaluate(data, labels, batch_size=128)

print("First Layer Nodes: " +str(params['firstLayerNodes']))
print("Second Layer Nodes: " + str(params['secondLayerNodes']))
print("Third Layer Nodes: " + str(params['thirdLayerNodes']))
print("Learning Rate: "+str(params['learningRate']))
print("Epochs: "+str(params['epochs']))
print("Accuracy on Testing Data:",str(score[1]*100)+"%")
sess.close()
return {'loss': score[0], 'status': STATUS_OK }