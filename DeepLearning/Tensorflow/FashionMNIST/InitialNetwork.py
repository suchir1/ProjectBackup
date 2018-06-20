# pylint: disable=E0611

import tensorflow as tf
import numpy as np
import mnist_reader
import os
import sys
from tensorflow.python.keras.layers import Activation, Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.optimizers import Adam
from matplotlib import pyplot as plt

num_classes = 10
filepath = sys.argv[0][:sys.argv[0].rfind("/")]+"/data/fashion"
x_train, y_train = mnist_reader.load_mnist(filepath, kind='train')
x_test, y_test = mnist_reader.load_mnist(filepath, kind='t10k')
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

model = None
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = tf.keras.Sequential()
model.add(Conv2D(32,kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=.0000001)

model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=32, epochs=int(100))
print(model.output_shape)


score = model.evaluate(x_test, y_test, batch_size=128)

print("Accuracy on Testing Data:",str(score[1]*100)+"%")
sess.close()