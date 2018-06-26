# pylint: disable=E0611

import tensorflow as tf
import numpy as np
import mnist_reader
import os
import sys
from tensorflow.python.keras.layers import Activation, Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.optimizers import Adam
from matplotlib import pyplot as plt


def binaryDecoder(x):
    ans = 0
    for i in range(len(x)):
        if(x[i]!=1):
            continue
        ans+=(x[i]*2)**(len(x)-i-1)
    return int(ans)

def runCNN(x):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    num_classes = 10
    filepath = os.path.dirname(os.path.abspath(__file__))+'/data/fashion/'
    x_train, y_train = mnist_reader.load_mnist(filepath, kind='train')
    x_test, y_test = mnist_reader.load_mnist(filepath, kind='t10k')
    x_train = x_train.reshape((60000,28,28,1))
    x_test = x_test.reshape((10000,28,28,1))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    filters = [16,32,64,128]
    kernels = [3,4,5,6]
    dropout = [.1,.2,.3,.4]
    denseNodes = [256, 512, 1024, 2048]
    learningRate = [.000000001,.00000001,.0000001,.000001,.00001,.0001,.001,.01]
    batchSize = [16,32,64,128]

    model = None
    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    model = tf.keras.Sequential()
    model.add(Conv2D(filters[binaryDecoder(x[0:2])], (kernels[binaryDecoder(x[2:4])], kernels[binaryDecoder(x[2:4])]), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(filters[binaryDecoder(x[4:6])], (kernels[binaryDecoder(x[6:8])], kernels[binaryDecoder(x[6:8])])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout[binaryDecoder(x[16:18])]))

    model.add(Conv2D(filters[binaryDecoder(x[8:10])], (kernels[binaryDecoder(x[10:12])], kernels[binaryDecoder(x[10:12])]), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters[binaryDecoder(x[12:14])], (kernels[binaryDecoder(x[14:16])], kernels[binaryDecoder(x[14:16])])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout[binaryDecoder(x[18:20])]))

    model.add(Flatten())
    model.add(Dense(denseNodes[binaryDecoder(x[20:22])]))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learningRate[binaryDecoder(x[22:25])])

    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=batchSize[binaryDecoder(x[25:27])], epochs=int(1))


    score = model.evaluate(x_test, y_test, batch_size=128)

    print("Accuracy on Testing Data:",str(score[1]*100)+"%")
    sess.close()
    return score[0]