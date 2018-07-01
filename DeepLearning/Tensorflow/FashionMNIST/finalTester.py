# pylint: disable=E0611, E0602

import tensorflow as tf
import numpy as np
import mnist_reader
import os
import sys
from tensorflow.python.keras.layers import Activation, Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.optimizers import Adam
from matplotlib import pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK
import hyperopt
import ast
import re

def runCNN(params):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    for key in params:
        if 'Filter' in key or 'Kernel' in key:
            params[key] = int(params[key])

    num_classes = 10
    filepath = os.path.dirname(os.path.abspath(__file__))+'/data/fashion/'
    x_train, y_train = mnist_reader.load_mnist(filepath, kind='train')
    x_test, y_test = mnist_reader.load_mnist(filepath, kind='t10k')
    x_train = x_train.reshape((60000,28,28,1))
    x_test = x_test.reshape((10000,28,28,1))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    model = None
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    model = tf.keras.Sequential()
    model.add(Conv2D(params['layer1Filters'], params['layer1Kernel'], padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params['layer2Filters'], params['layer2Kernel']))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['firstDropout']))

    model.add(Conv2D(params['layer3Filters'], params['layer3Kernel'], padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Conv2D(params['layer4Filters'], params['layer4Kernel']))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['secondDropout']))

    model.add(Flatten())
    model.add(Dense(params['denseNodes']))
    model.add(Activation('relu'))
    model.add(Dropout(.3))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=params['learningRate'])

    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    for i in range(100):
        model.fit(x=x_train, y=y_train, batch_size=params['batchSize'], epochs=1)
        score = model.evaluate(x_test, y_test, batch_size=128)
        print("Accuracy on Testing Data:",str(score[1]*100)+"%")

    print("Hyperparameters: "+ str(params))
    sess.close()
    return {'loss': score[0], 'status': STATUS_OK }


filepath = os.path.dirname(os.path.abspath(__file__))
filepath += '/tenEpochTest.txt'

with open(filepath, "r") as file:
    lines = file.readlines()
fullOutput = '\n'.join(lines)

accuracyIndices = [m.end() for m in re.finditer('Accuracy on Testing Data: ', fullOutput)]
paramIndices = list()
for i in range(len(accuracyIndices)):
    paramIndices.append(fullOutput.find('Hyperparameters: ', accuracyIndices[i])+len('Hyperparameters: ')-1)
    accuracyIndices[i] = fullOutput[accuracyIndices[i]:fullOutput.find('%', accuracyIndices[i])]
    accuracyIndices[i] = float(accuracyIndices[i])
    paramIndices[i] = fullOutput[paramIndices[i]:fullOutput.find('\n', paramIndices[i])]
    paramIndices[i] = ast.literal_eval(paramIndices[i].strip())

accuracyIndices, paramIndices = zip(*sorted(zip(accuracyIndices, paramIndices)))
accuracyIndices = list(accuracyIndices)
paramIndices = list(paramIndices)

paramIndices[-1]['secondDropout']=.5
paramIndices[-1]['firstDropout']=.5

print(paramIndices[-1])

runCNN(paramIndices[-1])

# for i in range(len(paramIndices)):
#     runCNN(paramIndices[-i-1])
