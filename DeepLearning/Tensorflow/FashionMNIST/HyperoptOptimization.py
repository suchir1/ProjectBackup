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

def runCNN(params):
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
    model.add(Conv2D(params['layer4Filters'], params['layer4Kernel']))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['secondDropout']))

    model.add(Flatten())
    model.add(Dense(params['denseNodes']))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=params['learningRate'])

    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=params['batchSize'], epochs=1)
    #Turns out that according to the 6 tests I ran, 1 epoch is enough to see which network is the most accurate, testing wasn't extensive though


    score = model.evaluate(x_test, y_test, batch_size=128)

    print("Accuracy on Testing Data:",str(score[1]*100)+"%")
    print(params)
    sess.close()
    return {'loss': score[0], 'status': STATUS_OK }

batchSizes = [32,64,128]
parameters = {'secondDropout': hp.uniform("secondDropout",0,.5), 'firstDropout': hp.uniform("firstDropout",0,.5), 'layer3Filters': hp.quniform("layer3Filters",16,128,1), 'layer2Filters': hp.quniform("layer2Filters",16,128,1), 'layer2Kernel': hp.quniform("layer2Kernel",3,6,1), 'denseNodes': hp.quniform("denseNodes",256,2048,1), 'batchSize': hp.choice("batchSize",batchSizes), 'learningRate': hp.uniform("learningRate",.000000001,.0001), 'layer1Filters': hp.quniform("layer1Filters",16,128,1), 'layer3Kernel': hp.quniform("layer3Kernel",3,6,1), 'layer1Kernel': hp.quniform("layer1Kernel",3,6,1), 'layer4Filters': hp.quniform("layer4Filters",16,128,1), 'layer4Kernel': hp.quniform("layer4Kernel",3,6,1)}
best = fmin(fn=runCNN, space=parameters, algo=tpe.suggest, max_evals=5, verbose=True)
best['batchSize'] = batchSizes[best['batchSize']]
print("Best hyperparameters " + str(best))

#TODO: Figure out how to automatically start and stop the google compute engine based on a script
#TODO: Maybe delete the instance IFF your tensorflow installation is on a persistent disk, need to figure that out
#TODO: Run hyperopt on GCE