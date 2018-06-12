# pylint: disable=E0611

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Dense, Input, Dropout
from tensorflow.python.keras.optimizers import Adam
from hyperopt import fmin, tpe, hp, STATUS_OK


def objective(params={'firstLayerNodes':19.0, 
    'secondLayerNodes':18.0, 
    'thirdLayerNodes': 61.0, 
    'learningRate':0.08684948972911224,
    'epochs':68.0}):
    
    model = None
    tf.reset_default_graph()

    sess = tf.InteractiveSession()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "/TRAINING_DATA.csv"

    model = tf.keras.Sequential()
    model.add(Dense(params['firstLayerNodes'],input_dim=9))
    model.add(Activation('relu'))
    model.add(Dense(params['secondLayerNodes']))
    model.add(Activation('relu'))
    model.add(Dense(params['thirdLayerNodes']))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    optimizer = Adam(lr=params['learningRate'])

    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
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
    
    print("First Layer Nodes: " +str(params['firstLayerNodes']))
    print("Second Layer Nodes: " + str(params['secondLayerNodes']))
    print("Third Layer Nodes: " + str(params['thirdLayerNodes']))
    print("Learning Rate: "+str(params['learningRate']))
    print("Epochs: "+str(params['epochs']))
    print("Accuracy on Testing Data:",str(score[1]*100)+"%")
    sess.close()
    return {'loss': score[0], 'status': STATUS_OK }

def objectiveAverage(params):
    iterations = 3
    loss = 0
    for i in range(iterations):
        loss+=objective(params=params)['loss']
    loss=loss/iterations
    return loss

squad = {'firstLayerNodes':hp.quniform('firstLayerNodes',9, 90,1), 
    'secondLayerNodes':hp.quniform('secondLayerNodes',5, 90,1), 
    'thirdLayerNodes': hp.quniform('thirdLayerNodes',3, 90,1), 
    'learningRate':hp.uniform('learningRate', .001,.1),
    'epochs':hp.quniform('epochs', 10, 200, 1)}
best = fmin(fn=objectiveAverage, space=squad, algo=tpe.suggest, max_evals=200, verbose=True)

print("Best Hyperparameters: " + str(best))