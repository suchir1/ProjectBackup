import re

terminalOutput = """
Epoch 1/10
60000/60000 [==============================] - 35s 586us/step - loss: 1.5981 - acc: 0.7315
Epoch 2/10
60000/60000 [==============================] - 35s 591us/step - loss: 0.4175 - acc: 0.8501
Epoch 3/10
60000/60000 [==============================] - 35s 579us/step - loss: 0.3570 - acc: 0.8695
Epoch 4/10
60000/60000 [==============================] - 36s 595us/step - loss: 0.3135 - acc: 0.8857
Epoch 5/10
60000/60000 [==============================] - 34s 570us/step - loss: 0.2857 - acc: 0.8958
Epoch 6/10
60000/60000 [==============================] - 34s 571us/step - loss: 0.2595 - acc: 0.9042
Epoch 7/10
60000/60000 [==============================] - 34s 571us/step - loss: 0.2378 - acc: 0.9112
Epoch 8/10
60000/60000 [==============================] - 34s 571us/step - loss: 0.2191 - acc: 0.9177
Epoch 9/10
60000/60000 [==============================] - 34s 570us/step - loss: 0.2069 - acc: 0.9216
Epoch 10/10
60000/60000 [==============================] - 34s 567us/step - loss: 0.1932 - acc: 0.9265
10000/10000 [==============================] - 1s 86us/step
Accuracy on Testing Data: 91.32000000000001%
{'batchSize': 32, 'denseNodes': 1272.0, 'firstDropout': 0.27206556831049483, 'layer1Filters': 36, 'layer1Kernel': 5, 'layer2Filters': 32, 'layer2Kernel': 3, 'layer3Filters': 68, 'layer3Kernel': 6, 'layer4Filters': 118, 'layer4Kernel': 4, 'learningRate': 6.549705573584309e-05, 'secondDropout': 0.3342362811919958}
2018-06-30 11:55:16.900339: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-06-30 11:55:16.900400: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-06-30 11:55:16.900424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-06-30 11:55:16.900429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-06-30 11:55:16.900586: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3052 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Epoch 1/10
60000/60000 [==============================] - 17s 291us/step - loss: 2.0963 - acc: 0.5663
Epoch 2/10
60000/60000 [==============================] - 17s 287us/step - loss: 0.7217 - acc: 0.7395
Epoch 3/10
60000/60000 [==============================] - 17s 286us/step - loss: 0.5958 - acc: 0.7848
Epoch 4/10
60000/60000 [==============================] - 17s 287us/step - loss: 0.5199 - acc: 0.8123
Epoch 5/10
60000/60000 [==============================] - 17s 287us/step - loss: 0.4685 - acc: 0.8306
Epoch 6/10
60000/60000 [==============================] - 17s 287us/step - loss: 0.4356 - acc: 0.8435
Epoch 7/10
60000/60000 [==============================] - 17s 287us/step - loss: 0.4094 - acc: 0.8531
Epoch 8/10
60000/60000 [==============================] - 17s 287us/step - loss: 0.3872 - acc: 0.8596
Epoch 9/10
60000/60000 [==============================] - 17s 286us/step - loss: 0.3680 - acc: 0.8663
Epoch 10/10
60000/60000 [==============================] - 17s 287us/step - loss: 0.3580 - acc: 0.8695
10000/10000 [==============================] - 1s 85us/step
Accuracy on Testing Data: 88.1%
{'batchSize': 64, 'denseNodes': 674.0, 'firstDropout': 0.17506922030178873, 'layer1Filters': 51, 'layer1Kernel': 3, 'layer2Filters': 24, 'layer2Kernel': 4, 'layer3Filters': 89, 'layer3Kernel': 6, 'layer4Filters': 56, 'layer4Kernel': 5, 'learningRate': 1.4308084976617113e-05, 'secondDropout': 0.2760358166806037}
2018-06-30 11:58:10.971495: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-06-30 11:58:10.971555: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-06-30 11:58:10.971581: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-06-30 11:58:10.971586: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-06-30 11:58:10.971727: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3052 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Epoch 1/10
60000/60000 [==============================] - 20s 330us/step - loss: 1.5283 - acc: 0.6089
Epoch 2/10
60000/60000 [==============================] - 19s 324us/step - loss: 0.6362 - acc: 0.7643
Epoch 3/10
60000/60000 [==============================] - 20s 326us/step - loss: 0.5334 - acc: 0.8049
Epoch 4/10
60000/60000 [==============================] - 19s 324us/step - loss: 0.4760 - acc: 0.8277
Epoch 5/10
60000/60000 [==============================] - 20s 331us/step - loss: 0.4345 - acc: 0.8422
Epoch 6/10
60000/60000 [==============================] - 20s 332us/step - loss: 0.4033 - acc: 0.8534
Epoch 7/10
60000/60000 [==============================] - 20s 333us/step - loss: 0.3772 - acc: 0.8630
Epoch 8/10
60000/60000 [==============================] - 20s 332us/step - loss: 0.3601 - acc: 0.8683
Epoch 9/10
60000/60000 [==============================] - 20s 334us/step - loss: 0.3385 - acc: 0.8772
Epoch 10/10
60000/60000 [==============================] - 19s 325us/step - loss: 0.3260 - acc: 0.8808
10000/10000 [==============================] - 1s 97us/step
Accuracy on Testing Data: 89.45%
{'batchSize': 128, 'denseNodes': 1034.0, 'firstDropout': 0.35624274674330014, 'layer1Filters': 19, 'layer1Kernel': 6, 'layer2Filters': 126, 'layer2Kernel': 3, 'layer3Filters': 63, 'layer3Kernel': 4, 'layer4Filters': 33, 'layer4Kernel': 4, 'learningRate': 6.425091199718967e-05, 'secondDropout': 0.3738774789823627}
2018-06-30 12:01:30.211981: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-06-30 12:01:30.212040: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-06-30 12:01:30.212065: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-06-30 12:01:30.212070: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-06-30 12:01:30.212193: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3052 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Epoch 1/10
60000/60000 [==============================] - 33s 543us/step - loss: 0.5910 - acc: 0.7971
Epoch 2/10
60000/60000 [==============================] - 32s 531us/step - loss: 0.3540 - acc: 0.8741
Epoch 3/10
60000/60000 [==============================] - 32s 530us/step - loss: 0.2986 - acc: 0.8931
Epoch 4/10
60000/60000 [==============================] - 32s 530us/step - loss: 0.2647 - acc: 0.9044
Epoch 5/10
60000/60000 [==============================] - 32s 533us/step - loss: 0.2392 - acc: 0.9138
Epoch 6/10
60000/60000 [==============================] - 32s 527us/step - loss: 0.2175 - acc: 0.9214
Epoch 7/10
60000/60000 [==============================] - 32s 529us/step - loss: 0.1992 - acc: 0.9274
Epoch 8/10
60000/60000 [==============================] - 33s 547us/step - loss: 0.1820 - acc: 0.9337
Epoch 9/10
60000/60000 [==============================] - 32s 538us/step - loss: 0.1644 - acc: 0.9406
Epoch 10/10
60000/60000 [==============================] - 32s 541us/step - loss: 0.1513 - acc: 0.9452
10000/10000 [==============================] - 2s 175us/step
Accuracy on Testing Data: 91.0%
{'batchSize': 128, 'denseNodes': 1695.0, 'firstDropout': 0.01401723735268362, 'layer1Filters': 110, 'layer1Kernel': 5, 'layer2Filters': 86, 'layer2Kernel': 3, 'layer3Filters': 110, 'layer3Kernel': 6, 'layer4Filters': 23, 'layer4Kernel': 3, 'learningRate': 4.151542755470882e-05, 'secondDropout': 0.01938713695957689}
2018-06-30 12:06:53.703609: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-06-30 12:06:53.703671: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-06-30 12:06:53.703696: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-06-30 12:06:53.703701: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-06-30 12:06:53.703826: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3052 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Epoch 1/10
60000/60000 [==============================] - 15s 252us/step - loss: 1.5870 - acc: 0.6274
Epoch 2/10
60000/60000 [==============================] - 15s 245us/step - loss: 0.6170 - acc: 0.7767
Epoch 3/10
60000/60000 [==============================] - 15s 245us/step - loss: 0.5171 - acc: 0.8142
Epoch 4/10
60000/60000 [==============================] - 15s 245us/step - loss: 0.4663 - acc: 0.8311
Epoch 5/10
60000/60000 [==============================] - 15s 243us/step - loss: 0.4293 - acc: 0.8440
Epoch 6/10
60000/60000 [==============================] - 15s 242us/step - loss: 0.4045 - acc: 0.8551
Epoch 7/10
60000/60000 [==============================] - 15s 242us/step - loss: 0.3827 - acc: 0.8616
Epoch 8/10
60000/60000 [==============================] - 14s 240us/step - loss: 0.3676 - acc: 0.8665
Epoch 9/10
60000/60000 [==============================] - 14s 240us/step - loss: 0.3527 - acc: 0.8718
Epoch 10/10
60000/60000 [==============================] - 14s 240us/step - loss: 0.3381 - acc: 0.8772
10000/10000 [==============================] - 1s 66us/step
Accuracy on Testing Data: 88.34%
{'batchSize': 128, 'denseNodes': 1030.0, 'firstDropout': 0.30298406677836387, 'layer1Filters': 28, 'layer1Kernel': 6, 'layer2Filters': 21, 'layer2Kernel': 6, 'layer3Filters': 96, 'layer3Kernel': 5, 'layer4Filters': 99, 'layer4Kernel': 4, 'learningRate': 2.8287376070000416e-05, 'secondDropout': 0.20824617183203004}
Best hyperparameters {'batchSize': 32, 'denseNodes': 1272.0, 'firstDropout': 0.27206556831049483, 'layer1Filters': 36.0, 'layer1Kernel': 5.0, 'layer2Filters': 32.0, 'layer2Kernel': 3.0, 'layer3Filters': 68.0, 'layer3Kernel': 6.0, 'layer4Filters': 118.0, 'layer4Kernel': 4.0, 'learningRate': 6.549705573584309e-05, 'secondDropout': 0.3342362811919958}




"""

indices = [m.end() for m in re.finditer('acc: ', terminalOutput)]

accuracies = []

for index in indices:
    accuracies.append(terminalOutput[index:index+6])
accuracies = [float(acc) for acc in accuracies]
accuracyList = []
for i in range(len(accuracies)):
    if i%10==0:
        best = list()
    best.append(accuracies[i])
    if i%10==0:
        accuracyList.append(best)
print(accuracyList)