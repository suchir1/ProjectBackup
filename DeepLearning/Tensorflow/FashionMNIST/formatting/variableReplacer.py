import os

filepath = os.path.dirname(os.path.abspath(__file__))
filepath += '/HyperoptOptimization.py'
with open(filepath,"r") as f:
    lines = f.readlines()
variables = {'layer3Filters', 'layer1Filters', 'denseLayer', 'batchSize', 'layer2Filters', 'layer3Kernel', 'secondDropout', 'learningRate', 'firstDropout', 'layer1Kernel', 'layer4Kernel', 'layer2Kernel', 'layer4Filters'}
best = []
for line in lines:
    s = line
    for var in variables:
        print(var)
        s = s.replace(var,"params['"+var+"']")
    best.append(s)


filepath = os.path.dirname(os.path.abspath(__file__))
filepath += '/test.py'

with open(filepath,'w') as f:
    for line in best:
        f.write(line)