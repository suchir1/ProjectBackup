import os
import re
import ast

filepath = os.path.dirname(os.path.abspath(__file__))
filepath += '/GCEoutput.txt'

with open(filepath, 'r') as f:
    lines = f.readlines()

singleString = '\n'.join(lines)

accuracies = [m.end() for m in re.finditer('Accuracy on Testing Data: ', singleString)]
accuracyParameterPairs = {}


for i in range(len(accuracies)):
    accIndex = accuracies[i]
    accuracies[i] = singleString[accuracies[i]:singleString.find('%',accuracies[i])]
    accuracies[i] = float(accuracies[i])
    hyperparameterIndex = singleString.find('Hyperparameters: ',accIndex)+len('Hyperparameters: ')-1
    hyperparameters = singleString[hyperparameterIndex:singleString.find('\n',hyperparameterIndex)]
    if accuracies[i] not in accuracyParameterPairs:
        accuracyParameterPairs[accuracies[i]] = list()
    accuracyParameterPairs[accuracies[i]].append(hyperparameters)

for key in accuracyParameterPairs.keys():
    for i in range(len(accuracyParameterPairs[key])):
        accuracyParameterPairs[key][i] = ast.literal_eval(accuracyParameterPairs[key][i].strip())
    
sortedKeys = sorted(accuracyParameterPairs.keys())
filepath = os.path.dirname(os.path.abspath(__file__))
filepath += '/sortedHyperparameters.txt'
with open(filepath,'w') as file:
    for key in sortedKeys:
        file.write(str(key)+": "+str(accuracyParameterPairs[key])+"\n")