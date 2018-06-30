import os

filepath = os.path.dirname(os.path.abspath(__file__))
filepath += '/options.txt'
with open(filepath,"r") as f:
    lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].strip()
    lines[i] = lines[i][0:-1]
lines = set(lines)
print(lines)
parameters = {}
for line in lines:
    if 'Kernel' in line:
        parameters[line] = 'hp.quniform("'+line+'",3,6,1)'
    if 'Filters' in line:
        parameters[line] = 'hp.quniform("'+line+'",16,128,1)'
    if 'Dropout' in line:
        parameters[line] = 'hp.uniform("'+line+'",0,1)'
    if 'batch' in line:
        parameters[line] = 'hp.quniform("'+line+'",16,128,1)'
    if 'learning' in line:
        parameters[line] = 'hp.uniform("'+line+'",.000000001,.0001)'
    if 'dense' in line:
        parameters[line] = 'hp.quniform("'+line+'",256,2048,1)'
print(parameters)