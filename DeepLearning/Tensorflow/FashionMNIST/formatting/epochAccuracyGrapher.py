import matplotlib.pyplot as plt
import os 

x = [i+1 for i in range(10)]
epochAccuracy = [[0.7315, 0.8501, 0.8695, 0.8857, 0.8958, 0.9042, 0.9112, 0.9177, 0.9216, 0.9265], [0.5663, 0.7395, 0.7848, 0.8123, 0.8306, 0.8435, 0.8531, 0.8596, 0.8663, 0.8695], [0.6089, 0.7643,0.8049, 0.8277, 0.8422, 0.8534, 0.863, 0.8683, 0.8772, 0.8808], [0.7971, 0.8741, 0.8931, 0.9044, 0.9138, 0.9214, 0.9274, 0.9337, 0.9406, 0.9452], [0.6274, 0.7767, 0.8142, 0.8311, 0.844, 0.8551, 0.8616, 0.8665, 0.8718, 0.8772]]
testingAccuracy = [91.32000000000001, 88.1, 89.45, 91.0, 88.34]
plt.xlabel("First Epoch Accuracy")
plt.ylabel("Testing Accuracy")
plt.title("Testing vs Training")
firstEpoch = list()
for i in range(len(testingAccuracy)):
    firstEpoch.append(epochAccuracy[i][9])
plt.scatter(firstEpoch, testingAccuracy)
plt.legend()
plt.show()