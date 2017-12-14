"""
The implementation of SAMME algorithm for multi-class Adaboost
Revised with updating weight in one iteration
Base classifier: Chou-liu tree
"""
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from mnist import MNIST

import ChowLiu as CL

"""
construct experimental data set, data should be in the form of list of list
label is the position of class
"""
start_time = time.time()
mndata = MNIST('data/MNIST')
training = mndata.load_training()
testing = mndata.load_testing()

for i, img in enumerate(training[0]):
    img[:] = [1 if x > 7 else 0 for x in img]
    img.append(training[1][i])
train = training[0]

for i, img in enumerate(testing[0]):
    img[:] = [1 if x > 7 else 0 for x in img]
    img.append(testing[1][i])
test = testing[0]

# get the size of the data-set
label = 784
K = 10
n = len(train)
M = 700
W = [1. / n] * n
C = []
Error = []


def benchmark(data, models):
    correct = 0.
    for d in data:
        votes = defaultdict(float)
        for model in models:
            votes[CL.predict_label(d, model[0])] += model[1]
        if d[label] == max(votes, key=votes.get):
            correct += 1
    correct = correct / len(data)
    Error.append(1 - correct)
    log = "ChowLiu Boosting Round: " + str(len(models)) + " Accuracy:" + str(correct)
    f = open('results/mnist-ChowLiuTree.txt', 'a')
    f.write('\n' + log)
    f.close()
    print(log)
    return correct


for m in range(M):
    CLT = CL.ChowLiuTree(train, label, W)
    e = CLT.error
    C.append([CLT, math.log((1 / e - 1) * (K - 1))])
    for i in range(n):
        W[i] = W[i] * (K - 1) / (K * e) if CLT.cache[i] == 0 else W[i] / (K * (1 - e))
    if benchmark(test, C) == 1:
        break

print("The running time is: ", time.time() - start_time)
fig = plt.figure()
plt.plot(Error)
fig.suptitle("MNIST")
plt.ylabel('Training Error')
plt.xlabel("Boosting Rounds")
plt.show()
