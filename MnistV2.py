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
    img.append(training[1][i])
train = training[0]

for i, img in enumerate(testing[0]):
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

for m in range(M):
    CLT = CL.RandomNaiveBayes(train, label, W, 3, 7)
    C.append([CLT.lb_degree, CLT.lb_margin, CLT.lb_nb_pair_margin])
    e = CLT.error
    C[m].append(math.log((1 / e - 1) * (K - 1)))
    for i in range(n):
        W[i] = W[i] * (K - 1) / (K * e) if CLT.cache[i] == 0 else W[i] / (K * (1 - e))
    print("Boosting",i,"round compeleted")

# Bench mark the last 10 rounds
for i in reversed(range(10)):
    correct = 0.
    votes = defaultdict(float)
    for d in test:
        for model in C[:-i]:
            votes[CL.predict_label(d, None, model)] += model[-1]
        if d[label] == max(votes, key=votes.get):
            correct += 1
        votes.clear()
    correct /= len(test)
    Error.append(1 - correct)
    print("The accuracy for up to", len(C) - i, "rounds is:", correct)

print("The running time is: ", time.time() - start_time)
fig = plt.figure()
plt.plot(Error)
fig.suptitle("MNIST-Random Tree")
plt.ylabel('Training Error')
plt.xlabel("Boosting Rounds")
plt.show()
