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

# get the size of the data-set

train = []
for i, img in enumerate(training[0]):
    temp0 = [0] * 28 * 7
    for j, scale in enumerate(img):
        temp0[j // 4] += scale
    temp1 = [0] * 7 * 7
    for j, scale in enumerate(temp0):
        temp1[(j // 28) * 7 + j % 7] += scale
    temp1[:] = [x // 40 for x in temp1]
    temp1.append(training[1][i])
    train.append(temp1)

test = []
for i, img in enumerate(testing[0]):
    temp0 = [0] * 28 * 7
    for j, scale in enumerate(img):
        temp0[j // 4] += scale
    temp1 = [0] * 7 * 7
    for j, scale in enumerate(temp0):
        temp1[(j // 28) * 7 + j % 7] += scale
    temp1[:] = [x // 40 for x in temp1]
    temp1.append(testing[1][i])
    test.append(temp1)

# get the size of the data-set
label = 49
K = 10
n = len(train)
M = 500
W = [1. / n] * n
C = []
Error = []

for m in range(M):
    CLT = CL.ChowLiuTree(train, label, W)
    e = CLT.error_rate()
    C.append([CLT.lb_degree, CLT.lb_margin, CLT.lb_nb_pair_margin, math.log((1 / e - 1) * (K - 1))])
    for i in range(n):
        W[i] = W[i] * (K - 1) / (K * e) if CLT.cache[i] == 0 else W[i] / (K * (1 - e))
    print("Boosting", m, "round completed, e =", e)

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
