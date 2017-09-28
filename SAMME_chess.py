"""
The implementation of SAMME algorithm for multi-class Adaboost
Revised with updating weight in one iteration
Base classifier: Chou-liu tree
"""
import csv
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt

import ChowLiu as CL

"""
construct experimental data set, data should be in the form of list of list
label is the position of class
"""
start_time = time.time()
data = []

with open("data/chess/kr-vs-kp.data") as file:
    content = csv.reader(file, delimiter=',')
    for row in content:
        data.append(row)

# get the size of the data-set
label = 36
K = 2
n = len(data)
M = 300
W = [1. / n] * n
C = {}
Error = []


def benchmark(t_data, C):
    correct = 0.
    scores = defaultdict(float)
    for d in t_data:
        for k, v in C.items():
            scores[CL.predict_label(d, None, v)] += v[-1]
        if d[label] == max(scores, key=scores.get):
            correct += 1
        scores.clear()
    correct /= len(t_data)
    Error.append(1 - correct)
    print("The accuracy for up to", len(C), "round is:", correct)


for m in range(M):
    CLT = CL.ChowLiuTree(data, label, W)
    C[m] = [CLT.lb_degree, CLT.lb_margin, CLT.lb_nb_pair_margin]
    e = CLT.error_rate()
    C[m].append(math.log((1 / e - 1) * (K - 1)))
    for i in range(n):
        if CLT.cache[i] == 0:
            W[i] = W[i] * (K - 1) / (K * e)
        else:
            W[i] = W[i] / (K * (1 - e))
    benchmark(data, C)

print("The running time is: ", time.time() - start_time)
fig = plt.figure()
plt.plot(Error)
fig.suptitle("CHESS")
plt.ylabel('Training Error')
plt.xlabel("Boosting Round")
plt.show()
