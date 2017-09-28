"""
The implementation of SAMME algorithm for multi-class Adaboost
Revised with updating weight in one iteration
Base classifier: Chou-liu tree
"""
import csv
import time
import math

import ChowLiu as CL

"""
construct experimental data set, data should be in the form of list of list
label is the position of class
"""
start_time = time.time()
data = []
label = 36
with open("data/kr-vs-kp.data") as file:
    content = csv.reader(file, delimiter=',')
    for row in content:
        x = ""
        if row[-1] == 'won':
            row[-1] = '1'
        else:
            row[-1] = '0'
        for r in row:
            x += r
        data.append(x)

# get the size of the data-set
n = len(data)
M = 5
W = [1. / n] * n
K = 2
C = {}
for m in range(M):
    CLT = CL.ChowLiuTree(data, label, W)
    C[m] = [CLT.lb_degree, CLT.lb_margin, CLT.lb_nb_pair_margin]
    e = CLT.error_rate()
    a = math.log((1 / e - 1) * (K - 1))
    C[m].append(a)
    for i in range(n):
        if CLT.predict[i] == 0:
            W[i] = W[i] * (K - 1) / (K * e)
        else:
            W[i] = W[i] / (K * (1 - e))

print("The running time is: ", time.time() - start_time)
print(C)
