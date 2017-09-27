import csv
import ChowLiu as CL

"""
construct experimental data set
"""
data_train = []
data_test = []
label = 36
counter = 0
with open("kr-vs-kp.data") as file:
    content = csv.reader(file, delimiter=',')
    for row in content:
        counter += 1
        x = ""
        if row[-1] == 'won':
            row[-1] = '1'
        else:
            row[-1] = '0'
        for r in row:
            x += r

        if counter % 2 == 0:
            data_train.append(x)
        else:
            data_test.append(x)

# build chow-liu tree
CLT = CL.ChowLiuTree(data_train, label)
print(CLT.accuracy(data_test))
print(CLT.label_margin)
print(CLT.lb_nb_pair_margin)
print(CLT.tree)
print(CLT.degree)
