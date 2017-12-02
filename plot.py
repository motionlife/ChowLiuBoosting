import matplotlib.pyplot as plt

data = []
with open('results/mnist7-7.txt', 'r') as f:
    for line in f:
        data.append(1-float(line.split()[-1]))

fig = plt.figure()
plt.plot(data)
fig.suptitle("MNIST-Random NB 7-7")
plt.ylabel('Training Error')
plt.xlabel("Boosting Rounds")
plt.show()
