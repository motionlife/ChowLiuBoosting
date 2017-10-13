import matplotlib.pyplot as plt

data = []
with open('results/connect4-random10.txt', 'r') as f:
    for line in f:
        data.append(1-float(line.split()[-1]))

fig = plt.figure()
plt.plot(data)
fig.suptitle("Connect4-Random Tree")
plt.ylabel('Training Error')
plt.xlabel("Boosting Rounds")
plt.show()
