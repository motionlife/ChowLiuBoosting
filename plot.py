import matplotlib.pyplot as plt

data = []
with open('results/car_sm1.txt', 'r') as f:
    for line in f:
        data.append(1-float(line.split()[-1]))

fig = plt.figure()
plt.plot(data)
fig.suptitle("MNIST-Random NB 7-7")
plt.ylabel('Training Error')
plt.xlabel("Boosting Rounds")
plt.savefig('results/car_sm1.png')
plt.show()
