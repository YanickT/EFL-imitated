from CreateData import DataGenerator
from Network import Network

import RBM
import matplotlib.pyplot as plt

size = 32
gen = DataGenerator(32, 2, mass=0., c=4., batch_size=7)
network = Network([32, 16, 8, 4, 2], c=4.0, flavor=2)

energys = []
for i, batch in zip(range(40), gen):
    print(i)
    # FIXME: Training bei Netzwerk ausgeschalten (opimizer wird nicht mehr aufgerufen)
    loss = network.train_it(*batch, its=16)
    # plt.plot(list(range(len(loss))), loss)
    # plt.xlabel("Iterations")
    # plt.ylabel("MSE")
    # plt.show()
    RBM.J = network.J
    energys.append((RBM.get_free_energy(batch[0]), loss[-1][1].tolist()))



y, x = list(zip(*energys))
x = [-e for batch in x for e in batch]
y = [e for batch in y for e in batch]

fig, axs = plt.subplots(2)
fig.suptitle('2nd Reni entropies')
axs[0].plot(y, x, ".")
axs[0].set_xlabel("SOLL")
axs[0].set_ylabel("IST")

axs[1].plot(list(range(len(y))), [ex / ey for ex, ey in zip(x, y)])
axs[1].set_xlabel("STEP")
axs[1].set_ylabel("IST / SOLL")

axs[0].grid()
axs[1].grid()
plt.show()


