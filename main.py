from CreateData import DataGenerator
from Network import Network
import math


size = 32
gen = DataGenerator(size, 2, mass=0., c=4., batch_size=7)
structure = [2**i for i in range(1, int(math.log(size, 2)) + 1)][::-1]
network = Network(structure, c=4.0, flavor=2)

for i, batch in zip(range(40), gen):
    loss = network.train_it(*batch, its=16)
    print(network.J)




