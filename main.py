from CreateData import DataGenerator
from Network import Network
import matplotlib.pyplot as plt

size = 32
gen = DataGenerator(32, 2, mass=0., c=4., batch_size=7)
network = Network([32, 16, 8, 4, 2], c=4.0, flavor=2)

energys = []
for i, batch in zip(range(40), gen):
    loss = network.train_it(*batch, its=16)
    print(network.J)




