from typing import List
import numpy as np


class DataGenerator:

    """
    Generator class presenting the data for the EFL.
    Attention! Never ending generator use next() or zip(range(n), DataGenerator) for limitation.
    """

    def __init__(self, size: int, flavors: int, n: int = 2, mass: float = 0, c: float = 1., batch_size: int = 7):
        """
        Initialize the generator class
        :param size: number of lattice sides (1D chain)
        :param flavors: number of fermion flavors in the lattice
        :param n: mean number of partitions
        :param mass: mass of the fermion
        :param c: central charge of the CFT
        :param batch_size: number of samples per iteration
        """
        # lattice stats
        self.size = size
        self.flavors = flavors
        self.n = n
        self.ns = np.arange(1, self.size // 2 + 1)
        probs = np.exp(-self.ns / n)
        self.probs = probs / np.sum(probs)
        self.flav_regions = np.arange(self.size * self.flavors).reshape([self.size, self.flavors])

        # fermion stats
        self.mass = mass
        self.c = c

        # See: https://github.com/EverettYou/EFL
        # construct single-particle density matrix
        u = np.tile([1. + self.mass, 1. - self.mass], (self.size * self.flavors) // 2)  # hopping
        A = np.roll(np.diag(u), 1, axis=1)  # periodic rolling
        A[-1, 0] = -A[-1, 0]  # antiperiodic boundary condition, to avoid zero mode
        A = 1j * (A - np.transpose(A))  # Hamiltonian
        (_, U) = np.linalg.eigh(A)  # digaonalization
        V = U[:, :(self.size * self.flavors) // 2]  # take lower half levels
        self.G = np.dot(V, V.conj().T)  # construct density matrix

        # Generator stats
        self.bs = batch_size

    def __iter__(self):
        """
        Creates a generator instance.
        :return: A generator instance which creates new data each iteration.
        """
        while True:
            ns = np.random.choice(self.ns, size=self.bs, p=self.probs)
            data = np.ones((self.bs, self.size))
            ents = np.zeros((self.bs,))

            for i, n in enumerate(ns):
                partitions = np.random.choice(self.size, size=2*n, replace=False)
                partitions.sort()
                regions = []
                for j in range(0, len(partitions) - 1, 2):
                    regions += list(range(partitions[j], partitions[j+1]))

                # create region configuration
                data[i][regions] = -1

                # calculate entanglement
                ents[i] = self.entanglement_entropy(regions)

            yield data, ents

    def __next__(self):
        """
        For next() Method on class.
        :return: Random Dataset (Batch) for Training.
        """
        for data in self:
            return data

    def entanglement_entropy(self, regions: List[int]) -> np.float:
        """
        Calculate the entanglement entropy for Fermions.
        See: https://github.com/EverettYou/EFL
        :param regions: List indices describing regions
        :return: entanglement entropy
        """
        flav_region = self.flav_regions[regions].flatten()
        p = np.linalg.eigvalsh(self.G[flav_region, :][:, flav_region])
        return - self.c * np.sum(np.log(np.abs(p ** 2 + (1. - p) ** 2)))
