from typing import List
import torch
import numpy as np


# make results reproducible
torch.manual_seed(3841)


class Network(torch.nn.Module):

    def __init__(self, structure: List[int], flavor: int = 1, c: float = 0., pattern: List[int] = [-1, -1, 1]):
        """
        Creates a network based on the given structure.
        :param structure: List of Nodes per Layer
        :param flavor: Number of flavors the fermion system has
        :param c: central charge of the system
        :param pattern: Pattern how layers are built. -1: UV, 1: IR
        """
        super().__init__()

        self.structure = structure  # [32, 16, 8, 4, 2]
        self.pattern = pattern
        self.stats = {"depth": len(self.structure),
                      "size": self.structure[0],
                      "nodes": 6 * sum(self.structure)}  # 2 for doubling, 3 for cells

        # create layers
        self.layers = []
        self.fill_layer()

        # create adjacent matrix
        self.nodes = 0
        self.indices = torch.zeros((self.stats["depth"] + 1, self.structure[0] + self.structure[1], 3),
                                   requires_grad=False, dtype=torch.int32)  # [layer, cell, position]
        self.indices.fill_(-1)

        self.a1 = torch.zeros((self.stats["nodes"], self.stats["nodes"]), requires_grad=False, dtype=torch.float64)
        self.awJ = torch.zeros((self.stats["depth"] + 1, self.stats["nodes"], self.stats["nodes"]), requires_grad=False,
                               dtype=torch.float64)
        self.awh = torch.zeros((self.stats["size"], self.stats["nodes"], self.stats["nodes"]), requires_grad=False,
                               dtype=torch.float64)
        self.fill_adjacent()
        self.count_J = torch.absolute(self.awJ).sum(1).sum(1) / 2
        self.count_h = torch.absolute(self.awh).sum(1).sum(1) / 2

        # boundary conditions
        # See: https://github.com/EverettYou/EFL
        self.h = flavor * c * np.log(2) / 2
        self.h0 = self.h * torch.ones(self.stats["size"], dtype=torch.float64)

        # trainable parameter
        self.J = torch.nn.Parameter(0.27 * torch.ones(self.stats["depth"], dtype=torch.float64))

        # setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002, betas=(0.9, 0.9))

    def index(self, layer: int, cell: int, position: int) -> int:
        """
        Assigns a given node to its matrix position.
        :param layer: layer the node belongs to
        :param cell: cell the node belongs to
        :param position: position within the cell
        :return: index in the adjacent matrix
        """
        if self.indices[layer, cell, position] == -1:
            self.indices[layer, cell, position] = self.nodes
            self.nodes += 1
        return self.indices[layer, cell, position]

    def fill_layer(self):
        """
        Assigns the nodes and cells to the layers
        :return: void
        """
        for layer_i, (layeruv, layerir) in enumerate(zip([0] + self.structure, self.structure + [0])):
            pattern = [1] if layer_i == 0 else [-1] if layer_i == self.stats["depth"] else self.pattern
            layer = {-1: [], 1: []}
            for i in range(layeruv + layerir):
                layer[pattern[i % len(pattern)]].append(i)
            self.layers.append(layer)

    def fill_adjacent(self):
        """
        Fills the adjacent matrix for the given network structure.
        See: https://github.com/EverettYou/EFL
        :return: void
        """

        for layer_i, layer in enumerate(self.layers):
            for i in range(len(layer[1]) + len(layer[-1])):
                self.a1[self.index(layer_i, i, 0), self.index(layer_i, i, 1)] = 1
                self.a1[self.index(layer_i, i, 1), self.index(layer_i, i, 0)] = -1

            # uv cells
            for i in layer[-1]:
                self.a1[self.index(layer_i, i, 0), self.index(layer_i, i, 2)] = 1
                self.a1[self.index(layer_i, i, 2), self.index(layer_i, i, 0)] = -1
                self.a1[self.index(layer_i, i, 1), self.index(layer_i, i, 2)] = 1
                self.a1[self.index(layer_i, i, 2), self.index(layer_i, i, 1)] = -1

            # ir cells
            for i in layer[1]:
                self.a1[self.index(layer_i, i, 0), self.index(layer_i, i, 2)] = 1
                self.a1[self.index(layer_i, i, 2), self.index(layer_i, i, 0)] = -1
                self.a1[self.index(layer_i, i, 1), self.index(layer_i, i, 2)] = -1
                self.a1[self.index(layer_i, i, 2), self.index(layer_i, i, 1)] = 1

            # cell forwarding
            if layer_i == 0:
                for i in range(1, len(layer[1]) + len(layer[-1])):
                    self.awh[i, self.index(layer_i, i - 1, 1), self.index(layer_i, i, 0)] = 1
                    self.awh[i, self.index(layer_i, i, 0), self.index(layer_i, i - 1, 1)] = -1
                self.awh[0, self.index(layer_i, 0, 0), self.index(layer_i, len(layer[1]) + len(layer[-1]) - 1, 1)] = 1
                self.awh[0, self.index(layer_i, len(layer[1]) + len(layer[-1]) - 1, 1), self.index(layer_i, 0, 0)] = -1

            elif layer_i == self.stats["depth"]:
                for i in range(1, len(layer[1]) + len(layer[-1])):
                    self.a1[self.index(layer_i, i - 1, 1), self.index(layer_i, i, 0)] = 1
                    self.a1[self.index(layer_i, i, 0), self.index(layer_i, i - 1, 1)] = -1
                self.a1[self.index(layer_i, 0, 0), self.index(layer_i, len(layer[1]) + len(layer[-1]) - 1, 1)] = 1
                self.a1[self.index(layer_i, len(layer[1]) + len(layer[-1]) - 1, 1), self.index(layer_i, 0, 0)] = -1

            else:
                for i in range(1, len(layer[1]) + len(layer[-1])):
                    self.awJ[layer_i, self.index(layer_i, i - 1, 1), self.index(layer_i, i, 0)] = 1
                    self.awJ[layer_i, self.index(layer_i, i, 0), self.index(layer_i, i - 1, 1)] = -1
                self.awJ[
                    layer_i, self.index(layer_i, 0, 0), self.index(layer_i, len(layer[1]) + len(layer[-1]) - 1, 1)] = 1
                self.awJ[
                    layer_i, self.index(layer_i, len(layer[1]) + len(layer[-1]) - 1, 1), self.index(layer_i, 0, 0)] = -1

            # inter layer (downwards)
            if layer_i >= self.stats["depth"]:
                continue

            for i, j in zip(layer[1], self.layers[layer_i + 1][-1]):
                self.awJ[layer_i + 1, self.index(layer_i, i, 2), self.index(layer_i + 1, j, 2)] = 1
                self.awJ[layer_i + 1, self.index(layer_i + 1, j, 2), self.index(layer_i, i, 2)] = -1

        self.awJ = self.awJ[1:, :, :]

    def train_it(self, data: np.array, solutions: np.array, its: int = 1) -> List[float]:
        """
        Train the network with the given data.
        :param data: 2D array of configurations (input for the network) \forall x \in array: x \in {-1, 1}
        :param solutions: 1D array of floats (2nd Reni entropy of the network) for the given data
        :param its: number of iterations to train network with
        :return: list of losses (loss per iteration)
        """
        self.train()
        h_s = self.h * torch.from_numpy(data)

        ah0 = torch.tensordot(torch.exp(2. * self.h0), self.awh, dims=1)
        ah_s = torch.tensordot(torch.exp(2. * h_s), self.awh, dims=1)

        losses = []
        for i in range(its):
            aJ = torch.tensordot(torch.exp(2. * self.J), self.awJ, dims=1)

            a_s = ah_s + self.a1 + aJ
            a0 = ah0 + self.a1 + aJ

            f0 = -0.5 * torch.linalg.slogdet(a0)[1] + torch.sum(self.J * self.count_J) + torch.sum(
                self.h0 * self.count_h)
            fs = -0.5 * torch.linalg.slogdet(a_s)[1] + torch.sum(self.J * self.count_J) + (h_s * self.count_h).sum(-1)

            loss = torch.mean(torch.square(torch.sub(fs, f0) / torch.from_numpy(solutions) - 1.))
            losses.append((loss.item(), torch.sub(fs, f0).detach().numpy()))  # FIXME: loss gibt sachen mit aus
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.force_boundaries()
        self.eval()
        return losses

    def force_boundaries(self):
        """
        This is not an actual regularization.
        It forces the values to be in a certain rage and manually fixes them with min() and max().
        Furthermore, it does not force J_1 <= J_2 <= J_3 ... at each moment.
        Due to the simple shift in cat the following can occur:
        J (at the moment): 0.3778, 0.2750, 0.2748, 0.2751, 0.2755
        Comparison Tensor: 2.7726, 0.3778, 0.2750, 0.2748, 0.2751
        J (after the com): 0.3778, 0.2750, 0.2748, 0.2748, 0.2751
        :return:
        """
        with torch.no_grad():
            j = torch.maximum(self.J, torch.zeros(self.J.size()))
            h = torch.tensor([self.h], dtype=torch.float64)
            state_dict = self.state_dict()
            state_dict['J'] = torch.nn.Parameter(torch.minimum(self.J, torch.cat((h, j[:-1]))))
            self.load_state_dict(state_dict)