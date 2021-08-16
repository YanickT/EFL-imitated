from typing import List
import torch
import math


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
        self.h = flavor * c * math.log(2) / 2
        self.h0 = self.h * torch.ones(self.stats["size"], dtype=torch.float64)

        # trainable parameter
        self.J = torch.nn.Parameter(0.27 * torch.ones(self.stats["depth"], dtype=torch.float64))

        # setup optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002, betas=(0.9, 0.9))

    def index(self, layer, cell, position):
        if self.indices[layer, cell, position] == -1:
            self.indices[layer, cell, position] = self.nodes
            self.nodes += 1
        return self.indices[layer, cell, position]

    def fill_layer(self):
        for layer_i, (layeruv, layerir) in enumerate(zip([0] + self.structure, self.structure + [0])):
            pattern = [1] if layer_i == 0 else [-1] if layer_i == self.stats["depth"] else self.pattern
            layer = {-1: [], 1: []}
            for i in range(layeruv + layerir):
                layer[pattern[i % len(pattern)]].append(i)
            self.layers.append(layer)

    def fill_adjacent(self):
        # See: https://github.com/EverettYou/EFL
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

    def train_it(self, data, solutions, its=1):
        h_s = self.h * torch.from_numpy(data)

        ah0 = torch.tensordot(torch.exp(2. * self.h0), self.awh, dims=1)
        ah_s = torch.tensordot(torch.exp(2. * h_s), self.awh, dims=1)

        losses = []
        for i in range(its):
            aJ = torch.tensordot(torch.exp(2. * self.J), self.awJ, dims=1)

            a_s = ah_s + self.a1 + aJ
            a0 = ah0 + self.a1 + aJ

            f0 = -0.5 * torch.linalg.slogdet(a0)[1] + torch.sum(self.J * self.count_J) + torch.sum(self.h0 * self.count_h)
            fs = -0.5 * torch.linalg.slogdet(a_s)[1] + torch.sum(self.J * self.count_J) + (h_s * self.count_h).sum(-1)

            loss = torch.mean(torch.square(torch.sub(fs, f0) / torch.from_numpy(solutions) - 1.))
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return losses
