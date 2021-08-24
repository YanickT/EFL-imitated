import torch
import numpy as np


J = []


def wrapper(dim_in, dim_out, layer):
    weight = torch.tensor(
        [[J[layer] if 2 * i == j or (2 * i == j - 1) else 0 for j in range(dim_out)] for i in range(dim_in)],
        dtype=torch.float64)

    def f(value):
        free_energy = - torch.sum(torch.log(1 + torch.exp(value @ weight)))
        result = 2 * (torch.sigmoid(value @ weight).bernoulli()) - 1
        return result, free_energy

    return f


def get_free_energy(batchs):
    layers_n = [32, 16, 8, 4, 2]
    layers = [wrapper(layers_n[i], layers_n[i + 1], i) for i in range(len(layers_n) - 1)]
    fs = []
    for batch in batchs:
        datas = [(torch.from_numpy(batch),)]
        for layer in layers:
            datas.append(layer(datas[-1][0]))

        # data = zip(activations, wi)
        acs, fs_ = tuple(zip(*datas[1:]))
        fs.append(sum(fs_))
    return fs