import numpy as np


def cross_layer_v1(x0, xl, w, b):
    # TODO: x0 * (xl @ w)[:, None] + b + xl
    return x0 * (xl @ w)[:, None] + b + xl


def cross_layer_v2(x0, xl, W, b):
    # TODO: x0 * (xl @ W.T + b) + xl
    return x0 * (xl @ W.T + b) + xl


def cross_network(x0, params, version):
    # TODO: apply the layers in order from xl = x0; return (x_L, params_per_layer)
    d = x0.shape[1]
    xl = x0
    for i, (w, b) in enumerate(params):
        if version == 1:
            xl = cross_layer_v1(x0, xl, w, b)
        else:
            xl = cross_layer_v2(x0, xl, w, b)
    params_per_layer = 2 * d if version == 1 else d * d + d
    return xl, params_per_layer
