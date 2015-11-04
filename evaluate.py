import numpy as np

import theano
import theano.tensor as T
import lasagne

import kohonen
import nnbase.vis


def fit(data, net_fn, sampleSourceFunction, inDim, height, width, gridSizeForSampling, name):
    n = len(data)
    n_x = gridSizeForSampling
    n_y = gridSizeForSampling
    assert n <= n_x * n_y
    m = 100*n
    initial, sampled = sampleSourceFunction(net_fn, m, inDim)
    distances = kohonen.distanceMatrix(sampled, data)
    bestDists = np.argmin(distances, axis=1)
    initial = initial[bestDists]
    sampled = sampled[bestDists]

    # TODO The smart thing here would be to run a gradient descent
    # TODO on initial, further minimizing distances.

    totalDist = distances.min(axis=1).sum()

    vis_n = min((n, n_x*n_y))
    nnbase.vis.diff_vis(data[:vis_n], sampled[:vis_n], height, width, n_x, n_y, name)

    return totalDist