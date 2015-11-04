import numpy as np

import theano
import theano.tensor as T
import lasagne

import kohonen
import nnbase.vis

# TODO If this functionality is important,
# TODO I'll probably have to rewrite it in Theano,
# TODO together with the workhorse kohonen.distanceMatrix().
# TODO Especially if the gradient descent based finetuning comes in.

def approximateMinibatch(data, net_fn, sampleSourceFunction, inDim, sampleForEach):
    n = len(data)
    initial, sampled = sampleSourceFunction(net_fn, sampleForEach, inDim)
    initial, sampled = sampleSourceFunction(net_fn, sampleForEach, inDim)
    distanceMatrix = kohonen.distanceMatrix(sampled, data)
    bestDists = np.argmin(distanceMatrix, axis=1)
    distances = np.min(distanceMatrix, axis=1)
    initial = initial[bestDists]
    sampled = sampled[bestDists]
    return initial, sampled, distances

# We generate sampleTotal data points, and for each gold data point
# we find the closest generated one.
def approximate(data, net_fn, sampleSourceFunction, inDim, sampleTotal):
    bestInitial, bestSampled, bestDistances = None, None, None
    # approximate_minibatch builds a matrix of size (len(data), sampleForEachMinibatch).
    # We want this matrix to fit into memory.
    distanceMatrixSizeLimit = int(1e6)
    sampleForEachMinibatch = distanceMatrixSizeLimit / len(data)
    batchCount = sampleTotal / sampleForEachMinibatch + 1
    for indx in xrange(batchCount):
        initial, sampled, distances = approximateMinibatch(data, net_fn, sampleSourceFunction, inDim, sampleForEachMinibatch)
        if bestDistances is None:
            bestInitial, bestSampled, bestDistances = initial, sampled, distances
        else:
            # Could easily vectorize but not a bottleneck.
            for i in range(len(bestDistances)):
                if distances[i]<bestDistances[i]:
                    bestInitial[i] = initial[i]
                    bestSampled[i]= sampled[i]
                    bestDistances[i] = distances[i]
    return bestInitial, bestSampled, bestDistances

def fitAndVis(data, net_fn, sampleSourceFunction, inDim, height, width, gridSizeForSampling, name):
    n = len(data)
    n_x = gridSizeForSampling
    n_y = gridSizeForSampling
    assert n <= n_x * n_y
    sampleTotal = int(1e6)

    initial, sampled, distances = approximate(data, net_fn, sampleSourceFunction, inDim, sampleTotal)

    # TODO The smart thing here would be to run a gradient descent
    # TODO on initial, further minimizing distances.

    totalDist = distances.sum()

    nnbase.vis.plot_distance_histogram(distances, name.replace("diff", "hist"))

    vis_n = min((n, n_x*n_y))
    nnbase.vis.diff_vis(data[:vis_n], sampled[:vis_n], height, width, n_x, n_y, name)

    return totalDist