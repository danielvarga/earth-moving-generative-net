import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys

import next_permutation
import munkres

def pretty(m):
    for row in m:
        print "\t".join(map(str, row))

def halfCircle():
    x = 1.0
    y = 1.0
    while x*x+y*y>1.0:
        x = random.uniform( 0.0, +1.0)
        y = random.uniform(-1.0, +1.0)
    return (x,y)

def wave():
    x = random.uniform( -math.pi, +math.pi)
    y = math.sin(x)+random.uniform( -0.2, +0.2)
    return (x,y)

def triangle():
    x = random.uniform(-1.0, +1.0)
    y = random.uniform(-1.0, x)
    return (x,y)


def sampleFromTarget():
    # return wave()
    # return halfCircle()
    return triangle()

def samplesFromTarget(n):
    return np.array([sampleFromTarget() for i in xrange(n)])

def samplesFromInit(n, d, e):
    norm = np.random.normal(loc=0.0, scale=1.0, size=(n,e))
    z = np.zeros((n,d-e))
    data = np.hstack((norm, z))
    assert data.shape==(n,d)
    return data

# Both are (n x d) arrays.
def sumOfDistances(x,y):
    return np.sum(np.linalg.norm(x-y, axis=1))

# Both are (n x d) arrays.
# Scales with O(n!) boo!
# We could bring it down by reducing it to minimum-weight
# matching on a complete bipartite graph.
# If we need really large n, then a sequential
# greedy alg is probably more than good enough.
# Probably we'll have something partially parallel that's even
# faster than the naive sequential greedy alg.
def slowOptimalPairing(x,y):
    n,d = x.shape
    assert y.shape==(n,d)
    bestDist = np.inf
    bestP = None
    for p in next_permutation.next_permutation(range(n)):
        dist = sumOfDistances(x[p],y)
        if dist<bestDist:
            bestDist = dist
            bestP = p
    return bestP

def distanceMatrix(x, y):
    xL2S = np.sum(np.abs(x)**2,axis=-1)
    yL2S = np.sum(np.abs(y)**2,axis=-1)
    xL2SM = np.tile(xL2S, (len(y), 1))
    yL2SM = np.tile(yL2S, (len(x), 1))
    squaredDistances = xL2SM + yL2SM.T - 2.0*y.dot(x.T)
    distances = np.sqrt(squaredDistances+1e-6) # elementwise. +1e-6 is to supress sqrt-of-negative warning.
    return distances

def optimalPairing(x, y):
    distances = distanceMatrix(x,y)
    perm = munkres.Munkres().compute(distances)
    p = []
    for i,(a,b) in enumerate(perm):
        assert i==a
        p.append(b)
    # assert p==slowOptimalPairing(x,y)
    return p


class LocalMapping(object):
    KERNEL_SIZE = 0.33 # Ad hoc is an understatement
    def __init__(self, source, gradient):
        self.source = source
        self.gradient = gradient
    # TODO Make it work on arrays as well.
    def __call__(self, x):
        y = x.copy()
        dists = np.linalg.norm(self.source-x, axis=1)
        for (d, g) in zip(dists, self.gradient):
            if d<self.KERNEL_SIZE:
                y += g * (self.KERNEL_SIZE-d) / self.KERNEL_SIZE
        return y

def testLocalMapping():
    d = 2
    source = np.array(  [[0.0, 0.0], [ 1.0, 1.0]])
    gradient = np.array([[0.5, 0.5], [-0.5, 0.5]])
    f = LocalMapping(source, gradient)
    np.testing.assert_array_almost_equal( f(np.array([1.0,  1.0])), np.array([0.95,  1.05]) )
    np.testing.assert_array_almost_equal( f(np.array([0.09, 0.0])), np.array([0.095, 0.005]) )


class GlobalMapping(LocalMapping):
    def __init__(self, source, gradient, ancestor):
        super(GlobalMapping, self).__init__(source, gradient)
        self.ancestor = ancestor
    def __call__(self, x):
        if self.ancestor is None:
            return super(GlobalMapping, self).__call__(x)
        else:
            intermediate = self.ancestor(x)
            return super(GlobalMapping, self).__call__(intermediate)


def testGlobalMapping():
    source = np.array(  [[0.0, 0.0], [ 1.0, 1.0]])
    gradient = np.array([[0.5, 0.5], [-0.5, 0.5]])
    f = GlobalMapping(source, gradient, None)
    g = GlobalMapping(source, gradient, f)
    x = np.array([1.0,  1.0])
    np.testing.assert_array_almost_equal( f(f(x)), g(x) )

# testLocalMapping()
# testGlobalMapping()

def drawMapping(ax, f):
    n = 30
    window = 2
    ax.set_xlim((-window, +window))
    ax.set_ylim((-window, +window))
    for x in np.linspace(-window, +window, num=n):
        for y in np.linspace(-window, +window, num=n):
            x2, y2 = f(np.array([x, y]))
            ax.arrow(x, y, x2-x, y2-y, head_width=0.05, head_length=0.1, fc='k', ec='k')

# n is the number of data points.
# e is the dimension of the initial Gaussian.
# (But it's embedded in the d-dimensional feature space.)
# f is the actual mapping, a python function
# mapping from R^d to R^d.
def findMapping(n, e, f, learningRate):
    y = samplesFromTarget(n)
    d = y.shape[1]
    # TODO It's dumb not to init with the e principal components of the data.
    init = samplesFromInit(n, d, e)
    # TODO vectorize f
    x = np.array([f(i) for i in init])
    p = optimalPairing(x,y)
    x = x[p]
    # pretty(np.hstack((x,y)))
    source, gradient = x, learningRate*(y-x)
    dumpMapping = False
    if dumpMapping:
        f = LocalMapping(source, gradient)
        for xp,yp in zip(x,y):
            print xp, yp, f(xp), np.linalg.norm(xp-yp), np.linalg.norm(f(xp)-yp)
    return source, gradient

def iteration():
    d = 2
    e = 2
    n = 50
    learningRate = 0.3
    minibatchCount = 50
    plotEvery = 10
    plotCount = minibatchCount/plotEvery

    f = GlobalMapping(np.zeros((0,d)), np.zeros((0,d)), None)
    gaussSample = samplesFromInit(100, d, e)

    fig, axarr = plt.subplots(minibatchCount/plotEvery, 3)
    fig.set_size_inches(10.0*2, 10.0*plotCount)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i in range(minibatchCount):
        source, gradient = findMapping(n, e, f, learningRate)
        # That's much the same as
        # f = lambda x: LocalMapping(source, gradient)(f(x))
        f = GlobalMapping(source, gradient, f)
        print i,
        sys.stdout.flush()
        if i%plotEvery==0:
            plotIndex = i/plotEvery
            drawMapping(axarr[plotIndex][0], f)
            drawMapping(axarr[plotIndex][1], LocalMapping(source, gradient))
            sampleFromTarget = samplesFromTarget(100)
            axarr[plotIndex][2].scatter(sampleFromTarget[:,0], sampleFromTarget[:,1], color='red')
            sampleFromLearned = np.array([ f(p) for p in gaussSample ])
            axarr[plotIndex][2].scatter(sampleFromLearned[:,0], sampleFromLearned[:,1])

    print
    plt.savefig("vis.pdf")

def iterationMNIST():
    d = 784
    e = 10
    n = 50
    learningRate = 1.0
    minibatchCount = 90
    plotEvery = 10
    plotCount = minibatchCount/plotEvery

    f = GlobalMapping(np.zeros((0,d)), np.zeros((0,d)), None)
    gaussSample = samplesFromInit(100, d, e)

    fig, axarr = plt.subplots(minibatchCount/plotEvery, 3)
    fig.set_size_inches(10.0*2, 10.0*plotCount)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i in range(minibatchCount):
        source, gradient = findMapping(n, e, f, learningRate)
        # That's much the same as
        # f = lambda x: LocalMapping(source, gradient)(f(x))
        f = GlobalMapping(source, gradient, f)
        print i,
        sys.stdout.flush()
        if i%plotEvery==0:
            plotIndex = i/plotEvery
            drawMapping(axarr[plotIndex][0], f)
            drawMapping(axarr[plotIndex][1], LocalMapping(source, gradient))
            sampleFromTarget = samplesFromTarget(100)
            axarr[plotIndex][2].scatter(sampleFromTarget[:,0], sampleFromTarget[:,1], color='red')
            sampleFromLearned = np.array([ f(p) for p in gaussSample ])
            axarr[plotIndex][2].scatter(sampleFromLearned[:,0], sampleFromLearned[:,1])

    print
    plt.savefig("vis.pdf")

def mnist():
    datasetFile = "../rbm/data/mnist.pkl.gz"
    f = gzip.open(datasetFile, 'rb')
    datasets = cPickle.load(f)
    train_set, valid_set, test_set = datasets
    f.close()
    return train_set

def main():
    iteration()


if __name__ == "__main__":
    main()
