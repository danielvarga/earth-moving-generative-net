import numpy as np
import random

import next_permutation

def sampleFromTarget():
    x = 1.0
    y = 1.0
    while x*x+y*y>1.0:
        x = random.uniform( 0.0, +1.0)
        y = random.uniform(-1.0, +1.0)
    return (x,y)

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
def optimalPairing(x,y):
    n,d = x.shape
    assert y.shape==(n,d)
    bestDist = 1e50
    bestP = None
    for p in next_permutation.next_permutation(range(n)):
        dist = sumOfDistances(x[p],y)
        if dist<bestDist:
            bestDist = dist
            bestP = p
    return p

class LocalMapping(object):
    KERNEL_SIZE = 0.1 # Ad hoc is an understatement
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
    return x, learningRate*(y-x)

def pretty(m):
    for row in m:
        print "\t".join(map(str, row))

def iteration():
    d = 2
    e = 2
    n = 8
    learningRate = 0.1
    f = GlobalMapping(np.zeros((0,d)), np.zeros((0,d)), None)
    source, gradient = findMapping(n, e, f, learningRate)
    pretty(source)
    print
    pretty(gradient)

def permTest():
    n = 3
    x = samplesFromTarget(n)
    for p in next_permutation.next_permutation(range(n)):
        print x[p]

def main():
    iteration()

main()

