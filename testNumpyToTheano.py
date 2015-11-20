import time
import sys

import numpy as np
import theano
import theano.tensor as T

def logg(*ss):
    s = " ".join(map(str,ss))
    sys.stderr.write(s+"\n")

def start(s):
    global startTime
    global phase
    phase = s
    logg(phase+".")
    startTime = time.clock()

def end(s=None):
    global startTime
    global phase
    if s is not None:
        phase = s
    endTime = time.clock()
    logg(phase,"finished in",endTime-startTime,"seconds.")


def randomMatrix(n, f):
    return np.random.uniform(size=n*f).reshape((n, f))

def distanceMatrixSlow(x, y):
    xL2S = np.sum(np.abs(x)**2,axis=-1)
    yL2S = np.sum(np.abs(y)**2,axis=-1)
    xL2SM = np.tile(xL2S, (len(y), 1))
    yL2SM = np.tile(yL2S, (len(x), 1))
    squaredDistances = xL2SM + yL2SM.T - 2.0*y.dot(x.T)
    distances = np.sqrt(squaredDistances+1e-6) # elementwise. +1e-6 is to supress sqrt-of-negative warning.
    return distances

def distanceMatrix(x, y):
    xL2S = np.sum(x*x, axis=-1)
    yL2S = np.sum(y*y, axis=-1)
    xL2SM = np.tile(xL2S, (len(y), 1))
    yL2SM = np.tile(yL2S, (len(x), 1))
    squaredDistances = xL2SM + yL2SM.T - 2.0*y.dot(x.T)
    distances = np.sqrt(squaredDistances+1e-6) # elementwise. +1e-6 is to supress sqrt-of-negative warning.
    return distances


# Newer theano builds allow tile() with scalar variable as reps.
# https://github.com/Theano/Theano/pull/2875
# That could make this nicer.
def constructDistanceMatrixVariable(x, y, n, m):
    # ([n, f] , [m, f]) -> (n, m)
    xL2S = T.sum(x*x, axis=-1) # [n]
    yL2S = T.sum(y*y, axis=-1) # [m]
    xL2SM = T.zeros((m, n)) + xL2S # broadcasting, [m, n]
    yL2SM = T.zeros((n, m)) + yL2S # # broadcasting, [n, m]

    squaredDistances = xL2SM.T + yL2SM - 2.0*T.dot(x, y.T) # [n, m]
    distances = T.sqrt(squaredDistances+1e-6)
    return distances

def constructDistanceMatrixFunction(n, m):
    x = T.matrix('x')
    y = T.matrix('y')
    distances = constructDistanceMatrixVariable(x, y, n, m)
    return theano.function([x, y], distances)

def constructMinimalDistancesVariable(x, y, initials, n, m):
    distances = constructDistanceMatrixVariable(x, y, n, m)
    bestIndices = T.argmin(distances, axis=1)
    bestXes = x[bestIndices]
    bestInitials = initials[bestIndices]
    return bestXes, bestInitials

def constructMinimalDistancesFunction(n, m):
    x = T.matrix('x')
    y = T.matrix('y')
    initials = T.matrix('initials')
    bestXes, bestInitials = constructMinimalDistancesVariable(x, y, initials, n, m)
    return theano.function([x, y], bestXes)

def test():
    # I'm not using variable names n and m, because unfortunately
    # the order is switched between sampleAndUpdate() and
    # constructDistanceMatrixFunction().
    batchSize = 3000
    oversampling = 4.324
    sampleSize = int(batchSize*oversampling)
    f = 28*28
    data = randomMatrix(batchSize, f)
    generated = randomMatrix(sampleSize, f)

    dm_fn = constructDistanceMatrixFunction(sampleSize, batchSize)

    md_fn = constructMinimalDistancesFunction(sampleSize, batchSize)

    start("minimal distances theano")
    bestXes = md_fn(generated, data)
    print bestXes.shape
    print np.sum(bestXes)
    end()

    start("all distances theano")
    ds = dm_fn(generated, data)
    print ds.shape
    print np.sum(ds)
    end()

    start("all distances slow numpy")
    ds = distanceMatrixSlow(generated, data)
    print ds.shape
    print np.sum(ds)
    end()

    start("all distances fast numpy")
    ds = distanceMatrix(generated, data)
    print ds.shape
    print np.sum(ds)
    end()

test()
