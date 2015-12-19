import time
import sys

import numpy as np
import theano
import theano.tensor as T
import lasagne


import theano.sandbox.rng_mrg

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
    return np.random.normal(size=n*f).astype(np.float32).reshape((n, f))


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
def constructSquaredDistanceMatrixVariable(x, y, n, m):
    # ([n, f] , [m, f]) -> (n, m)
    xL2S = T.sum(x*x, axis=-1) # [n]
    yL2S = T.sum(y*y, axis=-1) # [m]
    xL2SM = T.zeros((m, n)) + xL2S # broadcasting, [m, n]
    yL2SM = T.zeros((n, m)) + yL2S # # broadcasting, [n, m]

    squaredDistances = xL2SM.T + yL2SM - 2.0*T.dot(x, y.T) # [n, m]
    return squaredDistances

def constructSDistanceMatrixFunction(n, m):
    x = T.matrix('x')
    y = T.matrix('y')
    sDistances = constructSquaredDistanceMatrixVariable(x, y, n, m)
    return theano.function([x, y], sDistances)

# For each y, it returns the index of the closest x in L2 distance.
# x is [n, f], y is [m, f] for some f. Output is [m], the values are in range(n).
def constructMinimalDistanceIndicesVariable(x, y, n, m):
    sDistances = constructSquaredDistanceMatrixVariable(x, y, n, m)
    lamblinsTrick = False
    if lamblinsTrick:
        # https://github.com/Theano/Theano/issues/1399
        # https://gist.github.com/danielvarga/d0eeacea92e65b19188c
        # https://groups.google.com/forum/#!topic/theano-users/E7ProqnGUMk
        s = sDistances
        bestIndices = T.cast( ( T.arange(n).dimshuffle(0, 'x') * T.cast(T.eq(s, s.min(axis=0, keepdims=True)), 'float32') ).sum(axis=0), 'int32')
        # This is a heavy-handed workaround for the fact that in
        # lamblin's hack, ties lead to completely screwed results.
        bestIndices = T.clip(bestIndices, 0, n-1)
    else:
        bestIndices = T.argmin(sDistances, axis=0)
    return bestIndices


# The theano.function returned by this is usually called like this:
# closest_fn(sampled, data), output is an index (pointing to a sampled row)
# for each row of data.
#
# To elaborate:
#  n = candidateCount, m = targetCount,
# typically candidateCount = sampleSize, targetCount = minibatchSize
# BEWARE: super confusingly, in generativeMLP.py sampleSize is called m.
#
# See testMinimalDistanceIndicesFunction for how to turn indices into samples and distances.
def constructMinimalDistanceIndicesFunction(n, m):
    x = T.matrix('x')
    y = T.matrix('y')
    bestIndices = constructMinimalDistanceIndicesVariable(x, y, n, m)
    return theano.function([x, y], bestIndices)

def constructMinimalDistancesVariable(x, y, initials, n, m):
    sDistances = constructSquaredDistanceMatrixVariable(x, y, n, m)
    bestIndices = T.argmin(sDistances, axis=0)
    bestXes = x[bestIndices]
    bestInitials = initials[bestIndices]
    return bestXes, bestInitials

def constructMinimalDistancesFunction(n, m):
    x = T.matrix('x')
    y = T.matrix('y')
    initials = T.matrix('initials')
    bestXes, bestInitials = constructMinimalDistancesVariable(x, y, initials, n, m)
    return theano.function([x, y], bestXes)


def testMinimalDistanceIndicesFunction(batchSize, sampleSize, featureDim):
    closest_fn = constructMinimalDistanceIndicesFunction(sampleSize, batchSize)

    data = randomMatrix(batchSize, featureDim)
    sampled = randomMatrix(sampleSize, featureDim)

    import kohonen
    start("CPU nearest neighbors")
    distances = kohonen.distanceMatrix(sampled, data)
    assert distances.shape == (len(data), len(sampled)) # Beware the transpose!
    bestIndicesByCPU = np.argmin(distances, axis=1)
    closestSampledByCPU = sampled[bestIndicesByCPU]
    bestDistancesByCPU = np.linalg.norm(data-closestSampledByCPU, axis=1)
    end()

    start("GPU nearest neighbors")
    bestIndicesByGPU = closest_fn(sampled, data)
    # The next two steps are practically instant.
    closestSampledByGPU = sampled[bestIndicesByGPU]
    bestDistancesByGPU = np.linalg.norm(data-closestSampledByGPU, axis=1)
    end()

    print "total bestDistances CPU", bestDistancesByCPU.sum()
    print "total bestDistances GPU", bestDistancesByGPU.sum()




# A cool little toy learning problem:
# We want to learn a translated 2D standard normal's translation, that's a 2D vector.
# We generate batchSize samples from this target distribution.
# We generate sampleSize samples from our current best bet for the distribution.
# We find the closest generated sample to each target sample.
# We calculate the sum of distances.
# That's the loss that we optimize by gradient descent.
# Note that Theano doesn't even break a sweat when doing backprop
# through a layer of distance minimization.
# Of course that's less impressive than it first sounds, because
# locally, the identity of the nearest target sample never changes.
def toyLearner():
    batchSize = 2000
    sampleSize = 2000
    inDim = 2
    srng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=234)

    dataVar = T.matrix("data")
    initialsVar = srng.normal((sampleSize, inDim))
    parametersVar = theano.shared(np.zeros(inDim, dtype=np.float32), "parameters")
    generatedVar = initialsVar + parametersVar # broadcast


    bestXesVar, bestInitialsVar = constructMinimalDistancesVariable(generatedVar, dataVar, initialsVar, sampleSize, batchSize)

    deltaVar = bestXesVar - dataVar
    # mean over samples AND feature coordinates!
    # Very frightening fact: with .sum() here, the learning process diverges.
    lossVar = (deltaVar*deltaVar).mean()

    updates = lasagne.updates.nesterov_momentum(
            lossVar, [parametersVar], learning_rate=0.2, momentum=0.0)

    train_fn = theano.function([dataVar], updates=updates)

    for epoch in range(1000):
        data = randomMatrix(batchSize, inDim) + np.array([-5.0, 12.0], dtype=np.float32)
        train_fn(data)
        print parametersVar.get_value()

def distanceSpeedTest():
    # I'm not using variable names n and m, because unfortunately
    # the order is switched between sampleAndUpdate() and
    # constructDistanceMatrixFunction().
    batchSize = 3000
    oversampling = 4.324
    sampleSize = int(batchSize*oversampling)
    f = 28*28
    np.random.seed(0)
    data = randomMatrix(batchSize, f)
    generated = randomMatrix(sampleSize, f)

    dm_fn = constructSDistanceMatrixFunction(sampleSize, batchSize)

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

    start("all distances numpy")
    ds = distanceMatrix(generated, data)
    print ds.shape
    print np.sum(ds)
    end()

if __name__ == "__main__":
    whichTest = sys.argv[1]
    assert whichTest in ("distances", "toyLearner", "speeds")
    if whichTest=="distances":
        testMinimalDistanceIndicesFunction(batchSize=3000, sampleSize=12972, featureDim=28*28)
    elif whichTest=="speeds":
        distanceSpeedTest()
    elif whichTest=="toyLearner":
        toyLearner()
