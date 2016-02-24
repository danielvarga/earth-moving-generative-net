import time
import sys

import numpy as np
import theano
import theano.tensor as T
import lasagne

import distances


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


    bestXesVar, bestInitialsVar = distances.constructMinimalDistancesVariable(generatedVar, dataVar, initialsVar, sampleSize, batchSize)

    deltaVar = bestXesVar - dataVar
    # mean over samples AND feature coordinates!
    # Very frightening fact: with .sum() here, the learning process diverges.
    lossVar = (deltaVar*deltaVar).mean()

    updates = lasagne.updates.nesterov_momentum(
            lossVar, [parametersVar], learning_rate=0.2, momentum=0.8)

    train_fn = theano.function([dataVar], updates=updates)

    for epoch in range(1000):
        data = distances.randomMatrix(batchSize, inDim) + np.array([-5.0, 12.0], dtype=np.float32)
        train_fn(data)
        print parametersVar.get_value()
        sys.stdout.flush()

if __name__ == "__main__":
    toyLearner()
