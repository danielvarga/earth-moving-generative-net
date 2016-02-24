import sys
import math

import numpy as np
import matplotlib.pyplot as plt


# A cool little toy learning problem:
# We want to learn a 1D distribution, e.g. uniform on (-1,+1).
# We want to model it with a gaussian mixture model.
# (Mixture of k 1D standard normals, parametrized by the k means.)
# We generate n samples from the target distribution.
# We generate n samples from our current best bet for the model.
# We find the pairing that minimizes the summed distance between paired points.
def toyLearner():
    n = 2000
    k = 100
    sigma = 0.05
    learningRate = 0.005
    epochCount = 100

    centers = np.random.normal(size=k).astype(np.float32)

    def generate(centers, n):
        picks = np.random.randint(k, size=n)
        currentCenters = centers[picks] # smart indexing
        generated = currentCenters + sigma * np.random.normal(size=n).astype(np.float32)
        return generated, picks

    for epoch in range(epochCount):
        DIST = "triangle"
        if DIST=="uniform":
            data = np.sort(np.random.uniform(low=-1, high=+1, size=(n,)).astype(np.float32))
        elif DIST=="triangle":
            bi = np.random.uniform(low=0, high=1, size=(n,2)).astype(np.float32)
            data = np.max(bi, axis=-1)
        else:
            assert False, "unknown distribution"

        data.sort()
        generated, picks = generate(centers, n)

        if epoch%5==0:
            plt.hist(generate(centers, 100000)[0], 50, normed=0, facecolor='green')
            plt.savefig("emd"+str(epoch)+".pdf")
            plt.close()
            plt.scatter(centers[:-1], centers[1:]-centers[:-1])
            plt.savefig("delta"+str(epoch)+".pdf")
            plt.close()

        sortedPairs = zip(generated, picks)
        sortedPairs.sort()
        triplets = zip(sortedPairs, data)
        # both are sorted at this point, this pairing is the earth mover's pairing.
        totalLoss = 0.0
        for (g,p), d in triplets:
            # linear derivative, corresponds to L2squared.
            # math.copysign(1, d-g) would be the derivative of L1=L2unsquared
            differential = d-g
            totalLoss += abs(differential) # NOT L2 squared, proper L2!
            centers[p] += differential * learningRate
        centers.sort()
        print "loss", totalLoss

        sys.stdout.flush()

if __name__ == "__main__":
    toyLearner()
