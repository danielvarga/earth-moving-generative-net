# This piece of code was prepared because I asked about its
# performance on the theano-users list:
# https://groups.google.com/forum/#!topic/theano-users/E7ProqnGUMk
# https://gist.github.com/danielvarga/d0eeacea92e65b19188c


import numpy as np
import theano
import theano.tensor as T


def randomMatrix(n, f):
    return np.random.normal(size=n*f).astype(np.float32).reshape((n, f))

n = 5000 # number of candidates
m = 1000 # number of targets
f = 500  # number of features

x = T.matrix('x') # candidates
y = T.matrix('y') # targets

xL2S = T.sum(x*x, axis=-1) # [n]
yL2S = T.sum(y*y, axis=-1) # [m]
xL2SM = T.zeros((m, n)) + xL2S # broadcasting, [m, n]
yL2SM = T.zeros((n, m)) + yL2S # # broadcasting, [n, m]
squaredPairwiseDistances = xL2SM.T + yL2SM - 2.0*T.dot(x, y.T) # [n, m]
bestIndices = T.argmin(squaredPairwiseDistances, axis=0)

nearests_fn = theano.function([x, y], bestIndices, profile=True)

print nearests_fn(randomMatrix(n, f), randomMatrix(m, f)).shape
