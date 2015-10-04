import cPickle
import gzip
import random
import math

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import lasagne

import kohonen

def buildNet(input_var, inDim, hidden, outDim):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, inDim),
                                     input_var=input_var)
    l_hid = lasagne.layers.DenseLayer(
            l_in, num_units=hidden,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())
    l_out = lasagne.layers.DenseLayer(
            l_hid, num_units=outDim,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())
    return l_out

def sampleInitial(n, inDim):
    return np.random.normal(loc=0.0, scale=1.0, size=(n, 1, 1, inDim))

def sample(net, n, inDim, input_var):
    initial = sampleInitial(n, inDim)
    output = lasagne.layers.get_output(net)
    net_fn = theano.function([input_var], output)
    return initial, net_fn(initial)
    # params = lasagne.layers.get_all_params(network, trainable=True)

def learn(data):
    inDim = 2
    outDim = 2
    hidden = 5
    assert len(data[0])==outDim
    input_var = T.tensor4('inputs')
    net = buildNet(input_var, inDim, hidden, outDim)
    n = 100
    initial, sampled = sample(net, n, inDim, input_var)
    plt.scatter(sampled.T[0], sampled.T[1])
    plt.savefig("gen.pdf")

def mnist():
    np.random.seed(1)
    datasetFile = "../rbm/data/mnist.pkl.gz"
    f = gzip.open(datasetFile, 'rb')
    datasets = cPickle.load(f)
    train_set, valid_set, test_set = datasets
    f.close()
    digit = 5
    input, output = train_set
    input = input[output==digit]
    output = output[output==digit]
    return input


def main():
    n = 1000
    # TODO Refactor, I can't even change the goddamn target distribution in this source file!
    data = kohonen.samplesFromTarget(n)
    learn(data)

if __name__ == "__main__":
    main()
