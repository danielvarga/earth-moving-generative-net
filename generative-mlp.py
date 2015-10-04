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
    l_in = lasagne.layers.InputLayer(shape=(None, inDim),
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
    return np.random.normal(loc=0.0, scale=1.0, size=(n, inDim))

def sample(net, n, inDim, input_var):
    initial = sampleInitial(n, inDim)
    output = lasagne.layers.get_output(net)
    net_fn = theano.function([input_var], output)
    return initial, net_fn(initial)

def initialNet():
    return input_var, net

def plot(input_var, net, name):
    n = 1000
    inDim = 2
    initial, sampled = sample(net, n, inDim, input_var)
    assert sampled.shape == (n,2)
    plt.scatter(sampled.T[0], sampled.T[1])
    plt.savefig(name+".pdf")

def update(input_var, net, initial, sampled, data):
    n = len(data)
    output = lasagne.layers.get_output(net)
    data_var = T.matrix('targets')
    loss = lasagne.objectives.squared_error(output, data_var).mean()
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.05, momentum=0.9)
    train_fn = theano.function([input_var, data_var], updates=updates)
    train_fn(initial, data)

def sampleAndUpdate(input_var, net, inDim, n):
    # TODO Refactor, I can't even change the goddamn target distribution in this source file!
    data = kohonen.samplesFromTarget(n)
    initial, sampled = sample(net, n, inDim, input_var)
    permutation = kohonen.optimalPairing(sampled, data)
    initial = initial[permutation]
    sampled = sampled[permutation]
    update(input_var, net, initial, sampled, data)

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
    inDim = 2
    outDim = 2
    hidden = 5
    input_var = T.matrix('inputs')
    net = buildNet(input_var, inDim, hidden, outDim)
    minibatchSize = 20
    sampleAndUpdate(input_var, net, inDim, n=minibatchSize)

if __name__ == "__main__":
    main()
