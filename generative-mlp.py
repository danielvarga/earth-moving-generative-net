import cPickle
import gzip
import sys
import os
import time
import random
import math

import numpy as np

import theano
import theano.tensor as T
import lasagne

import kohonen
import evaluate

import nnbase.inputs
import nnbase.vis

def logg(*ss):
    s = " ".join(map(str,ss))
    sys.stderr.write(s+"\n")


def buildNet(input_var, layerNum, inDim, hidden, outDim, useReLU):
    if useReLU:
        nonlinearity = lasagne.nonlinearities.rectify
        gain = 'relu'
    else:
        nonlinearity = lasagne.nonlinearities.tanh
        gain = 1.0
    assert layerNum in (2,3)

    l_in = lasagne.layers.InputLayer(shape=(None, inDim),
                                     input_var=input_var)
    l_hid = lasagne.layers.DenseLayer(
            l_in, num_units=hidden,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
    if layerNum==2:
        l_out = lasagne.layers.DenseLayer(
            l_hid, num_units=outDim,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
    else:
        l_hid2 = lasagne.layers.DenseLayer(
            l_hid, num_units=hidden,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
        l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=outDim,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
    return l_out

def sampleInitial(n, inDim):
    discrete = np.random.randint(0, 2, (n, inDim))
    continuous = np.random.normal(loc=0.0, scale=1.0/4, size=(n, inDim))
    return discrete + continuous

def sampleSource(net_fn, n, inDim):
    initial = sampleInitial(n, inDim)
    return initial, net_fn(initial)

def constructSamplerFunction(input_var, net):
    output = lasagne.layers.get_output(net)
    net_fn = theano.function([input_var], output)
    return net_fn

def constructTrainFunction(input_var, net):
    output = lasagne.layers.get_output(net)
    data_var = T.matrix('targets')
    loss = lasagne.objectives.squared_error(output, data_var).mean()
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.02, momentum=0.5)
    train_fn = theano.function([input_var, data_var], updates=updates)
    return train_fn

def sampleAndUpdate(train_fn, net_fn, inDim, n, data=None, m=None):
    if data is None:
        data = kohonen.samplesFromTarget(n) # TODO Refactor, I can't even change the goddamn target distribution in this source file!
    else:
        assert len(data)==n
    if m is None:
        m = n

    initial, sampled = sampleSource(net_fn, m, inDim)
    bipartiteMatchingBased = False
    if bipartiteMatchingBased:
        permutation = kohonen.optimalPairing(sampled, data)
        # If fudge>1 then this is not in fact a permutation, but it's still meaningful.
        # Never played with it, though.
        initial = initial[permutation]
        sampled = sampled[permutation]
    else:
        distances = kohonen.distanceMatrix(sampled, data)
        findGenForData = True
        if findGenForData:
            # Counterintuitively, this seems to be better. Understand, verify.
            bestDists = np.argmin(distances, axis=1)
            initial = initial[bestDists]
            sampled = sampled[bestDists]
            print distances.min(axis=1).sum()
        else:
            bestDists = np.argmin(distances, axis=0)
            data = data[bestDists]

    # The idea is that big n is good because matches are close,
    # but big n is also bad because large minibatch sizes are generally bad.
    # We throw away data to combine the advantages of big n with small minibatch size.
    # Don't forget that this means that in an epoch we only see 1/overSamplingFactor
    # fraction of the dataset. There must be a less heavy-handed way.
    overSamplingFactor = 1
    subSample = np.random.choice(len(data), len(data)/overSamplingFactor)
    initial = initial[subSample]
    sampled = sampled[subSample]
    data = data[subSample]

    # That's where the update happens.
    train_fn(initial, data)

    updated = net_fn(initial)
    doPlot = False
    if doPlot:
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        for x, y, z in zip(data, sampled, updated):
            # print np.linalg.norm(x-y)-np.linalg.norm(x-z), x, y, z
            plt.arrow(y[0], y[1], (x-y)[0], (x-y)[1], color=(1,0,0),  head_width=0.05, head_length=0.1)
            plt.arrow(y[0], y[1], (z-y)[0], (z-y)[1], color=(0,0,1), head_width=0.05, head_length=0.1)
        plt.savefig("grad.pdf")

def mainMNIST(expName, minibatchSize):
    face = False
    if face:
        directory = "../face/SCUT-FBP/thumb.big/"
        data, (height, width) = nnbase.inputs.faces(directory)
        gridSizeForSampling = 10
        gridSizeForInterpolation = 20
        plotEach = 1000
    else:
        digit = None # None if we want all of them.
        data, (height, width) = nnbase.inputs.mnist(digit) # Don't just rewrite it here, validation!

        gridSizeForSampling = 20
        gridSizeForInterpolation = 30
        plotEach = 1000

        validation, (_, _) = nnbase.inputs.mnist(digit)

    nnbase.vis.plotImages(data[:gridSizeForSampling**2], gridSizeForSampling, expName+"/input")

    # My network works with 1D input.
    data = nnbase.inputs.flattenImages(data)
    validation = nnbase.inputs.flattenImages(validation)

    inDim = 20
    outDim = height*width
    hidden = 100
    layerNum = 2
    input_var = T.matrix('inputs')
    net = buildNet(input_var, layerNum, inDim, hidden, outDim, useReLU=False)

    minibatchCount = len(data)/minibatchSize
    epochCount = 500000

    train_fn = constructTrainFunction(input_var, net)
    net_fn = constructSamplerFunction(input_var, net)

    for epoch in range(epochCount):
        print "epoch", epoch
        data = np.random.permutation(data)
        for i in range(minibatchCount):
            print i,
            sys.stdout.flush()
            dataBatch = data[i*minibatchSize:(i+1)*minibatchSize]
            sampleAndUpdate(train_fn, net_fn, inDim, n=minibatchSize, data=dataBatch)
        print

        # initial, oneSample = sampleSource(net, 1, inDim, input_var)
        # print oneSample.reshape((width,height))

        if epoch%plotEach==0:
            start_time = time.time()
            train_distance = evaluate.fitAndVis(data[:gridSizeForSampling*gridSizeForSampling],
                                          net_fn, sampleSource, inDim,
                                          height, width, gridSizeForSampling, name=expName+"/diff_train"+str(epoch))
            validation_distance = evaluate.fitAndVis(validation[:gridSizeForSampling*gridSizeForSampling],
                                          net_fn, sampleSource, inDim,
                                          height, width, gridSizeForSampling, name=expName+"/diff_validation"+str(epoch))
            print "epoch %d train_distance %f validation_distance %f" % (epoch, train_distance, validation_distance)
            print "time elapsed %f" % (time.time() - start_time)
            sys.stdout.flush()

            nnbase.vis.plotSampledImages(net_fn, inDim, expName+"/xy"+str(epoch),
                height, width, fromGrid=True, gridSize=gridSizeForInterpolation, plane=(0,1))
            nnbase.vis.plotSampledImages(net_fn, inDim, expName+"/yz"+str(epoch),
                height, width, fromGrid=True, gridSize=gridSizeForInterpolation, plane=(1,2))
            nnbase.vis.plotSampledImages(net_fn, inDim, expName+"/xz"+str(epoch),
                height, width, fromGrid=True, gridSize=gridSizeForInterpolation, plane=(0,2))
            nnbase.vis.plotSampledImages(net_fn, inDim, expName+"/s"+str(epoch),
                height, width, fromGrid=False, gridSize=gridSizeForSampling, sampleSourceFunction=sampleSource)

            with open(expName+"/som-generator.pkl", 'w') as f:
                cPickle.dump(net, f)

def sampleAndPlot(net_fn, inDim, n, name):
    initial, sampled = sampleSource(net_fn, n, inDim)
    nnbase.vis.plot(sampled, name)

def mainLowDim(expName, minibatchSize):
    inDim = 2
    outDim = 2
    layerNum = 3
    hidden = 100
    input_var = T.matrix('inputs')
    net = buildNet(input_var, layerNum, inDim, hidden, outDim, useReLU=False)
    train_fn = constructTrainFunction(input_var, net)
    net_fn = constructSamplerFunction(input_var, net)
    for i in range(100):
        print i,
        sys.stdout.flush()
        sampleAndUpdate(train_fn, net_fn, inDim, n=minibatchSize)
        sampleAndPlot(net_fn, inDim, 1000, expName+"/d"+str(i))
    print

def main():
    expName = sys.argv[1]
    minibatchSize = int(sys.argv[2])
    try:
        expName = expName.rstrip("/")
        os.mkdir(expName)
    except OSError:
        logg("Warning: target directory already exists, or can't be created.")
    doMNIST = True
    if doMNIST:
        mainMNIST(expName, minibatchSize)
    else:
        mainLowDim(expName, minibatchSize)

if __name__ == "__main__":
    main()
