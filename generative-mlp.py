import cPickle
import gzip
import sys
import os
import random
import math

import numpy as np

import matplotlib.pyplot as plt
import PIL.Image as Image

import theano
import theano.tensor as T
import lasagne

import kohonen


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

def plot(net_fn, inDim, name):
    n = 1000
    initial, sampled = sampleSource(net_fn, n, inDim)
    assert sampled.shape[0]==n
    # If feature dim >> 2, and PCA has not happened, it's not too clever to plot the first two dims.
    plt.scatter(sampled.T[0], sampled.T[1])
    plt.savefig(name+".pdf")
    plt.close()

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

def mnist(digit=None, torusHack=False):
    np.random.seed(1)
    datasetFile = "../rbm/data/mnist.pkl.gz"
    f = gzip.open(datasetFile, 'rb')
    datasets = cPickle.load(f)
    train_set, valid_set, test_set = datasets
    f.close()
    input, output = train_set
    if digit is not None:
        input = input[output==digit]
    if torusHack:
        # This is a SINGLE sample, translated and multiplied.
        sample = input[0].reshape((28, 28))
        inputRows = []
        for dx in range(28):
            for dy in range(28):
                s = sample.copy()
                s = np.hstack((s[:, dy:], s[:, :dy]))
                s = np.vstack((s[dx:, :], s[:dx, :]))
                inputRows.append(s.reshape(28*28))
        input = np.array(inputRows)
        input = np.vstack([[input]*10])
    np.random.permutation(input)
    return input

def plotDigits(net_fn, inDim, name, fromGrid, gridSize, plane=None):
    if fromGrid:
        if plane is None:
            plane = (0, 1)
        n_x = gridSize
        n_y = gridSize
        n = n_x*n_y
        initial = []
        for x in np.linspace(-2, +2, n_x):
            for y in np.linspace(-2, +2, n_y):
                v = np.zeros(inDim)
                v[plane[0]] = x
                v[plane[1]] = y
                initial.append(v)
        data = net_fn(initial)
    else:
        assert plane is None, "unsupported"
        n_x = gridSize
        n_y = gridSize
        n = n_x*n_y
        initial, data = sampleSource(net_fn, n, inDim)

    image_data = np.zeros(
        (29 * n_y + 1, 29 * n_x - 1),
        dtype='uint8'
    )
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = data[idx].reshape((28,28))
        image_data[29*x:29*x+28, 29*y:29*y+28] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data)
    img.save(name+".png")

def faces():
    imgs = []
    directory = "../face/SCUT-FBP/thumb/"
    for f in os.listdir(directory):
        if f.endswith(".jpg"):
            img = Image.open(directory+f).convert("L")
            arr = np.array(img).flatten()
            assert len(arr)==28*28, "Bad size %s %d" % (f, len(arr))
            imgs.append(arr)
    return np.array(imgs).astype(float) / 255

def mainMNIST(expName, minibatchSize):
    face = True
    if face:
        data = faces()
    else:
        data = mnist()

    inDim = 7
    outDim = 28*28
    hidden = 100
    layerNum = 2
    input_var = T.matrix('inputs')
    net = buildNet(input_var, layerNum, inDim, hidden, outDim, useReLU=False)

    minibatchCount = len(data)/minibatchSize
    epochCount = 500000

    train_fn = constructTrainFunction(input_var, net)
    net_fn = constructSamplerFunction(input_var, net)

    plotEach = 1000

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
        # print oneSample.reshape((28,28))

        if epoch%plotEach==0:
            plotDigits(net_fn, inDim, expName+"/xy"+str(epoch), fromGrid=True, gridSize=50, plane=(0,1))
            plotDigits(net_fn, inDim, expName+"/yz"+str(epoch), fromGrid=True, gridSize=50, plane=(1,2))
            plotDigits(net_fn, inDim, expName+"/xz"+str(epoch), fromGrid=True, gridSize=50, plane=(0,2))
            plotDigits(net_fn, inDim, expName+"/s"+str(epoch), fromGrid=False, gridSize=20)

            with open(expName+"/som-generator.pkl", 'w') as f:
                cPickle.dump(net, f)

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
        plot(input_var, net, inDim, expName+"/d"+str(i))
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
