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
from nnbase.attrdict import AttrDict

# These are only included to make the unpickling of the autoencoder possible:
from nnbase.layers import Unpool2DLayer
from nnbase.utils import FlipBatchIterator



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
    assert layerNum in (2,3,4)

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
    elif layerNum==3:
        l_hid2 = lasagne.layers.DenseLayer(
            l_hid, num_units=hidden,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
        l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=outDim,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
    elif layerNum==4:
        l_hid2 = lasagne.layers.DenseLayer(
            l_hid, num_units=hidden,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
        l_hid3 = lasagne.layers.DenseLayer(
            l_hid2, num_units=hidden,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
        l_out = lasagne.layers.DenseLayer(
            l_hid3, num_units=outDim,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
    return l_out

def sampleInitial(n, inDim):
    return np.random.normal(loc=0.0, scale=1.0/4, size=(n, inDim)).astype(np.float32)
    # The old cubemixture model is now unreachable:
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

def constructTrainFunction(input_var, net, learningRate, momentum):
    output = lasagne.layers.get_output(net)
    data_var = T.matrix('targets')
    loss = lasagne.objectives.squared_error(output, data_var).mean()
    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learningRate, momentum=momentum)
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
        assert distances.shape == (len(data), len(sampled)) # Beware the transpose!
        findGenForData = True
        if findGenForData:
            # Counterintuitively, this seems to be better. Understand, verify.
            bestIndices = np.argmin(distances, axis=1)
            initial = initial[bestIndices]
            sampled = sampled[bestIndices]
            bestDists = distances.min(axis=1)
        else:
            bestIndices = np.argmin(distances, axis=0)
            data = data[bestIndices]
            # TODO bestDists currently unimplemented here

    # The idea is that big n is good because matches are close,
    # but big n is also bad because large minibatch sizes are generally bad.
    # We throw away data to combine the advantages of big n with small minibatch size.
    # Don't forget that this means that in an epoch we only see 1/overSamplingFactor
    # fraction of the dataset. There must be a less heavy-handed way.
    # TODO This will be retired probably. Until then, don't confuse it with params.oversampling.
    overSamplingFactor = 1
    subSample = np.random.choice(len(data), len(data)/overSamplingFactor)
    initial = initial[subSample]
    sampled = sampled[subSample]
    data = data[subSample]

    # That's where the update happens.
    train_fn(initial, data)

    doPlot = False
    if doPlot:
        # This goes to a different dir as the function does not access params.expName.
        nnbase.vis.plotGradients(data, sampled, initial, net_fn, "grad")

    # These values are a byproduct of the training step,
    # so they are from _before_ the training, not after it.
    return bestDists


def train(data, validation, params, logger=None):
    if logger is None:
        logger = sys.stdout
    expName = params.expName

    nnbase.vis.plotImages(data[:params.gridSizeForSampling**2], params.gridSizeForSampling, expName+"/input")

    # My network works with 1D input.
    data = nnbase.inputs.flattenImages(data)
    validation = nnbase.inputs.flattenImages(validation)

    height, width = params.height, params.width

    m = int(params.oversampling*params.minibatchSize)

    outDim = height*width
    input_var = T.matrix('inputs')
    net = buildNet(input_var, params.layerNum, params.inDim, params.hiddenLayerSize, outDim, useReLU=params.useReLU)

    minibatchCount = len(data)/params.minibatchSize

    train_fn = constructTrainFunction(input_var, net, params.learningRate, params.momentum)
    net_fn = constructSamplerFunction(input_var, net)

    # The reason for the +1 is that this way, if
    # epochCount is a multiple of plotEach, then the
    # last thing that happens is an evaluation.
    for epoch in range(params.epochCount+1):
        shuffledData = np.random.permutation(data)
        epochDistances = []
        for i in range(minibatchCount):
            dataBatch = shuffledData[i*params.minibatchSize:(i+1)*params.minibatchSize]
            minibatchDistances = sampleAndUpdate(train_fn, net_fn, params.inDim, n=params.minibatchSize, data=dataBatch, m=m)
            epochDistances.append(minibatchDistances)
        epochDistances = np.array(epochDistances)
        epochInterimMean = epochDistances.mean()
        epochInterimMedian = np.median(epochDistances)
        print >> logger, "epoch %d epochInterimMean %f epochInterimMedian %f" % (epoch, epochInterimMean, epochInterimMedian)
        logger.flush()

        # Remove the "epoch != 0" if you are trying to catch evaluation crashes.
        if epoch % params.plotEach == 0 and epoch != 0:
            # TODO This is mixing the responsibilities of evaluation and visualization:
            # TODO train_distance and validation_distance are calculated on only visImageCount images.
            doValidation = True
            if doValidation:
                start_time = time.time()
                visImageCount = params.gridSizeForSampling ** 2
                visualizedValidation = validation[:visImageCount]
                visualizedData = data[:visImageCount]
                trainMean, trainMedian = evaluate.fitAndVis(visualizedData,
                                              net_fn, sampleSource, params.inDim,
                                              height, width, params.gridSizeForSampling, name=expName+"/diff_train"+str(epoch))
                validationMean, validationMedian = evaluate.fitAndVis(visualizedValidation,
                                              net_fn, sampleSource, params.inDim,
                                              height, width, params.gridSizeForSampling, name=expName+"/diff_validation"+str(epoch))
                print >> logger, "epoch %d trainMean %f trainMedian %f validationMean %f validationMedian %f" % (
                    epoch, trainMean, trainMedian, validationMean, validationMedian)
                print >> logger, "time elapsed %f" % (time.time() - start_time)
                logger.flush()

            nnbase.vis.plotSampledImages(net_fn, params.inDim, expName+"/xy"+str(epoch),
                height, width, fromGrid=True, gridSize=params.gridSizeForInterpolation, plane=(0,1))
            nnbase.vis.plotSampledImages(net_fn, params.inDim, expName+"/yz"+str(epoch),
                height, width, fromGrid=True, gridSize=params.gridSizeForInterpolation, plane=(1,2))
            nnbase.vis.plotSampledImages(net_fn, params.inDim, expName+"/xz"+str(epoch),
                height, width, fromGrid=True, gridSize=params.gridSizeForInterpolation, plane=(0,2))
            nnbase.vis.plotSampledImages(net_fn, params.inDim, expName+"/s"+str(epoch),
                height, width, fromGrid=False, gridSize=params.gridSizeForSampling, sampleSourceFunction=sampleSource)

            with open(expName+"/som-generator.pkl", 'w') as f:
                cPickle.dump(net, f)

    return validationMean # The last calculated one, we don't recalculate.


def setupAndRun(params):
    data, validation = nnbase.inputs.readData(params)
    # We dump after readData() because it augments params
    # with width/height deduced from the input data.
    nnbase.inputs.dumpParams(params, file(params.expName+"/conf.txt", "w"))


    with file(params.expName+"/log.txt", "w") as logger:
        meanDist, medianDist = evaluate.fitAndVisNNBaselineMain(data, validation, params)
        print >> logger, "nnbaselineMean %f nnbaselineMedian %f" % (meanDist, medianDist)

        value = train(data, validation, params, logger)
        print >> logger, "final performance %f" % value

    return value

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


def setDefaultParams():
    params = AttrDict()
    params.inputType = "mnist"

    if params.inputType=="image":
        params.imageDirectory = "../face/SCUT-FBP/thumb.big/"
        params.gridSizeForSampling = 10
        params.gridSizeForInterpolation = 20
        params.plotEach = 1000
    elif params.inputType=="mnist":
        params.inputDigit = None
        params.everyNthInput = 10
        params.gridSizeForSampling = 20
        params.gridSizeForInterpolation = 30
        params.plotEach = 100 # That's too small for params.inputDigit = None, params.everyNthInput = 1
    else:
        assert False, "unknown inputType"

    params.inDim = 4
    params.minibatchSize = 100
    # m = oversampling*minibatchSize, that's how many
    # generated samples do we pair with our minibatchSize gold samples.
    params.oversampling = 1.0
    params.hiddenLayerSize = 100
    params.layerNum = 2
    params.useReLU = True
    params.learningRate = 0.2
    params.momentum = 0.5
    params.epochCount = 4800
    params.plotEach = 800
    return params


SHORTENED_PARAM_NAMES = { "learningRate":"lr", "minibatchSize":"n",
                          "momentum":"mom", "hiddenLayerSize":"hls",
                          "oversampling":"os"}

def spearmintDirName(spearmintParams):
    pairs = []
    for k in sorted(spearmintParams.keys()):
        v = spearmintParams[k]
        assert len(v)==1
        v = v[0]
        if k in SHORTENED_PARAM_NAMES:
            k = SHORTENED_PARAM_NAMES[k]
        # TODO if v is a float, normalize it. (0.2000001 and 0.199999 to 0.2)
        pairs.append((k, str(v)))
    pairs.sort()
    return "-".join(map(lambda (k,v): k+v, pairs))

def spearmintEntry(spearmintParams):
    params = setDefaultParams()
    for k,v in spearmintParams.iteritems():
        # v[0] because we only work with single values, and those are 1-element ndarrays in spearmint
        assert len(v)==1
        # We want int32 and float32, not the 64bit versions provided by spearmint.
        # http://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998#11389998
	params[k] = np.asscalar(v[0])
    params.expName = "spearmintOutput/" + spearmintDirName(spearmintParams)

    try:
        os.mkdir(params.expName)
    except OSError:
        logg("Warning: target directory already exists, or can't be created.")

    # If we are interested in consistent behavior across several datasets,
    # we can simply aggregate here: value = setupAndRun(params1) + setupAndRun(params2)
    # where params1 and params2 are the same except for imageDirectory or inputDigit or whatever (and expName).
    value = setupAndRun(params)
    return value

def main():
    assert len(sys.argv)==2
    confFilename = sys.argv[1]
    params = nnbase.inputs.paramsFromConf(file(confFilename))
    logg("Starting experiment, working directory: "+params.expName)

    try:
        os.mkdir(params.expName)
    except OSError:
        logg("Warning: target directory already exists, or can't be created.")

    value = setupAndRun(params)
    logg("final performance %f" % value)

    # TODO This codepath is temporarily abandoned:
    # mainLowDim(params.expName, params.minibatchSize)

if __name__ == "__main__":
    # import cProfile
    # cProfile.run("main()", "pstats")
    main()
