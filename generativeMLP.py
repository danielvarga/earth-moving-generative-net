import cPickle
import gzip
import sys
import os
import time
import random
import math
from operator import itemgetter

import numpy as np

import theano
import theano.tensor as T
import lasagne

import kohonen # TODO This should only be used on the abandoned bipartiteMatchingBased==True codepath.
import evaluate
import distances

import nnbase.inputs
import nnbase.vis
from nnbase.attrdict import AttrDict

# These are only included to make the unpickling of the autoencoder possible:
from nnbase.layers import Unpool2DLayer
from nnbase.shape import ReshapeLayer
from nnbase.utils import FlipBatchIterator

L1_LOSS = "l1"
L2_SQUARED_LOSS = "l2squared"
# The weird name is because I really don't want to accidentally use this instead of L2_SQUARED_LOSS:
L2_UNSQUARED_LOSS = "l2unsquared"


def logg(*ss):
    s = " ".join(map(str,ss))
    sys.stderr.write(s+"\n")


def buildConvNet(input_var, layerNum, inDim, hidden, outDim, useReLU, leakiness=0.0):
    # ('hidden', layers.DenseLayer),
    # ('unflatten', ReshapeLayer),
    # ('unpool', Unpool2DLayer),
    # ('deconv', layers.Conv2DLayer),
    # ('output_layer', ReshapeLayer),
    # TODO Copypasted, refactor.
    if useReLU:
        if leakiness==0.0:
            nonlinearity = lasagne.nonlinearities.rectify
            gain = 'relu'
        else:
            nonlinearity = lasagne.nonlinearities.LeakyRectify(leakiness)
            gain = math.sqrt(2/(1+leakiness**2))
    else:
        nonlinearity = lasagne.nonlinearities.tanh
        gain = 1.0

    filter_sizes = 7
    conv_filters = 32
    deconv_filters = 32
    width = 28 # TODO MNIST specific!
    height = 28

    l_in = lasagne.layers.InputLayer(shape=(None, inDim),
                                     input_var=input_var)
    l_hid = lasagne.layers.DenseLayer(
            l_in, num_units=hidden,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
    hid2_num_units= deconv_filters * (height + filter_sizes - 1) * (width + filter_sizes - 1) / 4
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid, num_units=hid2_num_units,
            nonlinearity=nonlinearity,
            W=lasagne.init.GlorotUniform(gain=gain))
    l_unflatten = ReshapeLayer(
            l_hid2, shape=(([0], deconv_filters, (height + filter_sizes - 1) / 2, (width + filter_sizes - 1) / 2 )))
    l_unpool = Unpool2DLayer(
            l_unflatten, ds=(2, 2))
    l_deconv = lasagne.layers.Conv2DLayer(
            l_unpool, num_filters=1, filter_size = (filter_sizes, filter_sizes),
            border_mode="valid", nonlinearity=None)
    l_output = ReshapeLayer(
            l_deconv, shape = (([0], -1)))
    return l_output

def buildNet(input_var, layerNum, inDim, hidden, outDim, useReLU, leakiness=0.0):
    if useReLU:
        if leakiness==0.0:
            nonlinearity = lasagne.nonlinearities.rectify
            gain = 'relu'
        else:
            nonlinearity = lasagne.nonlinearities.LeakyRectify(leakiness)
            gain = math.sqrt(2/(1+leakiness**2))
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

def sampleInitial(n, inDim, sd, inBoolDim):
    continuous = np.random.normal(loc=0.0, scale=sd, size=(n, inDim)).astype(np.float32)
    discrete = np.random.randint(0, 2, (n, inBoolDim))
    continuous[:, :inBoolDim] += discrete
    return continuous

def sampleSourceParametrized(net_fn, n, inDim, sd, inBoolDim):
    initial = sampleInitial(n, inDim, sd, inBoolDim)
    return initial, net_fn(initial)

def constructSamplerFunction(input_var, net):
    output = lasagne.layers.get_output(net)
    net_fn = theano.function([input_var], output)
    return net_fn

def constructTrainFunction(input_var, net, learningRate, momentum, regularization, lossType=L2_SQUARED_LOSS):
    output = lasagne.layers.get_output(net)
    data_var = T.matrix('targets')
    if lossType==L1_LOSS:
        loss = T.abs_(output-data_var).mean()
    elif lossType==L2_SQUARED_LOSS:
        loss = lasagne.objectives.squared_error(output, data_var).mean()
    elif lossType==L2_UNSQUARED_LOSS:
        lossSqr = ((output-data_var)**2).sum(axis=1)
        loss = T.sqrt(lossSqr+1e-6).mean() # Fudge constant to avoid numerical stability issues.
    else:
        assert False, "unknown similarity loss function: %s" % lossType

    if regularization!=0.0:
        logg('regularization', regularization)
        loss += lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2) * regularization

    params = lasagne.layers.get_all_params(net, trainable=True)

    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learningRate, momentum=momentum)
    # The rmsprop update rule is tricky. Properties (as measured on conf8):
    # - Converges twice as fast at the beginning.
    # - Goes way below nesterov on trainMean.
    # - ...which implies that s*.png is visually better, but that's just overfitting, because it
    #   - reaches approx. the same performance as nesterov on validationMean,
    #   - and visually it does not improve on diff_validation after convergence on validationMean.
    # - Performance has a hockey-stick dependence on epsilon:
    #   Smaller epsilon is better until 0.0001, and then at 0.00001 it explodes.
    # updates = lasagne.updates.rmsprop(loss, params, epsilon=0.0001)

    train_fn = theano.function([input_var, data_var], updates=updates)
    return train_fn

def sampleAndUpdate(train_fn, net_fn, closestFnFactory, inDim, sampleSource, n, data=None, m=None, innerGradientStepCount=1):
    if data is None:
        data = kohonen.samplesFromTarget(n) # TODO Refactor, I can't even change the goddamn target distribution in this source file!
    else:
        assert len(data)==n
    if m is None:
        m = n

    initial, sampled = sampleSource(net_fn, m, inDim)

    doDetailed1DVis = True and (data.shape[1]==1)

    bipartiteMatchingBased = False
    if bipartiteMatchingBased:
        if data.shape[1]==1:
            # In 1d we can actually solve the weighted bipartite matching
            # problem, by sorting. Basically that's what Magdon-Ismail and Atiya do.
            assert len(data)==len(initial)
            data.sort(axis=0)
            pairs = sorted(zip(sampled, initial))
            sampled = np.array(map(itemgetter(0), pairs))
            initial = np.array(map(itemgetter(1), pairs))
        else:
            # Pretty much obsoleted, because it can't be made fast.
            # Does a full weighted bipartite matching.
            # Left here for emotional reasons.
            permutation = kohonen.optimalPairing(sampled, data)
            initial = initial[permutation]
            sampled = sampled[permutation]
    else:
        # TODO We had this cool findGenForData=False experiment here
        # TODO that didn't go anywhere at first, but we shouldn't let it go this easily.
        findGenForData = True
        if findGenForData:
            bestIndices = closestFnFactory(sampled, data)
            initial = initial[bestIndices]
            sampled = sampled[bestIndices]
        else:
            bestIndices = closestFnFactory(data, sampled)
            data = data[bestIndices]

    bestDists = np.linalg.norm(data-sampled, axis=1)

    for i in range(innerGradientStepCount):
        # That's where the update happens.
        train_fn(initial, data)

    if doDetailed1DVis and random.randrange(100)==0:
        postSampled = net_fn(initial)
        nnbase.vis.gradientMap1D(data, sampled, postSampled, "gradient")

    # These values are a byproduct of the training step,
    # so they are from _before_ the training, not after it.
    return bestDists


def lowDimFitAndVis(data, validation, epoch, net, net_fn, closestFnFactory, sampleSource, params, logger):
    n, dim = data.shape
    inDim = params.inDim
    initial, sampled = sampleSource(net_fn, n, inDim)
    nnbase.vis.heatmap(sampled, params.expName+"/heatmap"+str(epoch))


def highDimFitAndVis(data, validation, epoch, net, net_fn, closestFnFactory, sampleSource, params, logger):
    height, width = params.height, params.width
    expName = params.expName

    # TODO This is mixing the responsibilities of evaluation and visualization:
    # TODO train_distance and validation_distance are calculated on only visImageCount images.
    doValidation = True
    if doValidation:
        start_time = time.time()
        visImageCount = params.gridSizeForSampling ** 2
        visualizedValidation = validation[:visImageCount]
        visualizedData = data[:visImageCount]
        trainMean, trainMedian = evaluate.fitAndVis(visualizedData,
                                      net_fn, closestFnFactory, sampleSource, params.inDim,
                                      height, width, params.gridSizeForSampling, name=expName+"/diff_train"+str(epoch))
        validationMean, validationMedian = evaluate.fitAndVis(visualizedValidation,
                                      net_fn, closestFnFactory, sampleSource, params.inDim,
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


def train(data, validation, params, logger=None):
    if logger is None:
        logger = sys.stdout

    isLowDim = "isLowDim" in params and params.isLowDim

    if isLowDim:
        nnbase.vis.heatmap(data, params.expName+"/input")
    else:
        # Have to do before flattening:
        nnbase.vis.plotImages(data[:params.gridSizeForSampling**2], params.gridSizeForSampling, params.expName+"/input")

    # My network works with 1D input.
    data = nnbase.inputs.flattenImages(data)
    validation = nnbase.inputs.flattenImages(validation)

    m = int(params.oversampling*params.minibatchSize)

    outDim = data.shape[1] # Flattening already happened.
    if "height" in params:
        assert params.height * params.width == outDim

    input_var = T.matrix('inputs')
    leakiness = 0.0 if 'reLULeakiness' not in params else params.reLULeakiness
    if not params.useReLU:
        assert leakiness==0.0, "reLULeakiness not allowed for tanh activation"
    if 'convolutional' in params and params.convolutional:
        net = buildConvNet(input_var, params.layerNum, params.inDim, params.hiddenLayerSize, outDim,
                   useReLU=params.useReLU, leakiness=leakiness)
    else:
        net = buildNet(input_var, params.layerNum, params.inDim, params.hiddenLayerSize, outDim,
                   useReLU=params.useReLU, leakiness=leakiness)

    minibatchCount = len(data)/params.minibatchSize

    regularization = 0.0 if 'regularization' not in params else params.regularization # L2

    innerGradientStepCount = 1 if 'innerGradientStepCount' not in params else params.innerGradientStepCount

    lossType = params.loss if "loss" in params else L2_SQUARED_LOSS

    learningRate_shared = theano.shared(np.array(params.learningRate, dtype=np.float32))

    # Per epoch, which means that this is super-sensitive to epoch size.
    learningRateDecay = np.float32(1.0 if 'learningRateDecay' not in params else params.learningRateDecay)

    train_fn = constructTrainFunction(input_var, net, learningRate_shared, params.momentum, regularization, lossType)
    net_fn = constructSamplerFunction(input_var, net)
    closestFnFactory = distances.ClosestFnFactory()

    sampleSource = lambda net_fn, n, inDim: sampleSourceParametrized(net_fn, n, inDim, params.initialSD, params.inBoolDim)

    validationMean = 1e10 # ad hoc inf-like value.

    # The reason for the +1 is that this way, if
    # epochCount is a multiple of plotEach, then the
    # last thing that happens is an evaluation.
    for epoch in range(params.epochCount+1):
        shuffledData = np.random.permutation(data)
        epochDistances = []
        for i in range(minibatchCount):
            dataBatch = shuffledData[i*params.minibatchSize:(i+1)*params.minibatchSize]

            # The issue with using a minibatchSize that's not a divisor of corpus size
            # is that m is calculated before the epoch loop. This is not trivial to fix,
            # because constructMinimalDistanceIndicesFunction gets n and m as args.
            assert params.minibatchSize==len(dataBatch)

            minibatchDistances = sampleAndUpdate(train_fn, net_fn, closestFnFactory, params.inDim, sampleSource,
                                                 n=params.minibatchSize, data=dataBatch, m=m,
                                                 innerGradientStepCount=innerGradientStepCount)
            epochDistances.append(minibatchDistances)
        epochDistances = np.array(epochDistances)
        epochInterimMean = epochDistances.mean()
        epochInterimMedian = np.median(epochDistances)

        # Remove the "epoch != 0" if you are trying to catch evaluation crashes.
        if epoch % params.plotEach == 0 and epoch != 0:
            print >> logger, "epoch %d epochInterimMean %f epochInterimMedian %f" % (epoch, epochInterimMean, epochInterimMedian)
            print >> logger, "learningRate", learningRate_shared.get_value()
            if isLowDim:
                lowDimFitAndVis(data, validation, epoch, net, net_fn, closestFnFactory, sampleSource, params, logger)
            else:
                highDimFitAndVis(data, validation, epoch, net, net_fn, closestFnFactory, sampleSource, params, logger)


        learningRate_shared.set_value( learningRateDecay * learningRate_shared.get_value() )

    return validationMean # The last calculated one, we don't recalculate.


def setupAndRun(params):
    data, validation = nnbase.inputs.readData(params)
    # We dump after readData() because it augments params
    # with width/height deduced from the input data.
    nnbase.inputs.dumpParams(params, file(params.expName+"/conf.txt", "w"))

    isLowDim = "isLowDim" in params and params.isLowDim

    with file(params.expName+"/log.txt", "w") as logger:
        if not isLowDim:
            meanDist, medianDist = evaluate.fitAndVisNNBaselineMain(data, validation, params)
            print >> logger, "nnbaselineMean %f nnbaselineMedian %f" % (meanDist, medianDist)

        value = train(data, validation, params, logger)
        print >> logger, "final performance %f" % value

    return value

def sampleAndPlot(net_fn, inDim, initialSD, inBoolDim, n, name):
    initial, sampled = sampleSourceParametrized(net_fn, n, inDim, initialSD, inBoolDim)
    nnbase.vis.plot(sampled, name)

def mainLowDim(expName, minibatchSize, initialSD):
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
        sampleAndPlot(net_fn, inDim, initialSD, 1000, expName+"/d"+str(i))
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

    # values coming from adhoc/spearmint-best-leaky.txt

    params.inDim = 50
    params.inBoolDim = 0
    params.initialSD = 0.25
    params.minibatchSize = 1000
    # m = oversampling*minibatchSize, that's how many
    # generated samples do we pair with our minibatchSize gold samples.
    params.oversampling = 8.0
    params.hiddenLayerSize = 673 
    params.layerNum = 3
    params.useReLU = True
    params.reLULeakiness = 0.01
    params.learningRate = 1.0
    params.momentum = 0.969849416169
    # in experiment regularization_initialSD used 6400 here, but that's
    # not nice to Spearmint, as validation optimum is usually
    # at 4800, and I don't have early stopping implemented.
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
    # np.float32 to float:
    value = np.asscalar(value)
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
    doCPUProfile = False
    if doCPUProfile:
        import cProfile
        cProfile.run("main()", "pstats")
    else:
        main()
