import numpy as np
import os
import gzip
import cPickle

# When I move out the synthetic distributions, these imports should move as well.
import math
import random
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

from nnbase.attrdict import AttrDict
import autoencoder


def mnist(digit=None, torusHack=False, autoencoded=False, which='train', everyNth=1):
    np.random.seed(1) # TODO Not the right place to do this.
    datasetFile = "mnist.pkl.gz"
    f = gzip.open(datasetFile, 'rb')
    datasets = cPickle.load(f)
    train_set, valid_set, test_set = datasets
    f.close()
    if which=='train':
        input, output = train_set
    elif which=='validation':
        input, output = valid_set
    elif which=='test':
        input, output = test_set
    else:
        assert which in ('train', 'validation', 'test')

    input = input.reshape((-1, 28, 28))
    if digit is not None:
        input = input[output==digit]
    if torusHack:
        # This is a SINGLE sample, translated and multiplied.
        sample = input[0]
        inputRows = []
        for dx in range(28):
            for dy in range(28):
                s = sample.copy()
                s = np.hstack((s[:, dy:], s[:, :dy]))
                s = np.vstack((s[dx:, :], s[:dx, :]))
                inputRows.append(s)
        input = np.array(inputRows)
        input = np.vstack([[input]*10])
    input = np.random.permutation(input)
    input = input[::everyNth]
    input = input.astype(np.float32)

    if autoencoded:
        autoencoderFile = "../lasagne-demo/conv_ae.pkl"
        mu = 0.13045
        sigma = 0.30729
        ae = autoencoder.Autoencoder(cPickle.load(open(autoencoderFile, 'r')), mu=mu, sigma=sigma)
        ae.split()
        encodedInput = ae.encode(input.reshape((-1, 1, 28, 28)))
        assert encodedInput.shape[1] == 40
        # encodedInput = encodedInput.reshape((-1, 8, 5))
        # print encodedInput.shape
        # return encodedInput, (8, 5)
        decodedInput = ae.decode(encodedInput)
        return decodedInput.reshape((-1, 28, 28)) , (28, 28)
    else:
        return input, (28, 28)

def flattenImages(input):
    shape = input.shape
    assert len(shape) in (2, 3)
    if len(shape)==2:
        return input
    l, height, width = shape
    return input.reshape((l, height*width))

def faces(directory):
    imgs = []
    height = None
    width = None
    for f in os.listdir(directory):
        if f.endswith(".jpg") or f.endswith(".png"):
            img = Image.open(os.path.join(directory, f)).convert("L")
            arr = np.array(img)
            if height is None:
                height, width = arr.shape
            else:
                assert (height, width) == arr.shape, "Bad size %s %s" % (f, str(arr.shape))
            imgs.append(arr)
    input = np.array(imgs).astype(float) / 255
    np.random.seed(1) # TODO Not the right place to do this.
    input = np.random.permutation(input)
    return input, (height, width)

def generateWave(n, height, width, waveCount):
    d = height*width
    phases = 2 * np.pi * np.random.uniform(size=n).astype(np.float32)
    rangeMat = np.zeros((n, d)).astype(np.float32) + np.linspace(start=0.0, stop=1.0, num=d).astype(np.float32) # broadcast, tiling rows
    phaseMat = np.zeros((n, d)).astype(np.float32) + phases[:, np.newaxis] # broadcast, tiling columns
    waves = (np.sin(rangeMat*(waveCount*2*np.pi) + phaseMat)+1.0)/2.0
    assert waves.dtype==np.float32
    assert np.sum(np.isnan(waves)) == 0
    return waves.reshape((n, height, width))

# Super ad hoc, but it shouldn't matter.
def generatePlane(n, height, width):
    normals = np.random.normal(size=(n,2)).astype(np.float32)
    zeros = np.zeros((n, height, width)).astype(np.float32)
    # two-dim broadcasts:
    heightMat = zeros + np.linspace(start=-1.0, stop=1.0, num=height).astype(np.float32)[:, np.newaxis]
    widthMat = zeros + np.linspace(start=-1.0, stop=1.0, num=width).astype(np.float32)[np.newaxis, :]
    planes = heightMat*normals[:, 0][:, np.newaxis, np.newaxis] + widthMat*normals[: ,1][:, np.newaxis, np.newaxis] + 0.5
    np.clip(planes, 0.0, 1.0, planes)
    return planes

def generateOneClock(width):
    data = np.zeros((width, width)).astype(np.float32)
    r = float(width/2)
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    theta = random.uniform(0, 2*math.pi)
    intensity = random.uniform(0.0, 1.0)
    p = ((r, r), (r*(1+math.cos(theta)), r*(1+math.sin(theta))), (r*(1+math.cos(theta+1)), r*(1+math.sin(theta+1))))
    draw.polygon(p, fill=intensity)
    return np.asarray(img)

def generateClock(n, height, width):
    assert height == width
    return np.array([ generateOneClock(width) for i in range(n) ])

def generateOneDot(width):
    data = np.zeros((width, width)).astype(np.float32)
    r = float(width/2)
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    theta = random.uniform(0, 2*math.pi)
    intensity = random.uniform(0.0, 1.0)
    p = ((r, r), (r*(1+math.cos(theta)), r*(1+math.sin(theta))), (r*(1+math.cos(theta+1)), r*(1+math.sin(theta+1))))
    draw.polygon(p, fill=intensity)
    return np.asarray(img)

def generateSine(n, height, width):
    waveLength = height*np.pi
    parameters = np.random.uniform(low=waveLength*1.2, high=waveLength*1.8, size=(n,2)).astype(np.float32)
    data = np.zeros((n, height*width)).astype(np.float32)
    data += np.linspace(start=0.0, stop=height*width-1, num=height*width).astype(np.float32)[np.newaxis, :]
    data /= parameters[:,1][:, np.newaxis]
    data += parameters[:,0][:, np.newaxis] * 10
    data = (np.sin(data)+1.0) / 2
    return data.reshape((n, height, width))

def generate1DUniform(n):
    return np.random.uniform(low=-1, high=+1, size=(n,1)).astype(np.float32)

def generate1DTriangle(n):
    bi = np.random.uniform(low=0, high=1, size=(n,2)).astype(np.float32)
    data = np.max(bi, axis=-1, keepdims=True)
    assert data.shape==(n,1)
    return data

def generate2DCircle(n):
    slacked = 2*n+100
    cartesian = np.random.rand(slacked, 2).astype(np.float32)
    cartesian *= 2
    cartesian -= 1
    cartesian = cartesian[np.sum(cartesian*cartesian, axis=1)<1, :]
    assert len(cartesian)>=n
    cartesian = cartesian[:n, :]

    # import matplotlib.pyplot as plt
    # plt.scatter(cartesian[:,0], cartesian[:,1])
    # plt.savefig("circle.pdf")
    # plt.close()

    return cartesian


def generate2DHalfcircle(n):
    slacked = 2*n+100
    cartesian = np.random.rand(slacked, 2).astype(np.float32)
    cartesian[:, 1] *= 2
    cartesian[:, 1] -= 1
    cartesian = cartesian[np.sum(cartesian*cartesian, axis=1)<1, :]
    assert len(cartesian)>=n
    cartesian = cartesian[:n, :]
    return cartesian


GENERATOR_FUNCTIONS = {"wave":  [generateWave, ["waveCount"]],
                       "plane": [generatePlane, []],
                       "clock": [generateClock, []],
                       "sine":  [generateSine, []],
                       "1d.uniform": [generate1DUniform, []],
                       "1d.triangle": [generate1DTriangle, []],
                       "2d.circle": [generate2DCircle, []],
                       "2d.halfcircle": [generate2DHalfcircle, []]
                       }

def readData(params):
    if params.inputType=="image":
        data, (height, width) = faces(params.imageDirectory)
        n = len(data)
        trainSize = 9*n/10
        validation = data[trainSize:]
        data = data[:trainSize]
    elif params.inputType=="mnist":
        autoencoded = params.get("autoencoded", False)
        data, (height, width) = mnist(params.inputDigit, which='train', everyNth=params.everyNthInput, autoencoded=autoencoded)
        validation, (_, _) = mnist(params.inputDigit, which='validation', autoencoded=autoencoded)
    elif params.inputType in GENERATOR_FUNCTIONS.keys():
        generatorFunction, argNames = GENERATOR_FUNCTIONS[params.inputType]
        arguments = { argName: params[argName] for argName in argNames }

        isLowDim = "isLowDim" in params and params.isLowDim
        if isLowDim:
            assert "height" not in params and "width" not in params, "For isLowDim==True, height and width params are meaningless."
            data = generatorFunction(params.trainSize, **arguments)
            validation = generatorFunction(params.validSize, **arguments)
        else:
            height, width = params.height, params.width
            data = generatorFunction(params.trainSize, height, width, **arguments)
            validation = generatorFunction(params.validSize, height, width, **arguments)
    else:
        assert False, "unknown params.inputType %s" % params.inputType
    if "height" in params or "width" in params:
        assert (params.height == height) and (params.width  == width), "%d!=%d or %d!=%d" % (params.height, height, params.width, width)
    return data, validation


def dumpParams(params, f):
    for k in sorted(params.keys()):
        print >>f, k+"\t"+str(params[k])

def heuristicCast(s):
    s = s.strip() # Don't let some stupid whitespace fool you.
    if s=="None":
        return None
    elif s=="True":
        return True
    elif s=="False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def paramsFromConf(f):
    params = AttrDict()
    for l in f:
        if l.startswith("#"):
            continue
        try:
            k, v = l.strip("\n").split("\t")
        except:
            assert False, "Malformed config line " + l.strip()
        try:
            v = heuristicCast(v)
        except ValueError:
            assert False, "Malformed parameter value " + v
        params[k] = v
    return params
