# -*- coding: utf-8 -*-

from lasagne import layers
import numpy as np

import sys
import gzip
import cPickle
from PIL import Image

from nnbase.layers import Unpool2DLayer
from nnbase.utils import FlipBatchIterator
### this is really dumb, current nolearn doesnt play well with lasagne,
### so had to manually copy the file I wanted to this folder
import nnbase.shape as shape

import nnbase.inputs
import nnbase.vis

# This is very error-prone.
# Optimally, there should be a guarantee that the
# corpus loaded here is the same as the one that the
# encoder was trained on.
def loadCorpus():
    face = True
    if face:
        directory = "../face/SCUT-FBP/thumb.big/"
        X, (height, width) = nnbase.inputs.faces(directory)
    else:
        X, (height, width) = nnbase.inputs.mnist()

    X = X.astype(np.float64).reshape((-1, 1, height, width))
    mu, sigma = np.mean(X), np.std(X)
    print "mu, sigma:", mu, sigma
    return X, mu, sigma

# TODO I don't think that .eval() is how this should work.
def get_output_from_nn(last_layer, X):
    indices = np.arange(128, X.shape[0], 128)
    # not splitting into batches can cause a memory error
    X_batches = np.split(X, indices)
    out = []
    for count, X_batch in enumerate(X_batches):
        out.append( layers.get_output(last_layer, X_batch).eval() )
    return np.vstack(out)

# This helper class deals with
# 1. normalizing input and de-normalizing output
# 2. reshaping output into shape compatible with input, namely (-1, 1, x ,y)
class Autoencoder:
    # sigma and mu should be trained on the same corpus as the autoencoder itself.
    # This is error-prone!
    def __init__(self, ae, mu, sigma):
        self.ae = ae
        self.mu = mu
        self.sigma = sigma

        self.encode_layer_index = map(lambda pair : pair[0], self.ae.layers).index('encode_layer')
        self.encode_layer = self.ae.get_all_layers()[self.encode_layer_index]
        self.afterSplit = False

    # from unnormalized to unnormalized [0,1] MNIST.
    # ae is trained on normalized MNIST data.
    # For 0-1 clipped digits this should be close to the identity function.
    def predict(self, X):
        assert not self.afterSplit
        self.x, self.y = X.shape[-2:]
        flatOutput = self.ae.predict((X - self.mu) / self.sigma).reshape(X.shape) * self.sigma + self.mu
        return flatOutput.reshape((-1, 1, self.x, self.y))

    def encode(self, X):
        self.x, self.y = X.shape[-2:]
        return get_output_from_nn(self.encode_layer, (X-self.mu)/self.sigma)

    # N.B after we do this, we won't be able to use the original autoencoder , as the layers are broken up
    def split(self):
        next_layer = self.ae.get_all_layers()[self.encode_layer_index + 1]
        self.final_layer = self.ae.get_all_layers()[-1]
        new_layer = layers.InputLayer(shape = (None, self.encode_layer.num_units))
        next_layer.input_layer = new_layer
        self.afterSplit = True

    def decode(self, X):
        assert self.afterSplit
        flatOutput = get_output_from_nn(self.final_layer, X) * self.sigma + self.mu
        # Evil hack: decode only knows the shape of the input space
        # if you did a predict or encode previously. TODO Fix asap.
        return flatOutput.reshape((-1, 1, self.x, self.y))


def main():
    X_train, mu, sigma = loadCorpus()

    # autoencoderFile = "../lasagne-demo/conv_ae.pkl" # Trained on the full mnist train dataset
    autoencoderFile = "../lasagne-demo/face.big.pkl" # Trained on the ../face/SCUT-FBP/thumb.big dataset.

    ae_raw = cPickle.load(open(autoencoderFile, 'r'))
    autoencoder = Autoencoder(ae_raw, mu, sigma)

    sampleIndices = map(int, sys.argv[1:])
    assert len(sampleIndices)==2, "the tool expects two sample indices"
    X_train = X_train[sampleIndices]

    X_pred = autoencoder.predict(X_train)
    print "ended prediction"
    sys.stdout.flush()

    nnbase.vis.get_random_images(X_train, X_pred)

    autoencoder.split()

    X_encoded = autoencoder.encode(X_train)

    x0 = X_encoded[0]
    x1 = X_encoded[1]
    stepCount = 100
    intervalBase = np.linspace(1, 0, num=stepCount)
    intervalEncoded = np.multiply.outer(intervalBase, x0)+np.multiply.outer(1.0-intervalBase, x1)

    X_decoded = autoencoder.decode(intervalEncoded)
    nnbase.vis.get_picture_array(X_decoded, 10, 10, "interval")

    intervalInputspace = np.multiply.outer(intervalBase, X_train[0])+np.multiply.outer(1.0-intervalBase, X_train[1])
    nnbase.vis.get_picture_array(intervalInputspace, 10, 10, "interval-inputspace")



if __name__ == "__main__":
    main()
