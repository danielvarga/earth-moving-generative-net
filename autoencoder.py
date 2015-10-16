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

def mnist():
    mnistFile = "../rbm/data/mnist.pkl.gz"
    f = gzip.open(mnistFile, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    X, y = train_set
    X = X.astype(np.float64).reshape((-1, 1, 28, 28))
    # sigma and mu should be trained on the same corpus as the autoencoder itself.
    # This is error-prone!
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
        return self.ae.predict((X - self.mu) / self.sigma).reshape(-1, 28, 28) * self.sigma + self.mu

    def encode(self, X):
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
        return get_output_from_nn(self.final_layer, X) * self.sigma + self.mu

def get_picture_array(X, index):
    array = X[index].reshape(28,28)
    array = np.clip(array*255, a_min = 0, a_max = 255)
    return  array.repeat(4, axis = 0).repeat(4, axis = 1).astype(np.uint8())

def get_picture_array_better(X, n_x, n_y, name):
    image_data = np.zeros(
        (29 * n_y + 1, 29 * n_x - 1),
        dtype='uint8'
    )
    n = len(X)
    assert n <= n_x * n_y
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = X[idx].reshape((28,28))
        image_data[29*x:29*x+28, 29*y:29*y+28] = (255*sample).clip(0, 255)
    img = Image.fromarray(image_data)
    img.save(name+".png")

def get_random_images(X_in, X_pred):
    index = np.random.randint(len(X_pred))
    print index
    original_image = Image.fromarray(get_picture_array(X_in, index))
    new_size = (original_image.size[0] * 2, original_image.size[1])
    new_im = Image.new('L', new_size)
    new_im.paste(original_image, (0,0))
    rec_image = Image.fromarray(get_picture_array(X_pred, index))
    new_im.paste(rec_image, (original_image.size[0],0))
    new_im.save('test1.png', format="PNG")

def main():
    X_train, mu, sigma = mnist()

    autoencoderFile = "../lasagne-demo/conv_ae.pkl"
    ae_raw = cPickle.load(open(autoencoderFile, 'r'))

    autoencoder = Autoencoder(ae_raw, mu, sigma)

    sampleIndices = map(int, sys.argv[1:])
    assert len(sampleIndices)==2, "the tool expects two sample indices"
    X_train = X_train[sampleIndices]

    X_pred = autoencoder.predict(X_train)
    print "ended prediction"
    sys.stdout.flush()

    get_random_images(X_train, X_pred)

    autoencoder.split()

    X_encoded = autoencoder.encode(X_train)

    x0 = X_encoded[0]
    x1 = X_encoded[1]
    stepCount = 100
    intervalBase = np.linspace(1, 0, num=stepCount)
    intervalEncoded = np.multiply.outer(intervalBase, x0)+np.multiply.outer(1.0-intervalBase, x1)

    X_decoded = autoencoder.decode(intervalEncoded)

    get_picture_array_better(X_decoded, 10, 10, "interval")


if __name__ == "__main__":
    main()
