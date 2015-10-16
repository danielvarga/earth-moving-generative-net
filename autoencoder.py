# -*- coding: utf-8 -*-

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.objectives import squared_error as mse

### this is really dumb, current nolearn doesnt play well with lasagne,
### so had to manually copy the file I wanted to this folder
import nnbase.shape as shape

from lasagne.nonlinearities import tanh
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
import os
import gzip
import cPickle
# from IPython.display import Image as IPImage
from PIL import Image

from nnbase.layers import Unpool2DLayer

from nnbase.utils import FlipBatchIterator

mnistFile = "../rbm/data/mnist.pkl.gz"
f = gzip.open(mnistFile, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
X, y = train_set
X = np.rint(X * 256).astype(np.int).reshape((-1, 1, 28, 28))  # convert to (0,255) int range (we'll do our own scaling)
mu, sigma = np.mean(X.flatten()), np.std(X.flatten())

X_train = X.astype(np.float64)
X_train = (X_train - mu) / sigma

# we need our target to be 1 dimensional
X_out = X_train.reshape((X_train.shape[0], -1))

autoencoderFile = "../lasagne-demo/conv_ae.pkl"
ae = cPickle.load(open(autoencoderFile, 'r'))

sampleIndices = map(int, sys.argv[1:])
assert len(sampleIndices)==2, "the tool expects two sample indices"
# OMG what a mess, refactor refactor.
X_train = X_train[sampleIndices]
X = X[sampleIndices]

X_train_pred = ae.predict(X_train).reshape(-1, 28, 28) * sigma + mu
print "ended prediction"
sys.stdout.flush()
X_pred = np.rint(X_train_pred).astype(int)
X_pred = np.clip(X_pred, a_min = 0, a_max = 255)
X_pred = X_pred.astype('uint8')
print X_pred.shape , X.shape

# <codecell>

###  show random inputs / outputs side by side

def get_picture_array(X, index):
    array = X[index].reshape(28,28)
    array = np.clip(array, a_min = 0, a_max = 255)
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
        image_data[29*x:29*x+28, 29*y:29*y+28] = sample.clip(0, 255)
    img = Image.fromarray(image_data)
    img.save(name+".png")

def get_random_images():
    index = np.random.randint(len(X_pred))
    print index
    # N.B. This line uses global X not X_pred:
    original_image = Image.fromarray(get_picture_array(X, index))
    new_size = (original_image.size[0] * 2, original_image.size[1])
    new_im = Image.new('L', new_size)
    new_im.paste(original_image, (0,0))
    rec_image = Image.fromarray(get_picture_array(X_pred, index))
    new_im.paste(rec_image, (original_image.size[0],0))
    new_im.save('test1.png', format="PNG")

get_random_images()

# <codecell>

## we find the encode layer from our ae, and use it to define an encoding function

encode_layer_index = map(lambda pair : pair[0], ae.layers).index('encode_layer')
encode_layer = ae.get_all_layers()[encode_layer_index]

def get_output_from_nn(last_layer, X):
    indices = np.arange(128, X.shape[0], 128)
    sys.stdout.flush()

    # not splitting into batches can cause a memory error
    X_batches = np.split(X, indices)
    out = []
    for count, X_batch in enumerate(X_batches):
        # out.append(last_layer.get_output(X_batch).eval())
        out.append( layers.get_output(last_layer, X_batch).eval() )
        sys.stdout.flush()
    return np.vstack(out)


def encode_input(X):
    return get_output_from_nn(encode_layer, X)

X_encoded = encode_input(X_train)

x0 = X_encoded[0]
x1 = X_encoded[1]
stepCount = 100
intervalBase = np.linspace(1, 0, num=stepCount)
intervalEncoded = np.multiply.outer(intervalBase, x0)+np.multiply.outer(1.0-intervalBase, x1)

# <codecell>

next_layer = ae.get_all_layers()[encode_layer_index + 1]
final_layer = ae.get_all_layers()[-1]
new_layer = layers.InputLayer(shape = (None, encode_layer.num_units))

# N.B after we do this, we won't be able to use the original autoencoder , as the layers are broken up
next_layer.input_layer = new_layer

def decode_encoded_input(X):
    return get_output_from_nn(final_layer, X)

# X_decoded = decode_encoded_input(X_encoded) * sigma + mu
X_decoded = decode_encoded_input(intervalEncoded) * sigma + mu
# ^^^^^

get_picture_array_better(X_decoded, 10, 10, "interval")


X_decoded = np.rint(X_decoded ).astype(int)
X_decoded = np.clip(X_decoded, a_min = 0, a_max = 255)
X_decoded  = X_decoded.astype('uint8')
print X_decoded.shape

### check it worked :

pic_array = get_picture_array(X_decoded, np.random.randint(len(X_decoded)))
image = Image.fromarray(pic_array)
image.save('test2.png', format="PNG")
# IPImage('test.png')

# <codecell>
