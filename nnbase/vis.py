import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import sys

def plotSampledImages(net_fn, inDim, name, height, width, fromGrid, gridSize, plane=None, sampleSourceFunction=None):
    data = sampleImageForVis(net_fn, inDim, height, width, fromGrid, gridSize, plane=plane, sampleSourceFunction=sampleSourceFunction)
    plotImages(data, gridSize, name)

def sampleImageForVis(net_fn, inDim, height, width, fromGrid, gridSize, plane=None, sampleSourceFunction=None):
    randomState = np.random.get_state()
    np.random.seed(1)
    if fromGrid:
        if plane is None:
            plane = (0, 1)
        n_x = gridSize
        n_y = gridSize
        initial = []
        for x in np.linspace(-2, +2, n_x):
            for y in np.linspace(-2, +2, n_y):
                v = np.zeros(inDim, dtype=np.float32)
                if plane[0]<inDim:
                    v[plane[0]] = x
                if plane[1]<inDim:
                    v[plane[1]] = y
                initial.append(v)
        data = net_fn(initial)
    else:
        assert plane is None, "unsupported"
        assert sampleSourceFunction is not None
        n_x = gridSize
        n_y = gridSize
        n = n_x*n_y
        initial, data = sampleSourceFunction(net_fn, n, inDim)

    data = data.reshape((-1, height, width))
    np.random.set_state(randomState)
    return data

# shape (-1, height, width)
def plotImages(data, gridSize, name):
    height, width = data.shape[-2:]
    height_inc = height+1
    width_inc = width+1
    n_x = n_y = gridSize
    n = len(data)
    assert n <= n_x*n_y

    image_data = np.zeros(
        (height_inc * n_y + 1, width_inc * n_x - 1),
        dtype='uint8'
    )
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = data[idx].reshape((height, width))
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data)
    img.save(name+".png")


def plot(sampled, name):
    # If feature dim >> 2, and PCA has not happened, it's not too clever to plot the first two dims.
    plt.scatter(sampled.T[0], sampled.T[1])
    plt.savefig(name+".pdf")
    plt.close()

def gradientMap1D(data, sampled, postSampled, name):
    n = len(data)
    assert data.shape == sampled.shape == postSampled.shape == (n,1)
    plt.clf()
    plt.axis((-2, +2, -2, +2))
    import random
    for i in range(len(data)):
        # plt.arrow(sampled[i,0], data[i,0], 0, postSampled[i,0]-data[i,0], head_width=0.005)
        h = random.random()
        plt.arrow(sampled[i,0], h, data[i,0]-sampled[i,0], 0.1, head_width=0.005, color="blue")
        plt.arrow(sampled[i,0], h, postSampled[i,0]-sampled[i,0], 0.2, head_width=0.005, color="red")
    plt.savefig(name+".pdf")
    plt.close()



def get_picture_array_simple(X, height, width, index):
    array = X[index].reshape((height, width))
    array = np.clip(array*255, a_min = 0, a_max = 255)
    return  array.repeat(4, axis = 0).repeat(4, axis = 1).astype(np.uint8())

def get_random_images(X_in, X_pred):
    index = np.random.randint(len(X_pred))
    print index
    height, width = X_in.shape[2:]
    original_image = Image.fromarray(get_picture_array_simple(X_in, height, width, index))
    new_size = (original_image.size[0] * 2, original_image.size[1])
    new_im = Image.new('L', new_size)
    new_im.paste(original_image, (0,0))
    rec_image = Image.fromarray(get_picture_array_simple(X_pred, height, width, index))
    new_im.paste(rec_image, (original_image.size[0],0))
    new_im.save('test1.png', format="PNG")

def get_numpy_picture_array(X, n_x, n_y):
    height, width = X.shape[-2:]
    image_data = np.zeros(
        ((height+1) * n_y - 1, (width+1) * n_x - 1),
        dtype='uint8'
    )
    n = len(X)
    assert n <= n_x * n_y
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = X[idx]
        image_data[(height+1)*y:(height+1)*y+height, (width+1)*x:(width+1)*x+width] = (255*sample).clip(0, 255)
    return image_data

def get_picture_array(X, n_x, n_y, name):
    image_data = get_numpy_picture_array(X, n_x, n_y)
    img = Image.fromarray(image_data)
    img.save(name+".png")

# That's awkward, (height, width) corresponds to (n_y, n_x),
# namely the image size is ~ (width*n_x, height*n_y), but the order is reversed between the two.
def diff_vis(dataOriginal, generatedOriginal, height, width, n_x, n_y, name, distances=None):
    data = dataOriginal.copy().reshape((-1, height, width))
    generated = generatedOriginal.copy().reshape((-1, height, width))
    if distances is not None:
        # Ad hoc values, especially now that there's no bimodality aka left bump of 1s.
        VALLEY = 0.3
        MAX_OF_REASONABLE = 1.0
        assert len(distances) == len(data) == len(generated)
        for i,distance in enumerate(distances):
            length = np.linalg.norm(data[i])
            relativeDistance = (distance+1e-5)/(length+1e-5)
            barHeight = min((int(height*relativeDistance/MAX_OF_REASONABLE), height))
            goGreen = float(relativeDistance<VALLEY) # 1.0 if we want green, 0.0 if we want red.
            # data is drawn red, generated is drawn green,
            # we hackishly manipulate them here to get the needed color.
            data     [i, :barHeight, :2] = 1.0-goGreen
            generated[i, :barHeight, :2] = goGreen

    image_data      = get_numpy_picture_array(data, n_x, n_y)
    image_generated = get_numpy_picture_array(generated, n_x, n_y)
    # To color-combine the images AFTER they are arranged in a grid is
    # more than a little hackish.
    blue = np.minimum(image_data, image_generated) # image_data/2 + image_generated/2
    rgb = np.dstack((image_data, image_generated, blue))
    img = Image.fromarray(rgb, 'RGB')
    img.save(name+".png")

def plotGradients(data, sampled, initial, net_fn, filename):
    updated = net_fn(initial)
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    for x, y, z in zip(data, sampled, updated):
        # print np.linalg.norm(x-y)-np.linalg.norm(x-z), x, y, z
        plt.arrow(y[0], y[1], (x-y)[0], (x-y)[1], color=(1,0,0),  head_width=0.05, head_length=0.1)
        plt.arrow(y[0], y[1], (z-y)[0], (z-y)[1], color=(0,0,1), head_width=0.05, head_length=0.1)
    plt.savefig(filename+".pdf")
    plt.close()


def plot_distance_histogram(distances, filename, data=None):
    doNormalization = False
    if doNormalization:
        assert data is not None
        assert len(data.shape)==2
        lengths = np.linalg.norm(data, axis=1)
        assert lengths.shape == distances.shape
        normalizedDistances = distances/lengths
        values = normalizedDistances
    else:
        values = distances

    try:
        # Ad hoc value, hist fails with even one nan value.
        values = np.clip(values, 0.0, 1000.0)
        plt.hist(values, 20, normed=0, facecolor='green')
        plt.savefig(filename+".pdf")
        plt.close()
    except AttributeError:
        sys.stderr.write("Unable to create %s.pdf\n" % filename)


def heatmap(data, filename):
    binGridSize = 100
    dim = data.shape[1]
    assert dim in (1,2)
    if dim==1:
        plt.clf()
        plt.hist(data[:,0], bins=50)
        plt.savefig(filename+".png")
    else:
        x = data[:, 0].tolist()
        y = data[:, 1].tolist()
        hmap, xedges, yedges = np.histogram2d(x, y, bins=(binGridSize, binGridSize))
        # TODO Output is mixed.
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.clf()
        plt.imshow(hmap, extent=extent)
        plt.savefig(filename+".png")
