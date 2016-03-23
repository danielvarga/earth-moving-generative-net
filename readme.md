# Earth moving generative net

This is a neural network model capable of learning high dimensional probability distributions by sampling them.
Its generative model is the most commonly used one: It samples from some simple fixed prior distribution,
and the trained feed-forward neural
network transforms that into the generated sample. It differs from models of this kind (Variational
Autoencoders, Moment Matching Networks, Generative Adversarial Networks) in how training happens.
(Another difference from some of these is that the input distribution can be arbitrary, for
example discrete and mixture input distributions are allowed and practical.)

The training optimizes an empirical approximation to the so-called
[Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance).
A minibatch SGD training step takes *n* observations, samples *n* points from the generative model,
pairs the generated points with the observations, and updates the neural weights to decrease
the sum of (squared) pairwise distances.

This repo contains a Theano implementation of the model. It's undocumented, but the code is relatively
self-explanatory. The main executable takes a configuration file as it's single parameter.
Such configuration files can be found in the `deepDives` and `adhoc` directories.

```
python earthMover.py deepDives/conf8.txt
```

Some visualizations on MNIST and synthetic distributions:

Generating:

![Generating from MNIST](http://people.mokk.bme.hu/~daniel/kohonen/conf8/s5600.png)

Approximating unseen samples:

![Approximating unseen samples from MNIST](http://people.mokk.bme.hu/~daniel/kohonen/conf8/diff_validation5600.png)

Applying the transformation to a fixed input plane:

![Applying the transformation to a fixed input plane](http://people.mokk.bme.hu/~daniel/kohonen/conf8/xy5600.png)

Finding the "right" parametrization for a simple synthetic distribution:

![Clock](http://people.mokk.bme.hu/~daniel/kohonen/clock1-sd1.0/input.png)

![Generated](http://people.mokk.bme.hu/~daniel/kohonen/clock1-sd1.0/xy200.png)
