# This is a line-by-line interpretation of the VAE coding example in: https://github.com/y0ast/Variational-Autoencoder/blob/master/VAE.py

from __future__ import division
# This line is ensuring that the division operator / behaves consistently between Python 2 and Python 3.
# In Python 2, dividing two integers (e.g., 5 / 2) would perform integer division (resulting in 2), but in Python 3,
# it would result in floating-point division (resulting in 2.5).
# Importing division from the __future__ module ensures that division behaves like Python 3 (floating-point division)
# even in Python 2, making the code more forward-compatible.
import numpy as np
# NumPy is a fundamental package for scientific computing in Python, providing support for large,
# multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
# Here, numpy is imported as np for shorthand use in the code. You’ll be able to use NumPy functions like np.array(),
# np.dot(), etc., to perform array and matrix operations.
import theano
# Theano is a deep learning library that allows you to define, optimize, and evaluate mathematical expressions involving
# multi-dimensional arrays efficiently.
# Theano is often used to build symbolic representations of functions and expressions, and it supports automatic differentiation,
# making it suitable for machine learning and deep learning tasks.
import theano.tensor as T
# This imports the tensor module from Theano and aliases it as T for convenience.
# Theano’s tensor module provides symbolic data structures for manipulating multi-dimensional arrays (tensors).
# These are analogous to NumPy arrays, but they represent symbolic expressions instead of actual data.
# For example, T.scalar(), T.vector(), and T.matrix() are used to define symbolic variables for scalars, vectors,
# and matrices, respectively, that can later be used in mathematical operations within Theano's computational graph.
import pickle
# This line imports the pickle module, which is used for serializing (converting Python objects into a byte stream)
# and deserializing (converting the byte stream back into Python objects).
# Common use case:
# Saving Python objects like lists, dictionaries, or even complex objects (e.g., machine learning models) to a file, so they can be loaded and used later.
# Loading previously saved objects from a file back into a Python program.
from collections import OrderedDict
#This line imports the OrderedDict class from the collections module.
# OrderedDict is a specialized dictionary that remembers the order in which key-value pairs are inserted.
# In contrast, a regular Python dictionary does not guarantee insertion order in versions prior to Python 3.7
# (though in Python 3.7+, regular dicts maintain insertion order as well).
# Common use case: When you need to ensure that the order of items in the dictionary is preserved,
# such as for serialization or when the insertion order has semantic meaning.

epsilon = 1e-8
def relu(x):
    return T.switch(x<0, 0, x)
#T.switch is a function from Theano that works like a conditional statement:, If x < 0, return 0. Otherwise, return x.

class VAE:
    """This class implements the Variational Auto Encoder"""
    def __init__(self, continuous, hu_encoder, hu_decoder, n_latent, x_train, b1=0.95, b2=0.999, batch_size=100, learning_rate=0.001, lam=0, L=1):
        # continuous: whether the model is continuous or discrete
        # hu_encoder: number of hidden nodes in the encoder
        # hu_decoder: number of hidden nodes in the decoder
        # n_latent: number of latent variables in the latent space
        # x_train: training data
        # b1, b2: hyperparameters for the adam optimizer, decay rates for the moving averages of the gradient and its square
        # batch_size: number of data points per batch
        # learning_rate: learning rate for optimization
        # lam: regularization parameter
        # L: number of datapoints drawn per point

        # Pass parameters to the defined item. N is the number of training samples, features is the number of covariates
        self.continuous = continuous
        self.hu_encoder = hu_encoder
        self.hu_decoder = hu_decoder
        self.n_latent = n_latent
        [self.N, self.features] = x_train.shape

        # Initialize a pseudo-random number generator with seed 42.
        self.prng = np.random.RandomState(42)

        # Set b1, b2 for Adam optimizer, learning rate, and the regularization parameter lambda
        self.b1 = b1
        self.b2 = b2
        self.learning_rate = learning_rate
        self.lam = lam

        # number of samples z^(i,l) per datapoint, Monte Carlo simulation
        self.L = L

        self.batch_size = batch_size

        sigma_init = 0.01

        # weights and biases in the NN are initialized.
        # Weights are initialized using a Gaussian distribution with standard deviation 0.01.
        # Biases are all zero
        create_weight = lambda dim_input, dim_output: self.prng.normal(0, sigma_init, (dim_input, dim_output)).astype(theano.config.floatX)
        create_bias = lambda dim_output: np.zeros(dim_output).astype(theano.config.floatX)

        # encoder
        # Input should be number of features, output is the number of hidden nodes in encoder NN
        W_xh = theano.shared(create_weight(self.features, hu_encoder), name='W_xh')
        b_xh = theano.shared(create_bias(hu_encoder), name='b_xh')

        # This theano.shared() function creates a shared variable in Theano. Shared variables are used to store values that need to be updated during training,
        # such as the weights of a neural network. They allow you to keep track of these values across different function calls and updates.

        # Construct the layer that calculate the latent variables mu and log of sigma
        W_hmu = theano.shared(create_weight(hu_encoder, n_latent), name='W_hmu')
        b_hmu = theano.shared(create_bias(n_latent), name='b_hmu')

        W_hsigma = theano.shared(create_weight(hu_encoder, n_latent), name='W_hsigma')
        b_hsigma = theano.shared(create_bias(n_latent), name='b_hsigma')

        # decoder
        # Construct decoder neural network, transform from latent variables to decoder
        W_zh = theano.shared(create_weight(n_latent, hu_decoder), name='W_zh')
        b_zh = theano.shared(create_bias(hu_decoder), name='b_zh')

        # save parameters
        self.params = OrderedDict([("W_xh", W_xh), ("b_xh", b_xh), ("W_hmu", W_hmu), ("b_hmu", b_hmu),
                                   ("W_hsigma", W_hsigma), ("b_hsigma", b_hsigma), ("W_zh", W_zh),
                                   ("b_zh", b_zh)])

        if self.continuous:
            W_hxmu = theano.shared(create_weight(hu_decoder, self.features), name='W_hxmu')
            b_hxmu = theano.shared(create_bias(self.features), name='b_hxmu')

            W_hxsig = theano.shared(create_weight(hu_decoder, self.features), name='W_hxsigma')
            b_hxsig = theano.shared(create_bias(self.features), name='b_hxsigma')

            self.params.update({'W_hxmu': W_hxmu, 'b_hxmu': b_hxmu, 'W_hxsigma': W_hxsig, 'b_hxsigma': b_hxsig})
        else:
            W_hx = theano.shared(create_weight(hu_decoder, self.features), name='W_hx')
            b_hx = theano.shared(create_bias(self.features), name='b_hx')

            self.params.update({'W_hx': W_hx, 'b_hx': b_hx})

        # Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key, value in self.params.items():
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)

        x_train = theano.shared(x_train.astype(theano.config.floatX), name="x_train")

        self.update, self.likelihood, self.encode, self.decode = self.create_gradientfunctions(x_train)

        # Define encoder function
        def encoder(self, x):
            h_encoder = relu(T.dot(x, self.params['W_xh']) + self.params['b_xh'].dimshuffle('x', 0)) # T.dot is matrix multiplication

            mu = T.dot(h_encoder, self.params['W_hmu']) + self.params['b_hmu'].dimshuffle('x', 0)
            log_sigma = T.dot(h_encoder, self.params['W_hsigma']) + self.params['b_hsigma'].dimshuffle('x', 0)

            return mu, log_sigma

        def sampler(self, mu, log_sigma):
            seed = 42

            if "gpu" in theano.config.device:
                srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
            else:
                srng = T.shared_randomstreams.RandomStreams(seed=seed)
            # Generate latent variables
            eps = srng.normal((self.L, mu.shape[0], self.n_latent))

            # Reparametrize
            z = mu + T.exp(0.5 * log_sigma) * eps

            return z

        def decoder(self, x, z):
            h_decoder = relu(T.dot(z, self.params['W_zh']) + self.params['b_zh'].dimshuffle('x', 0))

            if self.continuous:
                reconstructed_x = T.dot(h_decoder, self.params['W_hxmu']) + self.params['b_hxmu'].dimshuffle('x', 0)
                log_sigma_decoder = T.dot(h_decoder, self.params['W_hxsigma']) + self.params['b_hxsigma']

                logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                          0.5 * ((x - reconstructed_x) ** 2 / T.exp(log_sigma_decoder))).sum(axis=2).mean(axis=0)
            else:
                reconstructed_x = T.nnet.sigmoid(
                    T.dot(h_decoder, self.params['W_hx']) + self.params['b_hx'].dimshuffle('x', 0))
                logpxz = - T.nnet.binary_crossentropy(reconstructed_x, x).sum(axis=2).mean(axis=0)

            return reconstructed_x, logpxz

        def create_gradientfunctions(self, x_train):
            x = T.matrix("x")

            epoch = T.scalar("epoch")

            batch_size = x.shape[0]

            mu, log_sigma = self.encoder(x)
            z = self.sampler(mu, log_sigma)
            reconstructed_x, logpxz = self.decoder(x, z)

            # Expectation of (logpz - logqz_x) over logqz_x is equal to KLD (see appendix B):
            KLD = 0.5 * T.sum(1 + log_sigma - mu ** 2 - T.exp(log_sigma), axis=1)

            # Average over batch dimension
            logpx = T.mean(logpxz + KLD)

            # Compute all the gradients
            gradients = T.grad(logpx, list(self.params.values()))

            # Adam implemented as updates
            updates = self.get_adam_updates(gradients, epoch)

            batch = T.iscalar('batch')

            givens = {
                x: x_train[batch * self.batch_size:(batch + 1) * self.batch_size, :]
            }

            # Define a bunch of functions for convenience
            update = theano.function([batch, epoch], logpx, updates=updates, givens=givens)
            likelihood = theano.function([x], logpx)
            encode = theano.function([x], z)
            decode = theano.function([z], reconstructed_x)

            return update, likelihood, encode, decode
