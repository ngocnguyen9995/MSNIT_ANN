"""
----mynn3.py----

A simple neural network using Theano for optimization and running on GPU.

Supports several layer types (fully connected, convolutional, max pooling, softmax),
and activation functions (sigmoid, tanh, relu, etc)

"""

### Libraries
import pickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool


# Activation functions
def linear(z): return z
def ReLu(z): return T.maximum(0.0, z);
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

### CONST
GPU = True

if GPU:
    print("Running on GPU")
    try: theano.config.device = 'gpu'
    except: pass # already set
    theano.config.floatX = 'float32'
else:
    print("Running on CPU")

def load_data_shared(filename = "../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    def shared(data):
        """Place data into shared variables, allowing Theano to copy data to GPU"""
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow = True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

### Main Neural Network Class
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Constructor for Network object
            layers => list; Describes network architecture
            mini_batch_size => int; batch size to be used for SDC

        """

        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layers.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dopout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn = sigmoid, p_dropout = 0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p.p_dropout

        # Initialize weights & biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc = 0.0, scale = 1.0, size = (n_in, n_out)),
                dtype= theano.config.floatX
            ),
            name = 'w',
            borrow = True
        )

        self.b = theano.shared(
            np.asarray(
                np.random.normal(
                    loc = 0.0, scale = 1.0, size = (n_out,)
                ),
                dtype= theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )

        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
                        (1-self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis = 1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)),
            self.p_dropout
        )
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b
        )

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))

### Helper functions
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)