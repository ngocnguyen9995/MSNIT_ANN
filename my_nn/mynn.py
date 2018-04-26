import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in list(zip(sizes[:-1], sizes[1:]))]

    def feedforward(self, a):
        #return output of network if a is input
        for b, w in list(zip(self.biases, self.weights)):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        # Train the ANN using mini-batch stochastic gradient descent
        # training_data: tuples(x, y) x-> input; y-> desired input

        if (test_data):
            n_test = len(test_data)
        n = len(training_data)
        for j in list(range(epochs)):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in list(range(0, n, mini_batch_size))
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if (test_data):
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        # Update network's weights and biases by apply gradient descent
        # mini_batch: list of tuples(x,y)
        # eta: learning rate

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nable_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in list(zip(nabla_b, delta_nabla_b))]
            nabla_w = [nw+dnw for nw, dnw in list(zip(nabla_w, delta_nable_w))]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in list(zip(self.weights, nabla_w))]
        self.biases = [b-(eta/len(mini_batch))* nb for b, nb in list(zip(self.biases, nabla_b))]

    def backprop(self, x, y):
        # Return a tuple(nabla_b, nabla_w) -> gradient for cost func
        # nabla_b & nabla_w are layer by layer lists of numpy arrays

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] # store all activations layer by layer
        zs = [] # store all z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in list(range(2, self.num_layers)):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        # Return number of test inputs for which the ANN outputs correct result
        test_result = [np.argmax(self.feedforward(x), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_result)
    
    def cost_derivative(self, output_activations, y):
        # Return the vector of partial derivatives for cost func
        return (output_activations-y)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    # Return the derivative of the sigmoid func
    return sigmoid(z) * (1-sigmoid(z))