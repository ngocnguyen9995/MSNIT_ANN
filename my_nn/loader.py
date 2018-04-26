#### Libraries
import pickle
import gzip
import numpy as np



def load_data():
    # Return MNIST data set as a tuple of training data, validation data, and test data
    # training_data -> tuple(training_input, result) 50k inputs & results
    # validation_data -> tuple(validation_input, result) 10k
    # test_data -> tuple(test_input, result) 10k
    # some further formatting of data used in load_data_wrapper()

    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding = "bytes")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    # Return a formatted tuple(training_data,validation_data,test_data)
    # training_data -> tuple(x,y); x->784d array representing input image; y->10d array representing the 10 possible classifications
    # validation_data/test_data -> tuple(x,y); x -> same; y -> correct classification for x

    tr_d, va_d, te_d = load_data()

    # Format training_data
    training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    # Format validation_data & test_data
    validation_input = [np.reshape(x, (784,1)) for x in va_d[0]]
    validation_data = list(zip(validation_input, va_d[1]))
    test_input = [np.reshape(x, (784,1)) for x in te_d[0]]
    test_data = zip(test_input, te_d[1])

    return (training_data,validation_data,test_data)

def vectorized_result(j):
    # Return a 10d unit vector A with 1.0 in Aj in 0 elsewhere

    e = np.zeros((10,1))
    e[j] = 1.0
    return e