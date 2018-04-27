import loader
import mynn
import sys

def main():
    hidden_node_num = int(sys.argv[1])
    epoch_num = int(sys.argv[2])
    mini_batch_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])

    training_data, validation_data, test_data = \
        loader.load_data_wrapper()
    
    net = mynn.Network([784,hidden_node_num,10])
    net.SGD(training_data, epoch_num, mini_batch_size, learning_rate, test_data= test_data)

main()