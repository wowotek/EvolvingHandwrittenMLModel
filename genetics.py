import neural_network as wnn
import tensorflow as tf

class GeneticsAndNeuralNetworkUnite:
    _optimizers = {
        1: 'sgd',
        2: 'rmsprop',
        3: 'adagrad',
        4: 'adadelta',
        5: 'adam',
        6: 'adamax',
        7: 'nadam',
        8: 'sgd',
    }

    _activation = {
        1: 'tanh',
        2: 'sigmoid',
        3: 'exponential',
        4: 'linear',
        5: 'softmax',
        6: 'relu',
        7: 'elu',
        8: 'selu'
    }

    def __init__(self, init_population):
        self.init_population = init_population

    def _load_data(self):
        print("Preparing Genetics Algorithms :")
        print("    - Preparing MNIST datasets...");     mnist = tf.keras.datasets.mnist
        print("    - Loading MNIST datasets [60000]"); (self._x_train, self._y_train),\
                                                       (self._x_test,  self._y_test ) = mnist.load_data()  
        print("    - Normalizing Training data");       self._x_train = tf.keras.utils.normalize(self._x_train, axis=1)
        print("    - Normalizing Test Data");           self._x_test  = tf.keras.utils.normalize(self._x_test,  axis=1)
        print("   ...Finish!")

    def _generate_population(self):
        MAX_DENSE_LAYERS = 8
        MAX_NEURONS = 1280

        dense_layers = 2
        neurons_per_layer = [128, 128]
        activation_per_layer = [tf.nn.relu, tf.nn.relu]
        optimizer = 'adam'
        epochs = 8
        output_activation = tf.nn.softmax