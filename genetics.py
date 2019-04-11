import tensorflow as tf
import json
import sys
import neural_network as wnn
import stupidrng

class GeneticsAndNeuralNetworkUnite:
    _activation = {
        "tanh": tf.nn.tanh,
        "sigmoid": tf.nn.sigmoid,
        "softmax": tf.nn.softmax,
        "relu": tf.nn.relu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "exponential": tf.keras.activations.exponential,
        "linear": tf.keras.activations.linear
    }

    def __init__(self, chromosome_definition_file: str, verbosity=False):
        self.verbosity = verbosity
        self.definition_file = chromosome_definition_file
        self._load_data()
        self._generate_population()
    
    def _log(self, string, end="\n"):
        if(self.verbosity):
            print(string, end=end)
            sys.stdout.flush()

    def _load_data(self):
        self._log("Preparing Genetics Algorithms :")
        self._log("     - Preparing MNIST datasets...");     mnist = tf.keras.datasets.mnist
        self._log("     - Loading MNIST datasets [60000]"); (self._x_train, self._y_train),\
                                                            (self._x_test,  self._y_test ) = mnist.load_data()  
        self._log("     - Normalizing Training data");       self._x_train = tf.keras.utils.normalize(self._x_train, axis=1)
        self._log("     - Normalizing Test Data");           self._x_test  = tf.keras.utils.normalize(self._x_test,  axis=1)
        self._log("     ...Finish!")

    def _generate_population(self):
        self._log("Generating Populations...")
        self.population = Population(self._log)
        
        self._log("    | Opening Chromosome Definition File")
        with open(self.definition_file) as f:
            definition = json.loads(f.read())
        
        self._log("    | Generating Individuals...")
        for i in range(definition["individuals_per_population"]):
            self._log("         | Generating Individuals No. {}".format(i+1))
            self._log("         | Randomizing Dense Layers... [", end="")
            dense_layers = stupidrng.integer_range(0, definition["individual_settings"]["dense_layer"], 50, verbosity)
            self._log("]\n         | Random Choosing Activation Layer")
            activation_per_layer = [
                self._activation[stupidrng.iterable_choose(definition["individual_settings"]["hidden_layer_activation"])]\
                     for activation in range(dense_layers)
            ]
            self._log("         | Random Choosing Output Activation")
            output_activation = stupidrng.iterable_choose(definition["individual_settings"]["output_activation"])
            self._log("         | Randomizing Number of Epoch... [", end="")
            epochs = stupidrng.integer_range(
                definition["individual_settings"]["epochs"]["min"],
                definition["individual_settings"]["epochs"]["max"],
                50, verbosity
            )
            self._log("]\n         | Randomizing Number of Neurons per Layer... [", end="")
            neurons_per_layer = [stupidrng.integer_range(
                definition["individual_settings"]["neurons"]["min"],
                definition["individual_settings"]["neurons"]["max"],
                50, verbosity
            )]
            self._log("]\n         | Random Choosing Optimization Function")
            optimizer = stupidrng.iterable_choose(definition["individual_settings"]["optimizer"])
            self._log("         | Enlisting Individual to Population")

            individual = Individual(dense_layers, neurons_per_layer, activation_per_layer, output_activation, optimizer, epochs)
            self.population.add_individuals(individual)
            self._log("         Finished !")


class Population:
    def __init__(self, log_function):
        self._individuals = []
        self._log = log_function
    
    def add_individuals(self, individual):
        self._individuals.append(individual)
    
    def train_population(self, train_data, test_data):
        self._log("Training Individuals...")
        for individual in range(len(self._individuals)):
            self._log("    | Training individual no. {}".format(i))
            i = self._individuals[individual]
            while i.loss == None or i.accuracy == None:
                i.train(train_data, test_data)
        self._log("    Done Training !")

class Individual:
    def __init__(self, dense_layers, neurons_per_layer, activation_per_layer, output_activation, optimizer, epochs):
        self.dense_layer = dense_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_per_layer = activation_per_layer
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.epochs = epochs

        self.loss = None
        self.accuracy = None
    
    def train(self, normalized_train_data, normalized_test_data):
        self.loss, self.accuracy = wnn.train(normalized_train_data, normalized_test_data,
            dense_layers=self.dense_layer,
            neurons_per_layer=self.neurons_per_layer,
            activation_per_layer=self.activation_per_layer,
            optimizer=self.optimizer,
            epochs=self.epochs,
            output_activation=self.output_activation,
        )