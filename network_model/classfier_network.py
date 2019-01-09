from network_model.base_network import BaseNetwork
from utils import tensorflow_utils


class ClassifierNetwork(BaseNetwork):

    def __init__(self, number_of_neurons_per_layer, data):
        super(ClassifierNetwork, self).__init__(number_of_neurons_per_layer)
        self.data = data
        self.weights = []
        self.biases = []

        self.create_weights_biases_matrices()

    def predict(self):
        result_last_layer, predictions = tensorflow_utils.multiply_sequence_weight_matrices_with_activation(self.data,
                                                                                                            self.weights,
                                                                                                            self.biases)
        return result_last_layer, predictions
