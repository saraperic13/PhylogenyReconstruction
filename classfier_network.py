import tensorflow_utils
from base_network import BaseNetwork


class ClassifierNetwork(BaseNetwork):

    def __init__(self, number_of_neurons_per_layer, batch_size, sequence_length, dna_num_letters, data):
        super(ClassifierNetwork, self).__init__(number_of_neurons_per_layer, batch_size, sequence_length,
                                                dna_num_letters)
        self.data = data
        self.weights = []
        self.biases = []

        self.create_weights_biases_matrices()

    def predict(self):
        result_last_layer, predictions = tensorflow_utils.multiply_sequence_weight_matrices_with_activation(self.data,
                                                                                                            self.weights,
                                                                                                            self.biases)
        return result_last_layer, predictions
