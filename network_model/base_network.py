from utils import tensorflow_utils


class BaseNetwork:

    def __init__(self, number_of_neurons_per_layer, batch_size, sequence_length, dna_num_letters=4):
        self.number_of_neurons_per_layer = number_of_neurons_per_layer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dna_num_letters = dna_num_letters

        self.weights = []
        self.biases = []

    def create_weights_biases_matrices(self, dynamic_shape_dimension=None):
        start_range = 0

        if dynamic_shape_dimension is not None:
            self.create_weights_biases(dynamic_shape_dimension, self.number_of_neurons_per_layer[1])
            start_range = 1

        for layer in range(start_range, len(self.number_of_neurons_per_layer) - 1):
            self.create_weights_biases(self.number_of_neurons_per_layer[layer],
                                       self.number_of_neurons_per_layer[layer + 1])

    def create_weights_biases(self, rows, columns):
        tensorflow_utils.create_and_append_matrix(rows,
                                                  columns, self.weights)

        tensorflow_utils.create_and_append_matrix(1,
                                                  columns, self.biases)
