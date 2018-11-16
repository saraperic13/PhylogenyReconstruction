import tensorflow as tf
from network_model.base_network import BaseNetwork

from utils import tensorflow_utils


class EncoderNetwork(BaseNetwork):

    def __init__(self, number_of_neurons_per_layer, batch_size, sequence_length, dna_num_letters):
        super(EncoderNetwork, self).__init__(number_of_neurons_per_layer, batch_size, sequence_length, dna_num_letters)
        self.dna_subtree = None
        self.dna_sequence_node_1 = None
        self.dna_sequence_node_2 = None
        self.are_nodes_together = None
        self.weights = []
        self.biases = []

        self.create_placeholders()
        self.create_weights_biases_matrices()

    def create_placeholders(self):
        self.dna_subtree = tensorflow_utils.make_placeholder(
            shape=[self.batch_size, None, self.sequence_length * self.dna_num_letters],
            name="dna_subtree")

        self.dna_sequence_node_1 = tensorflow_utils.make_placeholder(
            shape=[self.batch_size, self.sequence_length * self.dna_num_letters],
            name="dna_sequence_node_1")
        self.dna_sequence_node_2 = tensorflow_utils.make_placeholder(
            shape=[self.batch_size, self.sequence_length * self.dna_num_letters],
            name="dna_sequence_node_2")
        self.are_nodes_together = tensorflow_utils.make_placeholder(shape=[self.batch_size, 2],
                                                                    name="are_nodes_together")

    def encode(self):
        encoded_dna_sequence_1 = tensorflow_utils.multiply_sequence_weight_matrices(self.dna_sequence_node_1,
                                                                                    self.weights, self.biases)
        encoded_dna_sequence_2 = tensorflow_utils.multiply_sequence_weight_matrices(self.dna_sequence_node_2,
                                                                                    self.weights, self.biases)

        encoded_dataset = self.encode_dataset()

        return encoded_dna_sequence_1, encoded_dna_sequence_2, encoded_dataset

    def encode_dataset(self):
        encoded_dataset = tf.map_fn(
            lambda x: tensorflow_utils.multiply_sequence_weight_matrices(x, self.weights, self.biases),
            self.dna_subtree, dtype=tf.float32)

        encoded_dataset = tf.map_fn(lambda x: tf.reduce_mean(x, axis=0), encoded_dataset,
                                    dtype=tf.float32)
        return encoded_dataset
