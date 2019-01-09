import tensorflow as tf

from network_model.base_network import BaseNetwork
from utils import tensorflow_utils


class EncoderNetwork(BaseNetwork):

    def __init__(self, number_of_neurons_per_layer):
        super(EncoderNetwork, self).__init__(number_of_neurons_per_layer)
        self.dna_subtree = None
        self.dna_sequence_node_1 = None
        self.dna_sequence_node_2 = None
        self.are_nodes_together = None
        self.number_of_leaves = None
        self.weights = []
        self.biases = []

        self.create_placeholders()
        num_of_rows = self.dna_sequence_node_1.get_shape().as_list()[1]
        self.create_weights_biases_matrices(num_of_rows)

    def create_placeholders(self):
        self.dna_subtree = tensorflow_utils.make_placeholder(
            # shape=[self.batch_size, None, self.sequence_length * self.dna_num_letters],
            shape=[None, None, None],
            name="dna_subtree")

        self.dna_sequence_node_1 = tensorflow_utils.make_placeholder(
            # shape=[self.batch_size, self.sequence_length * self.dna_num_letters],
            shape=[None, None],
            name="dna_sequence_node_1")
        self.dna_sequence_node_2 = tensorflow_utils.make_placeholder(
            # shape=[None, self.sequence_length * self.dna_num_letters],
            shape=[None, None],
            name="dna_sequence_node_2")
        self.are_nodes_together = tensorflow_utils.make_placeholder(shape=[None, 2],
                                                                    name="are_nodes_together")
        # self.number_of_leaves = tensorflow_utils.make_fill(name="number_of_leaves", value=20)
        self.number_of_leaves = tensorflow_utils.make_constant(shape=[], name="number_of_leaves", value=20)

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

        encoded_dataset = tf.reshape(encoded_dataset, [tf.shape(self.dna_sequence_node_1)[0], self.number_of_leaves *
                                                       self.number_of_neurons_per_layer[-1]])

        encoder_2_w, encoder_2_b = [], []

        tensorflow_utils.create_and_append_matrix_dynamic_shape((tf.shape(encoded_dataset)[1],
                                                                 self.number_of_neurons_per_layer[-1]), encoder_2_w)

        tensorflow_utils.create_and_append_matrix(1,
                                                  self.number_of_neurons_per_layer[-1], encoder_2_b)

        encoded_dataset = tensorflow_utils.multiply_sequence_weight_matrices(encoded_dataset, encoder_2_w, encoder_2_b)

        return encoded_dataset
