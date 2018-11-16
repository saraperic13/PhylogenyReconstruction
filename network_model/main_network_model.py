import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from network_model.encoder_network import EncoderNetwork
from network_model.classfier_network import ClassifierNetwork
from network_model.training_data_model import TrainingDataModel
from tree_files import tree_parser
from tree_files import tree_utils
from utils import load_data_utils


class MainNetworkModel:

    def __init__(self, number_of_neurons_per_layer_encoder, number_of_neurons_per_layer_classifier,
                 batch_size, sequence_length, learning_rate, num_training_iters, tree_file, dna_sequence_file,
                 model_path, dna_num_letters=4):
        self.number_of_neurons_per_layer_encoder = number_of_neurons_per_layer_encoder
        self.number_of_neurons_per_layer_classifier = number_of_neurons_per_layer_classifier

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dna_num_letters = dna_num_letters
        self.learning_rate = learning_rate
        self.num_training_iters = num_training_iters
        self.tree_file = tree_file
        self.dna_sequence_file = dna_sequence_file
        self.model_path = model_path

        self.encoder_network = None
        self.classifier_network = None

        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.training_alg = None

        self.initialize_networks()

    def initialize_networks(self):
        encoder_output = self.initialize_encoder_and_encode()
        self.initialize_classifier(encoder_output)

    def initialize_encoder_and_encode(self):
        self.encoder_network = EncoderNetwork(self.number_of_neurons_per_layer_encoder, self.batch_size,
                                              self.sequence_length,
                                              self.dna_num_letters)
        encoded_dataset, encoded_dna_sequence_1, encoded_dna_sequence_2 = self.encoder_network.encode()

        return tf.concat([encoded_dataset, encoded_dna_sequence_1, encoded_dna_sequence_2], 1)

    def initialize_classifier(self, input_data):
        # encoder_output_size * 3, feed_forward_hidden_units_1, feed_forward_hidden_units_2,
        # feed_forward_hidden_units_3, 2

        self.classifier_network = ClassifierNetwork(
            self.number_of_neurons_per_layer_classifier,
            self.batch_size, self.sequence_length, self.dna_num_letters, input_data)

    def train(self):
        self.predict()
        self.run_session()

    def predict(self):
        labels = self.encoder_network.are_nodes_together

        output, self.predictions = self.classifier_network.predict()

        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
        self.loss = tf.reduce_mean(losses, name="loss")

        self.training_alg = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        correct_pred = tf.equal(tf.argmax(self.predictions, axis=1), tf.argmax(labels, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    def run_session(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.train_network(sess)
            self.save_model(sess)

    def train_network(self, session):
        dna_sequences = load_data_utils.read_data(self.dna_sequence_file)
        trees = tree_parser.parse(self.tree_file)

        for step in range(self.num_training_iters + 1):

            tree_index = 0

            for tree in trees:
                self.train_over_tree(dna_sequences, tree, tree_index, session)
                tree_index += 1

    def train_over_tree(self, dna_sequences, tree, tree_index, session):
        training_data_model = TrainingDataModel(tree, dna_sequences, self.sequence_length,
                                                self.dna_num_letters, dataset_index=tree_index)

        tree_utils.get_batch_sized_data(self.batch_size, training_data_model)

        _loss, _predictions, _accuracy, _train_alg = session.run(
            [self.loss, self.predictions, self.accuracy, self.training_alg],
            feed_dict={
                self.encoder_network.dna_subtree: training_data_model.descendants_dna_sequences,
                self.encoder_network.dna_sequence_node_1: training_data_model.dna_sequences_node_1,
                self.encoder_network.dna_sequence_node_2: training_data_model.dna_sequences_node_2,
                self.encoder_network.are_nodes_together: training_data_model.are_nodes_together
            })
        self.print_to_screen(tree_index, _accuracy, _loss)

    def print_to_screen(self, step, accuracy, loss):

        if step % 1 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, accuracy))

    def save_model(self, session):
        builder, signature = self.create_model_signature()
        builder.add_meta_graph_and_variables(sess=session,
                                             tags=["phylogeny_reconstruction"],
                                             signature_def_map={'predict': signature})

        builder.save()

    def create_model_signature(self):
        builder = tf.saved_model.builder.SavedModelBuilder(self.model_path)

        signature = predict_signature_def(
            inputs={'dna_subtree': self.encoder_network.dna_subtree,
                    'dna_sequence_node_1': self.encoder_network.dna_sequence_node_1,
                    'dna_sequence_node_2': self.encoder_network.dna_sequence_node_2,
                    'are_nodes_together': self.encoder_network.are_nodes_together},
            outputs={'predictions': self.predictions})
        return builder, signature
