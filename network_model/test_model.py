import tensorflow as tf

from network_model.training_data_model import TrainingDataModel
from tree_files import tree_parser
from utils import file_utils
from utils import load_data_utils


class TestModel:

    def __init__(self, tree_file, dna_sequence_file, model_path, sequence_length, max_number_of_leaves, batch_size,
                 dna_num_letters,
                 num_iters_per_tree):
        self.tree_file = tree_file
        self.dna_sequence_file = dna_sequence_file
        self.model_path = model_path
        self.tag_name = "phylogeny_reconstruction"

        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.dna_num_letters = dna_num_letters
        self.num_iters_per_tree = num_iters_per_tree

        self.max_number_of_leaves = max_number_of_leaves

        self.number_of_trees = None
        self.losses, self.accuracies = [], []

    def test(self):
        with tf.Session() as sess:
            self.load_model(sess)

            dna_subtree, dna_sequences_node_1, dna_sequences_node_2, \
            are_nodes_together, accuracy, loss, predictions = self.load_tensors()

            dna_sequences, _, _= self.load_dna_sequences()
            trees = self.load_trees()

            for i in range(0, len(trees)):
                self.predict_over_tree(sess, trees[i], i, dna_sequences, dna_subtree,
                                       dna_sequences_node_1,
                                       dna_sequences_node_2, are_nodes_together, accuracy, loss, predictions)

            self.calculate_accuracy_and_loss_and_write_report()

    def load_model(self, session):
        tf.saved_model.loader.load(session, [self.tag_name], self.model_path)

    @staticmethod
    def load_tensors():
        graph = tf.get_default_graph()

        encoder_dataset_plc = graph.get_tensor_by_name("dna_subtree:0")
        encoder_dna_seq_1_plc = graph.get_tensor_by_name("dna_sequence_node_1:0")
        encoder_dna_seq_2_plc = graph.get_tensor_by_name("dna_sequence_node_2:0")
        together_plc = graph.get_tensor_by_name("are_nodes_together:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        loss = graph.get_tensor_by_name("loss:0")
        predictions = graph.get_tensor_by_name("predictions:0")

        return encoder_dataset_plc, encoder_dna_seq_1_plc, encoder_dna_seq_2_plc, together_plc, accuracy, loss, predictions

    def load_dna_sequences(self):
        return load_data_utils.read_data(self.dna_sequence_file)

    def load_trees(self):
        trees = tree_parser.parse(self.tree_file)
        self.number_of_trees = len(trees)
        return trees

    def predict_over_tree(self, session, tree, tree_index, dna_sequences, dna_subtree, dna_sequence_node_1,
                          dna_sequence_node_2,
                          are_nodes_together, accuracy, loss, prediction):

        current_tree_losses = []
        current_tree_accuracies = []
        for step in range(self.num_iters_per_tree):
            training_data_model = TrainingDataModel(tree, dna_sequences, self.sequence_length,
                                                    self.max_number_of_leaves,
                                                    self.dna_num_letters, dataset_index=tree_index)

            aa = tf.argmax(prediction, axis=1)

            training_data_model.prepare_randomized_batch_sized_data(self.batch_size)
            _accuracy, _loss, _predictions, _aa = session.run(
                [accuracy, loss, prediction, aa],
                feed_dict={
                    dna_subtree: training_data_model.descendants_dna_sequences,
                    dna_sequence_node_1: training_data_model.dna_sequences_node_1,
                    dna_sequence_node_2: training_data_model.dna_sequences_node_2,
                    are_nodes_together: training_data_model.are_nodes_together
                })

            current_tree_losses.append(_loss)
            current_tree_accuracies.append(_accuracy)

        accuracy = self.calculate_mean(current_tree_accuracies)
        loss = self.calculate_mean(current_tree_losses)

        self.accuracies.append(accuracy)
        self.losses.append(loss)

        print("Tree index: [{}/{}]\tLoss: {:.3f}\tAcc: {:.2%}".format(
            tree_index, self.number_of_trees, loss, accuracy))

    @staticmethod
    def calculate_mean(values):
        return sum(values) / float(len(values))

    def calculate_accuracy_and_loss_and_write_report(self):
        avg_acc = self.calculate_mean(self.accuracies)
        avg_loss = self.calculate_mean(self.losses)

        file_utils.write_to_file("\n" + str(self.losses), "\n" + str(self.accuracies), str(avg_loss), str(avg_acc),
                                 path="test/")

        print("Accuracy ", avg_acc)
        print("Loss", avg_loss)
