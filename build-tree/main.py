import tensorflow as tf

from network_model.training_data_model import TrainingDataModel
from tree_files import tree_parser, tree_utils
from utils import load_data_utils

dna_sequence_file = ""
tree_file = ""
model_path = ""
sequence_length = 100
dna_num_letters = 4
batch_size = 100


def test():
    with tf.Session() as sess:
        load_model(sess)

        dna_subtree, dna_sequences_node_1, dna_sequences_node_2, are_nodes_together, predictions = load_tensors()

        dna_sequences = load_dna_sequences()
        trees = load_trees()

        for i in range(0, len(trees)):
            predict_over_tree(sess, trees[i], i, dna_sequences, dna_subtree,
                              dna_sequences_node_1,
                              dna_sequences_node_2, are_nodes_together, predictions)


def load_dna_sequences():
    return load_data_utils.read_data(dna_sequence_file)


def load_trees():
    trees = tree_parser.parse(tree_file)
    return trees


def load_model(session):
    tf.saved_model.loader.load(session, ["phylogeny_reconstruction"], model_path)


def load_tensors():
    graph = tf.get_default_graph()

    encoder_dataset_plc = graph.get_tensor_by_name("dna_subtree:0")
    encoder_dna_seq_1_plc = graph.get_tensor_by_name("dna_sequence_node_1:0")
    encoder_dna_seq_2_plc = graph.get_tensor_by_name("dna_sequence_node_2:0")
    together_plc = graph.get_tensor_by_name("are_nodes_together:0")
    predictions = graph.get_tensor_by_name("predictions:0")

    return encoder_dataset_plc, encoder_dna_seq_1_plc, encoder_dna_seq_2_plc, together_plc, predictions


def predict_over_tree(session, tree, tree_index, dna_sequences, dna_subtree, dna_sequence_node_1,
                      dna_sequence_node_2,
                      are_nodes_together, accuracy, loss, prediction):
    training_data_model = TrainingDataModel(tree, dna_sequences, sequence_length,
                                            dna_num_letters, dataset_index=tree_index)

    aa = tf.argmax(prediction, axis=1)

    tree_utils.get_batch_sized_data(batch_size, training_data_model)
    _accuracy, _loss, _predictions, _aa = session.run(
        [accuracy, loss, prediction, aa],
        feed_dict={
            dna_subtree: training_data_model.descendants_dna_sequences,
            dna_sequence_node_1: training_data_model.dna_sequences_node_1,
            dna_sequence_node_2: training_data_model.dna_sequences_node_2,
            are_nodes_together: training_data_model.are_nodes_together
        })

    return aa
