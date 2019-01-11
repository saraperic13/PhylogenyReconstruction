import tensorflow as tf

from network_model.training_data_model import TrainingDataModel
from utils import load_data_utils

tree_file = "../dataset/5.2.tree"
dna_sequence_file = "../dataset/seq_5.2.txt"
model_path = "../models/5_20_2/"

sequence_length = 100
dna_num_letters = 4
batch_size = 50


def predict():
    with tf.Session() as sess:
        load_model(sess)

        dna_subtree, dna_sequences_node_1, dna_sequences_node_2, are_nodes_together, number_of_leaves, predictions \
            = load_tensors()

        dna_sequences, num_of_trees_to_infer = load_dna_sequences()

        for i in range(num_of_trees_to_infer):
            training_data_model = predict_tree(sess, i, dna_sequences, dna_subtree,
                                               dna_sequences_node_1,
                                               dna_sequences_node_2, are_nodes_together, number_of_leaves, predictions)

            tree = build_tree(training_data_model)

            print_tree(tree)


def load_dna_sequences():
    return load_data_utils.read_data(dna_sequence_file)


def load_model(session):
    tf.saved_model.loader.load(session, ["phylogeny_reconstruction"], model_path)


def load_tensors():
    graph = tf.get_default_graph()

    encoder_dataset_plc = graph.get_tensor_by_name("dna_subtree:0")
    encoder_dna_seq_1_plc = graph.get_tensor_by_name("dna_sequence_node_1:0")
    encoder_dna_seq_2_plc = graph.get_tensor_by_name("dna_sequence_node_2:0")
    together_plc = graph.get_tensor_by_name("are_nodes_together:0")
    number_of_leaves = graph.get_tensor_by_name("number_of_leaves:0")

    predictions = graph.get_tensor_by_name("predictions:0")

    return encoder_dataset_plc, encoder_dna_seq_1_plc, encoder_dna_seq_2_plc, together_plc, \
           number_of_leaves, predictions


def predict_tree(session, dataset_index, dna_sequences, dna_subtree, dna_sequence_node_1,
                 dna_sequence_node_2,
                 are_nodes_together, number_of_leaves, prediction):
    training_data_model = TrainingDataModel(None, dna_sequences, sequence_length,
                                            dna_num_letters, dataset_index=dataset_index)

    nodes_together_predicted = tf.argmax(prediction, axis=1)

    training_data_model.prepare_node_pairs()
    _predictions, _nodes_together_predicted = session.run(
        [prediction, nodes_together_predicted],
        feed_dict={
            dna_subtree: training_data_model.descendants_dna_sequences,
            dna_sequence_node_1: training_data_model.dna_sequences_node_1,
            dna_sequence_node_2: training_data_model.dna_sequences_node_2,
            are_nodes_together: training_data_model.are_nodes_together,
            number_of_leaves: len(dna_sequences.keys())
        })
    print(_predictions)
    training_data_model.are_nodes_together = _predictions
    return training_data_model


def build_tree(training_data_model):
    indexes = list(range(len(training_data_model.are_nodes_together)))

    # Calculate difference between predicted values, bigger the difference
    # more certain the model is about the prediction
    diff = [pair[0] - pair[1] for pair in training_data_model.are_nodes_together]
    abs_diff = [abs(i) for i in diff]

    # Sort absolute differences values in order to firstly connect nodes the model is certain about
    sorted_indexes = [x for _, x in sorted(zip(abs_diff, indexes), reverse=True)]

    tree = []

    for index in sorted_indexes:

        node_1_name = training_data_model.node_1[index]
        node_2_name = training_data_model.node_2[index]

        # Only connect nodes the model predicted are together and are not previously connected with any other nodes
        if diff[index] < 0 \
                or node_1_name not in training_data_model.dna_sequences \
                or node_2_name not in training_data_model.dna_sequences:
            continue

        tree.append([node_1_name, node_2_name])

        del (training_data_model.dna_sequences[node_1_name])
        del (training_data_model.dna_sequences[node_2_name])

    for unpaired_nodes in training_data_model.dna_sequences.keys():
        tree.append(unpaired_nodes)

    return tree


def print_tree(tree):
    print(tree)


if __name__ == "__main__":
    predict()
