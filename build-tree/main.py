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

        max_number_of_leaves = len(dna_sequences.keys())

        for i in range(num_of_trees_to_infer):

            subtrees_to_infer = [dna_sequences.keys()]
            tree = []
            unpaired_nodes = []

            while subtrees_to_infer:

                subtree = subtrees_to_infer.pop()
                subtree_dna_sequences = retrieve_dna_sequences_of_nodes(subtree, dna_sequences)

                training_data_model = TrainingDataModel(None, subtree_dna_sequences, sequence_length,
                                                        dna_num_letters, dataset_index=i)
                training_data_model = predict_tree(sess, training_data_model, dna_subtree,
                                                   dna_sequences_node_1,
                                                   dna_sequences_node_2, are_nodes_together, number_of_leaves,
                                                   predictions, max_number_of_leaves)

                connected_subtree, colliding_nodes, single_nodes = build_tree(training_data_model)

                if connected_subtree:

                    if unpaired_nodes:
                        node = unpaired_nodes.pop()
                        connected_subtree = [connected_subtree, node]

                    tree.extend(connected_subtree)

                print_tree(tree)

                if colliding_nodes:
                    subtrees_to_infer.append(colliding_nodes)

                if single_nodes:
                    unpaired_nodes.extend(single_nodes)

            tree.extend(unpaired_nodes)
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


def retrieve_dna_sequences_of_nodes(nodes_list, dna_sequences):
    sequences = {}
    for node in nodes_list:
        sequences[node] = dna_sequences[node]
    return sequences


def predict_tree(session, training_data_model, dna_subtree, dna_sequence_node_1,
                 dna_sequence_node_2,
                 are_nodes_together, number_of_leaves, prediction, max_number_of_leaves):
    nodes_together_predicted = tf.argmax(prediction, axis=1)

    training_data_model.prepare_node_pairs()
    _predictions, _nodes_together_predicted = session.run(
        [prediction, nodes_together_predicted],
        feed_dict={
            dna_subtree: training_data_model.descendants_dna_sequences,
            dna_sequence_node_1: training_data_model.dna_sequences_node_1,
            dna_sequence_node_2: training_data_model.dna_sequences_node_2,
            are_nodes_together: training_data_model.are_nodes_together,
            # number_of_leaves: len(training_data_model.dna_sequences.keys())
            number_of_leaves: max_number_of_leaves
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

    colliding_node_names, colliding_indexes = get_colliding_nodes(training_data_model, diff)
    sorted_indexes = remove_colliding_nodes_indexes(sorted_indexes, colliding_indexes)

    tree, single_nodes = [], []

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

    for unpaired_node in training_data_model.dna_sequences.keys():
        single_nodes.append(unpaired_node)

    return tree, colliding_node_names, single_nodes


def get_colliding_nodes(training_data_model, prediction_differences):
    paired_nodes = {}

    for index in range(len(prediction_differences)):

        if prediction_differences[index] <= 0:
            continue

        node_1_name = training_data_model.node_1[index]
        node_2_name = training_data_model.node_2[index]

        if node_1_name not in paired_nodes:
            paired_nodes[node_1_name] = []

        paired_nodes[node_1_name].append(index)

        if node_2_name not in paired_nodes:
            paired_nodes[node_2_name] = []

        paired_nodes[node_2_name].append(index)

    colliding_node_names = []
    colliding_indexes = set()

    for name, index in paired_nodes.items():
        if len(index) < 2:
            continue
        colliding_node_names.append(name)
        colliding_indexes.update(index)

    return colliding_node_names, colliding_indexes


def remove_colliding_nodes_indexes(indexes_list, colliding_indexes):
    for element in colliding_indexes:
        indexes_list.remove(element)
    return indexes_list


def print_tree(tree):
    print(tree)


if __name__ == "__main__":
    predict()
