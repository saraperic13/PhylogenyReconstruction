import tensorflow as tf

from network_model.training_data_model import TrainingDataModel
from utils import load_data_utils

dna_sequence_file = "../dataset/seq_10.2.txt"
model_path = "../models/5000_20/"

sequence_length = 100
dna_num_letters = 4
batch_size = 50

max_number_of_leaves = 20


def predict():
    with tf.Session() as sess:
        load_model(sess)

        dna_subtree, dna_sequences_node_1, dna_sequences_node_2, predictions \
            = load_tensors()
        dna_sequences, num_of_trees_to_infer = load_dna_sequences()

        for tree_num in range(num_of_trees_to_infer):

            iteration_num = 0

            subtrees_to_infer = [dna_sequences.keys()]
            tree = []
            unpaired_nodes = {}

            while subtrees_to_infer and iteration_num < 1000:

                subtree = subtrees_to_infer.pop()
                subtree_dna_sequences = retrieve_dna_sequences_of_nodes(subtree, dna_sequences)

                training_data_model = TrainingDataModel(None, subtree_dna_sequences, sequence_length,
                                                        max_number_of_leaves,
                                                        dna_num_letters, dataset_index=tree_num)
                training_data_model = predict_tree(sess, training_data_model, dna_subtree,
                                                   dna_sequences_node_1,
                                                   dna_sequences_node_2,
                                                   predictions)

                build_tree(training_data_model, iteration_num, unpaired_nodes, tree, subtrees_to_infer)

                iteration_num += 1

            nodes = [node for sublist in unpaired_nodes.values() for node in sublist]
            tree.extend(nodes)
            print_tree(tree)


def build_tree(training_data_model, iteration_num, unpaired_nodes, tree, subtrees_to_infer):
    connected_subtree, colliding_nodes_sets_list, single_nodes = pair_nodes(training_data_model)

    if connected_subtree:

        if single_nodes:
            connected_subtree.extend(single_nodes)
            single_nodes.clear()

        if iteration_num - 1 in unpaired_nodes:
            node = unpaired_nodes[iteration_num - 1]
            del (unpaired_nodes[iteration_num - 1])
            node.append(connected_subtree)
            connected_subtree = node

        tree.extend(connected_subtree)

    print_tree(tree)

    if colliding_nodes_sets_list:
        subtrees_to_infer.extend(colliding_nodes_sets_list)

    if single_nodes:
        unpaired_nodes[iteration_num] = single_nodes


def load_dna_sequences():
    return load_data_utils.read_data(dna_sequence_file)


def load_model(session):
    tf.saved_model.loader.load(session, ["phylogeny_reconstruction"], model_path)


def load_tensors():
    graph = tf.get_default_graph()

    encoder_dataset_plc = graph.get_tensor_by_name("dna_subtree:0")
    encoder_dna_seq_1_plc = graph.get_tensor_by_name("dna_sequence_node_1:0")
    encoder_dna_seq_2_plc = graph.get_tensor_by_name("dna_sequence_node_2:0")

    predictions = graph.get_tensor_by_name("predictions:0")

    return encoder_dataset_plc, encoder_dna_seq_1_plc, encoder_dna_seq_2_plc, predictions


def retrieve_dna_sequences_of_nodes(nodes_list, dna_sequences):
    sequences = {}
    for node in nodes_list:
        sequences[node] = dna_sequences[node]
    return sequences


def predict_tree(session, training_data_model, dna_subtree, dna_sequence_node_1,
                 dna_sequence_node_2, prediction):
    nodes_together_predicted = tf.argmax(prediction, axis=1)

    training_data_model.prepare_node_pairs()
    _predictions, _nodes_together_predicted = session.run(
        [prediction, nodes_together_predicted],
        feed_dict={
            dna_subtree: training_data_model.descendants_dna_sequences,
            dna_sequence_node_1: training_data_model.dna_sequences_node_1,
            dna_sequence_node_2: training_data_model.dna_sequences_node_2,
        })
    print_prediction_node_pairs(_predictions, training_data_model.node_1, training_data_model.node_2)
    training_data_model.are_nodes_together = _predictions
    return training_data_model


def print_prediction_node_pairs(predictions, node_1, node_2):
    for i in range(len(node_1)):
        print('\t(', node_1[i], ', ', node_2[i], ')\t', predictions[i])


def pair_nodes(training_data_model):
    tree, single_nodes = [], []

    indexes = list(range(len(training_data_model.are_nodes_together)))

    # Calculate difference between predicted values, bigger the difference
    # more certain the model is about the prediction
    diff = [pair[0] - pair[1] for pair in training_data_model.are_nodes_together]
    abs_diff = [abs(i) for i in diff]

    # Sort absolute differences values in order to firstly connect nodes the model is certain about
    sorted_indexes = [x for _, x in sorted(zip(abs_diff, indexes), reverse=True)]

    colliding_node_names, colliding_indexes, colliding_nodes_sets_list = get_colliding_nodes(training_data_model, diff)
    sorted_indexes = remove_colliding_nodes_indexes(sorted_indexes, colliding_indexes)

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
        if unpaired_node in colliding_node_names:
            continue
        single_nodes.append(unpaired_node)

    return tree, colliding_nodes_sets_list, single_nodes


def get_colliding_nodes(training_data_model, prediction_differences):
    paired_node_indexes = {}
    paired_nodes = {}

    for indexes in range(len(prediction_differences)):

        if prediction_differences[indexes] <= 0:
            continue

        node_1_name = training_data_model.node_1[indexes]
        node_2_name = training_data_model.node_2[indexes]

        if node_1_name not in paired_node_indexes:
            paired_node_indexes[node_1_name] = []
            paired_nodes[node_1_name] = []

        paired_node_indexes[node_1_name].append(indexes)
        paired_nodes[node_1_name].append(node_2_name)

        if node_2_name not in paired_node_indexes:
            paired_node_indexes[node_2_name] = []
            paired_nodes[node_2_name] = []

        paired_node_indexes[node_2_name].append(indexes)
        paired_nodes[node_2_name].append(node_1_name)

    colliding_node_names = []
    colliding_indexes = set()

    for name, indexes in paired_node_indexes.items():

        if len(indexes) < 2 and not paired_node_colliding_with_others(name, indexes[0], paired_node_indexes):
            continue
        colliding_node_names.append(name)
        colliding_indexes.update(indexes)

    colliding_nodes_sets_list = form_colliding_nodes_sets(colliding_node_names, paired_nodes)

    return colliding_node_names, colliding_indexes, colliding_nodes_sets_list


def paired_node_colliding_with_others(node_name, index, paired_nodes):
    # Checks if node paired with suspicious node (node_name) is colliding with some other nodes
    # that way the initial node is colliding as well

    for name, indexes in paired_nodes.items():
        if node_name != name and len(indexes) > 1 and index in indexes:
            return True
    return False


def form_colliding_nodes_sets(colliding_node_names, paired_nodes):

    colliding_nodes_sets = []

    for node in colliding_node_names:
        if is_node_in_sets_list(node, colliding_nodes_sets):
            continue

        nodes_set = {node}
        colliding_nodes_sets.append(nodes_set)
        add_paired_nodes_to_set_recursively(node, paired_nodes, nodes_set, colliding_nodes_sets)

    return colliding_nodes_sets


def add_paired_nodes_to_set_recursively(node_name, paired_nodes, node_set, colliding_nodes_sets):
    paired_nodes_list = paired_nodes[node_name]
    for pair in paired_nodes_list:

        if is_node_in_sets_list(pair, colliding_nodes_sets):
            continue

        node_set.add(pair)
        add_paired_nodes_to_set_recursively(pair, paired_nodes, node_set, colliding_nodes_sets)


def is_node_in_sets_list(node, sets_list):

    for node_set in sets_list:
        if node in node_set:
            return True
    return False


def remove_colliding_nodes_indexes(indexes_list, colliding_indexes):
    for element in colliding_indexes:
        indexes_list.remove(element)
    return indexes_list


def print_tree(tree):
    print(tree)


if __name__ == "__main__":
    predict()
