import tensorflow as tf

from network_model.training_data_model import TrainingDataModel
from utils import load_data_utils
from build_tree import minicut

dna_sequence_file = "../dataset/seq_10.2.txt"
model_path = "../models/10_root/"

sequence_length = 100
dna_num_letters = 4

max_number_of_leaves = 20


def predict():
    with tf.Session() as sess:
        load_model(sess)

        dna_subtree, dna_sequences_node_1, dna_sequences_node_2, predictions \
            = load_tensors()
        dna_sequences, num_of_trees_to_infer, node_names = load_dna_sequences()

        for tree_num in range(num_of_trees_to_infer):
            tree = []

            build_tree(node_names, tree, dna_sequences, tree_num, sess, dna_subtree, dna_sequences_node_1,
                       dna_sequences_node_2, predictions)

            print_tree(tree)


def build_tree(nodes_names, tree, dna_sequences, tree_num, session, dna_subtree, dna_sequences_node_1,
               dna_sequences_node_2, predictions):

    unconnected_nodes, stack = [], []

    training_data_model = create_training_data_model(nodes_names, dna_sequences, tree_num, session, dna_subtree,
                                                     dna_sequences_node_1,
                                                     dna_sequences_node_2, predictions, node_names=nodes_names)

    connected_subtree, single_nodes, subtree_to_infer = create_subgraphs(training_data_model)
    unconnected_nodes.extend(single_nodes)

    if connected_subtree:
        while unconnected_nodes:
            node = unconnected_nodes.pop()
            connected_subtree.extend(node)
        stack.extend(connected_subtree)
        # return stack
    if subtree_to_infer:

        for nodes in subtree_to_infer:
            stack.extend(build_tree(nodes, tree, dna_sequences, tree_num, session, dna_subtree, dna_sequences_node_1,
                       dna_sequences_node_2, predictions))

        while unconnected_nodes:
            if stack:
                node = unconnected_nodes.pop()
                node.append(stack)
                stack = []
                tree.append(node)
            else:
                tree.append(unconnected_nodes.pop())

        if stack:
            tree.append(stack)
            node = tree.copy()
            tree.clear()
            tree.append(list(node))
            stack = []
    return stack


def create_training_data_model(nodes, dna_sequences, tree_num, session, dna_subtree, dna_sequences_node_1,
                               dna_sequences_node_2, predictions, node_names):
    subtree_dna_sequences = retrieve_dna_sequences_of_nodes(nodes, dna_sequences)

    training_data_model = TrainingDataModel(None, subtree_dna_sequences, sequence_length,
                                            max_number_of_leaves,
                                            dna_num_letters, dataset_index=tree_num, nodes_names=node_names)
    training_data_model = predict_tree(session, training_data_model, dna_subtree,
                                       dna_sequences_node_1,
                                       dna_sequences_node_2,
                                       predictions)

    return training_data_model


def create_subgraphs(training_data_model):
    connected_subtree, single_nodes, subtree_to_infer = [], [], []
    partition = minicut.get_partition(training_data_model.node_1, training_data_model.node_2,
                                      training_data_model.are_nodes_together)

    for node_pairs in partition:
        if len(node_pairs) == 1:
            single_nodes.append(node_pairs)
        elif len(node_pairs) == 2:
            connected_subtree.append(node_pairs)
        else:
            subtree_to_infer.append(node_pairs)

    # connected_subtree = [node_pairs for node_pairs in partition if len(node_pairs) <= 2]
    return connected_subtree, single_nodes, subtree_to_infer


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

    print("\n******************************************************************\n")


def print_tree(tree):
    print(tree)


if __name__ == "__main__":
    predict()
