import tensorflow as tf

from network_model.training_data_model import TrainingDataModel
from tree_files import tree_parser, tree_utils
from utils import file_utils
from utils import load_data_utils

tree_file = "dataset/100-trees/50_20.2.tree"
dna_sequence_file = "dataset/100-trees/seq_50_20.2.txt"
model_path = "models/mean/"

sequence_length = 100
batch_size = 100

dna_num_letters = 4
number_of_iterations = 10


def run_session():
    with tf.Session() as sess:
        load_model(model_path, "phylogeny_reconstruction", sess)

        dna_subtree, dna_sequences_node_1, dna_sequences_node_2, \
        are_nodes_together, accuracy, loss, predictions = load_tensors()

        losses, accuracies = [], []

        dna_sequences = load_dna_sequences(dna_sequence_file)
        trees = load_trees(tree_file)

        dataset_index = 0
        for i in range(0, len(trees)):
            predict_over_tree(sess, trees[i], i, number_of_iterations, dna_sequences, dna_subtree,
                              dna_sequences_node_1,
                              dna_sequences_node_2, are_nodes_together, accuracy, loss, predictions, losses, accuracies)

            dataset_index += 1

        calculate_accuracy_and_loss_and_write_report(accuracies, losses)


def load_trees(file):
    return tree_parser.parse(file)


def load_dna_sequences(file):
    return load_data_utils.read_data(file)


def load_model(file_path, tag_name, session):
    tf.saved_model.loader.load(session, [tag_name], file_path)


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


def predict_over_tree(session, tree, tree_index, number_of_iterations, dna_sequences, dna_subtree, dna_sequence_node_1,
                      dna_sequence_node_2,
                      are_nodes_together, accuracy, loss, prediction, losses, accuracies):
    for step in range(number_of_iterations):
        training_data_model = TrainingDataModel(tree, dna_sequences, sequence_length,
                                                dna_num_letters, dataset_index=tree_index)

        tree_utils.get_batch_sized_data(batch_size, training_data_model)
        _accuracy, _loss, _predictions = session.run(
            [accuracy, loss, prediction],
            feed_dict={
                dna_subtree: training_data_model.descendants_dna_sequences,
                dna_sequence_node_1: training_data_model.dna_sequences_node_1,
                dna_sequence_node_2: training_data_model.dna_sequences_node_2,
                are_nodes_together: training_data_model.are_nodes_together
            })

        losses.append(_loss)
        accuracies.append(_accuracy)

        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
            step, _loss, _accuracy))


def calculate_accuracy_and_loss_and_write_report(accuracies, losses):
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accuracies) / len(accuracies)

    file_utils.write_to_file("\n" + str(losses), "\n" + str(accuracies), str(avg_loss), str(avg_acc), path="test/")

    print("Accuracy ", avg_acc)
    print("Loss", avg_loss)


run_session()
