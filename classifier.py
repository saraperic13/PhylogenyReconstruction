import tensorflow as tf

import load_data_utils
import tree_parser
import tree_utils
from training_data_model import TrainingDataModel

tree_file = "dataset/100-trees/50_20.2.tree"
dna_sequence_file = "dataset/100-trees/seq_50_20.2.txt"
model_file = './models/2update1/'

sequence_length = 100
batch_size = 100

dna_num_letters = 4


def write_to_file(losses, accuracy, avg_loss, avg_acc):
    with open("test/test_loss.txt", "a") as f:
        f.write("\n\n" + avg_loss)
        f.write(losses)

    with open("test/test_acc.txt", "a") as f:
        f.write("\n\n''" + avg_acc)
        f.write(accuracy)


trees = tree_parser.parse(tree_file)
max_size_dataset = trees[0].get_number_of_leaves()

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ["phylogeny_reconstruction"], model_file)

    graph = tf.get_default_graph()
    print(graph.get_operations())

    encoder_dataset_plc = graph.get_tensor_by_name("encoder_dataset_plc:0")
    encoder_dna_seq_1_plc = graph.get_tensor_by_name("encoder_dna_seq_1_plc:0")
    encoder_dna_seq_2_plc = graph.get_tensor_by_name("encoder_dna_seq_2_plc:0")
    together_plc = graph.get_tensor_by_name("together_plc:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    loss = graph.get_tensor_by_name("loss:0")

    losses, accs = [], []

    data = load_data_utils.read_data(dna_sequence_file)

    i = 0
    for tree in trees:

        for step in range(100):
            training_data_model = TrainingDataModel(tree, data, sequence_length, i,
                                                    dna_num_letters)

            tree_utils.get_batch_sized_data(batch_size, training_data_model)
            _accuracy, _loss = sess.run(
                [accuracy, loss],
                feed_dict={
                    encoder_dataset_plc: training_data_model.descendants_dna_sequences,
                    encoder_dna_seq_1_plc: training_data_model.dna_sequences_left_child,
                    encoder_dna_seq_2_plc: training_data_model.dna_sequences_right_child,
                    together_plc: training_data_model.are_nodes_together
                })

            losses.append(_loss)
            accs.append(_accuracy)

            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, _loss, _accuracy))
        i += 1

    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)

    write_to_file("\n" + str(losses), "\n" + str(accs), str(avg_loss), str(avg_acc))
    losses.clear()
    accs.clear()

    print("Accuracy ", avg_acc)
    print("Loss", avg_loss)
