import tensorflow as tf

import load_data_utils
import tree_parser
import tree_utils

tree_file = "dataset/really_small.tree"
dna_sequence_file = "dataset/really_small_"
model_file = './models/mali/'


def write_to_file(losses, accuracy):
    with open("test_loss.txt", "a") as f:
        f.write(losses)

    with open("test_acc.txt", "a") as f:
        f.write(accuracy)


tree = tree_parser.parse(tree_file)

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

    for i in range(1, 6):

        data = load_data_utils.read_data(dna_sequence_file + "{}.txt".format(str(i)))

        for step in range(100):
            dna_descendants, dna_child_1, dna_child_2, together = tree_utils.get_subroot_and_nodes(tree, data,
                                                                                                   batchSize=100)

            _accuracy, _loss = sess.run(
                [accuracy, loss],
                feed_dict={
                    encoder_dataset_plc: dna_descendants,
                    encoder_dna_seq_1_plc: dna_child_1,
                    encoder_dna_seq_2_plc: dna_child_2,
                    together_plc: together
                })

            losses.append(_loss)
            accs.append(_accuracy)

            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, _loss, _accuracy))

        write_to_file("\n" + str(losses), "\n" + str(accs))
        losses.clear()
        accs.clear()
