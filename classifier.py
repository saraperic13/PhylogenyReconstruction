import tensorflow as tf

import load_data_utils
import tree_parser
import tree_utils

tree_file = "dataset/20.2.tree"
dna_sequence_file = "dataset/seq_20.2_"
model_file = './models/1/'


def write_to_file(losses, accuracy, avg_loss, avg_acc):
    with open("test/test_loss.txt", "a") as f:
        f.write("\n\n" + avg_loss)
        f.write(losses)

    with open("test/test_acc.txt", "a") as f:
        f.write("\n\n''" + avg_acc)
        f.write(accuracy)


tree = tree_parser.parse(tree_file)
max_size_dataset = tree.get_number_of_leaves()

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
    average_loss = 0
    average_acc = 0

    for i in range(1, 6):

        data = load_data_utils.read_data(dna_sequence_file + "{}.txt".format(str(i)))

        for step in range(100):
            dna_descendants, dna_child_1, dna_child_2, together = tree_utils.get_subroot_and_nodes(tree, data,
                                                                                                   batchSize=100,
                                                                                                   max_size_dataset=max_size_dataset)

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

        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accs) / len(accs)

        average_acc += avg_acc
        average_loss += avg_loss

        write_to_file("\n" + str(losses), "\n" + str(accs), str(avg_loss), str(avg_acc))
        losses.clear()
        accs.clear()

    print("Accuracy ", average_acc / 5)
    print("Loss", average_loss / 5)
