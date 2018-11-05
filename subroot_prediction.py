import tensorflow as tf

import load_data_utils
import tree_parser
import tree_utils
from training_data_model import TrainingDataModel

tree_file = "dataset/20.2.tree"
dna_sequence_file = "dataset/internal_1.2.txt"
model_file = './subroot-prediction-models/10/'

sequence_length = 100
batch_size = 100
dna_num_letters = 4

tree = tree_parser.parse(tree_file)[0]

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ["subroot_prediction"], model_file)

    graph = tf.get_default_graph()
    print(graph.get_operations())

    encoder_dataset_plc = graph.get_tensor_by_name("encoder_dataset_plc:0")
    subroot_sequences = graph.get_tensor_by_name("subroot_sequences:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    loss = graph.get_tensor_by_name("loss:0")

    losses, accs = [], []

    data = load_data_utils.read_data(dna_sequence_file)

    for step in range(100):
        training_data_model = TrainingDataModel(tree, data, sequence_length,
                                                dna_num_letters)

        tree_utils.get_batch_sized_data(batch_size, training_data_model)

        _accuracy, _loss = sess.run(
            [accuracy, loss],
            feed_dict={
                encoder_dataset_plc: training_data_model.descendants_dna_sequences,
                subroot_sequences: training_data_model.dna_subroots
            })

        losses.append(_loss)
        accs.append(_accuracy)

        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
            step, _loss, _accuracy))

    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accs) / len(accs)

    # write_to_file("\n" + str(losses), "\n" + str(accs), str(avg_loss), str(avg_acc))
    losses.clear()
    accs.clear()

    print("Accuracy ", avg_acc)
    print("Loss", avg_loss)
