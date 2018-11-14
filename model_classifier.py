import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import load_data_utils
import tree_parser
import tree_utils
from classfier_network import ClassifierNetwork
from encoder_network import EncoderNetwork
from training_data_model import TrainingDataModel

tree_file = "dataset/100-trees/probaj_jedno.tree"
dna_sequence_file = "dataset/100-trees/jedno.txt"
model_path = "./models/nmn/"

encoder_output_size = 70

feed_forward_hidden_units_1 = 50
feed_forward_hidden_units_2 = 100
feed_forward_hidden_units_3 = 100

sequence_length = 100

dna_num_letters = 4

learning_rate = 0.02

batch_size = 100

num_training_iters = 1000

trees = tree_parser.parse(tree_file)
max_size_dataset = trees[0].get_number_of_leaves()

encoder = EncoderNetwork([sequence_length * dna_num_letters, encoder_output_size], batch_size, sequence_length,
                         dna_num_letters)
encoded_dataset, encoded_dna_sequence_1, encoded_dna_sequence_2 = encoder.encode()

# Classifier

classifier_input_data = tf.concat([encoded_dataset, encoded_dna_sequence_1, encoded_dna_sequence_2], 1)

classifier_network = ClassifierNetwork(
    [encoder_output_size * 3, feed_forward_hidden_units_1, feed_forward_hidden_units_2, feed_forward_hidden_units_3, 2],
    batch_size, sequence_length, dna_num_letters, classifier_input_data)
output, predictions = classifier_network.predict()

losses = tf.nn.softmax_cross_entropy_with_logits(labels=encoder.are_nodes_together, logits=output)
total_loss = tf.reduce_mean(losses, name="loss")

training_alg = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

correct_pred = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(encoder.are_nodes_together, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

builder = tf.saved_model.builder.SavedModelBuilder(model_path)

signature = predict_signature_def(
    inputs={'dna_subtree': encoder.dna_subtree, 'dna_sequence_node_1': encoder.dna_sequence_node_1,
            'dna_sequence_node_2': encoder.dna_sequence_node_2, 'are_nodes_together': encoder.are_nodes_together},
    outputs={'predictions': predictions})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    data = load_data_utils.read_data(dna_sequence_file)

    for step in range(num_training_iters + 1):

        i = 0

        for tree in trees:

            training_data_model = TrainingDataModel(tree, data, sequence_length,
                                                    dna_num_letters, dataset_index=i)

            tree_utils.get_batch_sized_data(batch_size, training_data_model)

            _totalLoss, _training_alg, _predictions, _accuracy = sess.run(
                [total_loss, training_alg, predictions, accuracy],
                feed_dict={
                    encoder.dna_subtree: training_data_model.descendants_dna_sequences,
                    encoder.dna_sequence_node_1: training_data_model.dna_sequences_node_1,
                    encoder.dna_sequence_node_2: training_data_model.dna_sequences_node_2,
                    encoder.are_nodes_together: training_data_model.are_nodes_together
                })
            if step % 1 == 0:
                print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                    step, _totalLoss, _accuracy))

            i += 1

    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=["phylogeny_reconstruction"],
                                         signature_def_map={'predict': signature})

builder.save()
