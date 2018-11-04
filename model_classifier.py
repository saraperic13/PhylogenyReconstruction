import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import load_data_utils
import tree_parser
import tree_utils

tree_file = "dataset/100-trees/100_20.2.tree"
dna_sequences_files = "dataset/100-trees/seq_100_20.2.txt"
model_path = "./models/trees/"

encoder_output_size = 70

feed_forward_hidden_units_1 = 700
feed_forward_hidden_units_2 = 1000
feed_forward_hidden_units_3 = 1000

sequenceLength = 100

dnaNumLetters = 4

learning_rate = 0.02

batchSize = 100

numTrainingIters = 500


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def encode_sequence(sequence):
    enc_h1 = tf.matmul(sequence, enc_w1) + enc_b1
    return tf.nn.relu(enc_h1)


trees = tree_parser.parse(tree_file)
max_size_dataset = trees[0].get_number_of_leaves()

data_input = tf.placeholder(tf.float32, [batchSize, None, sequenceLength * dnaNumLetters], name="encoder_dataset_plc")
dna_sequence_input_1 = tf.placeholder(tf.float32, [batchSize, sequenceLength * dnaNumLetters],
                                      name="encoder_dna_seq_1_plc")
dna_sequence_input_2 = tf.placeholder(tf.float32, [batchSize, sequenceLength * dnaNumLetters],
                                      name="encoder_dna_seq_2_plc")
inputY = tf.placeholder(tf.float32, [batchSize, 2], name="together_plc")

enc_w1 = init_weights((sequenceLength * dnaNumLetters, encoder_output_size))
enc_b1 = tf.Variable(np.zeros((1, encoder_output_size)), dtype=tf.float32)

encoded_dna_sequence_1 = encode_sequence(dna_sequence_input_1)
encoded_dna_sequence_2 = encode_sequence(dna_sequence_input_2)

encoded_dataset = tf.map_fn(lambda x: encode_sequence(x), data_input,
                            dtype=tf.float32)

encoded_dataset = tf.map_fn(lambda x: tf.reduce_mean(x, axis=0), encoded_dataset,
                            dtype=tf.float32)

# Classifier

feed_forward_inputX = tf.concat([encoded_dataset, encoded_dna_sequence_1, encoded_dna_sequence_2], 1)

w1 = init_weights((3 * encoder_output_size, feed_forward_hidden_units_1))
b1 = tf.Variable(np.zeros((1, feed_forward_hidden_units_1)), dtype=tf.float32)
h1 = tf.nn.relu(tf.matmul(feed_forward_inputX, w1) + b1)

w2 = init_weights((feed_forward_hidden_units_1, feed_forward_hidden_units_2))
b2 = tf.Variable(np.zeros((1, feed_forward_hidden_units_2)), dtype=tf.float32)
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

w3 = init_weights((feed_forward_hidden_units_2, feed_forward_hidden_units_3))
b3 = tf.Variable(np.zeros((1, feed_forward_hidden_units_3)), dtype=tf.float32)
h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

w4 = init_weights((feed_forward_hidden_units_3, 2))
b4 = tf.Variable(np.zeros((1, 2)), dtype=tf.float32)

output = tf.matmul(h3, w4) + b4

predictions = tf.nn.softmax(output, name="predictions")

losses = tf.nn.softmax_cross_entropy_with_logits(labels=inputY, logits=output)
total_loss = tf.reduce_mean(losses, name="loss")

training_alg = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

correct_pred = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(inputY, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

builder = tf.saved_model.builder.SavedModelBuilder(model_path)

signature = predict_signature_def(
    inputs={'encoder_dataset_plc': encoded_dataset, 'encoder_dna_seq_1_plc': dna_sequence_input_1,
            'encoder_dna_seq_2_plc': dna_sequence_input_2, 'together_plc': inputY},
    outputs={'predictions': predictions})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    data = load_data_utils.read_data(dna_sequences_files)

    i = 0
    for tree in trees:

        for step in range(numTrainingIters + 1):

            subroots, dna_descendants, dna_child_1, dna_child_2, together = tree_utils.get_subroot_and_nodes(tree, data,
                                                                                                             batchSize,
                                                                                                             max_size_dataset,
                                                                                                             sequence_length=sequenceLength,
                                                                                                             dataset_index=i)

            _encoded_dataset, _totalLoss, _training_alg, _predictions, _accuracy = sess.run(
                [encoded_dataset, total_loss, training_alg, predictions, accuracy],
                feed_dict={
                    data_input: dna_descendants,
                    dna_sequence_input_1: dna_child_1,
                    dna_sequence_input_2: dna_child_2,
                    inputY: together
                })
            if step % 50 == 0:
                print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                    step, _totalLoss, _accuracy))

        print(tree.randomly_selected)
        i += 1

    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=["phylogeny_reconstruction"],
                                         signature_def_map={'predict': signature})

builder.save()
