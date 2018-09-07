import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import load_data_utils
import tree_parser
import tree_utils

tree_file = "dataset/20.2.tree"
dna_sequences_file = "dataset/seq_20.2.txt"
model_path = "./models/1000a/"

encoder_hidden_size_1 = 100
encoder_hidden_size_2 = 100
encoder_output_size = 100

# encoder_hidden_units = 256
feed_forward_hidden_units_1 = 700
feed_forward_hidden_units_2 = 1000
feed_forward_hidden_units_3 = 1000

sequenceLength = 20

dnaNumLetters = 4

learning_rate = 0.002

batchSize = 100

numTrainingIters = 1000


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def encode_sequence(sequence):
    enc_h1 = tf.nn.relu(tf.matmul(sequence, enc_w1) + enc_b1)

    enc_h2 = tf.nn.relu(tf.matmul(enc_h1, enc_w2) + enc_b2)

    enc_output = tf.matmul(enc_h2, enc_w3) + enc_b3

    return tf.nn.relu(enc_output)


tree = tree_parser.parse(tree_file)

data_input = tf.placeholder(tf.float32, [batchSize, None, sequenceLength * dnaNumLetters], name="encoder_dataset_plc")
dna_sequence_input_1 = tf.placeholder(tf.float32, [batchSize, sequenceLength * dnaNumLetters],
                                      name="encoder_dna_seq_1_plc")
dna_sequence_input_2 = tf.placeholder(tf.float32, [batchSize, sequenceLength * dnaNumLetters],
                                      name="encoder_dna_seq_2_plc")
inputY = tf.placeholder(tf.float32, [batchSize, 2], name="together_plc")

# Encoder
# encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_size_1)

# rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [encoder_hidden_size_1, encoder_hidden_units]]

# create a RNN cell composed sequentially of a number of RNNCells
# encoder_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

#
# encoded_dataset = tf.map_fn(lambda x: tf.nn.dynamic_rnn(cell=encoder_cell,
#                                                         inputs=x,
#                                                         dtype=tf.float32, time_major=False)[0][:, -1, :], data_input,
#                             dtype=tf.float32)
#
# encoded_dataset = tf.map_fn(lambda x: tf.reduce_mean(x, axis=0), encoded_dataset,
#                             dtype=tf.float32)
#
# encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
#                                        inputs=dna_sequence_input_1, dtype=tf.float32)
# encoded_dna_sequence_1 = encoder_outputs[:, -1, :]
#
# encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
#                                        inputs=dna_sequence_input_2, dtype=tf.float32)
# encoded_dna_sequence_2 = encoder_outputs[:, -1, :]


enc_w1 = init_weights((sequenceLength * dnaNumLetters, encoder_hidden_size_1))
enc_b1 = tf.Variable(np.zeros((1, encoder_hidden_size_1)), dtype=tf.float32)

enc_w2 = init_weights((encoder_hidden_size_1, encoder_hidden_size_2))
enc_b2 = tf.Variable(np.zeros((1, encoder_hidden_size_2)), dtype=tf.float32)

enc_w3 = init_weights((encoder_hidden_size_2, encoder_output_size))
enc_b3 = tf.Variable(np.zeros((1, encoder_output_size)), dtype=tf.float32)

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

correct_pred = tf.equal(tf.round(predictions), inputY)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

builder = tf.saved_model.builder.SavedModelBuilder(model_path)

signature = predict_signature_def(
    inputs={'encoder_dataset_plc': encoded_dataset, 'encoder_dna_seq_1_plc': dna_sequence_input_1,
            'encoder_dna_seq_2_plc': dna_sequence_input_2, 'together_plc': inputY},
    outputs={'predictions': predictions})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = load_data_utils.read_data(dna_sequences_file)

    for step in range(numTrainingIters + 1):

        dna_descendants, dna_child_1, dna_child_2, together = tree_utils.get_subroot_and_nodes(tree, data, batchSize)

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

    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=["phylogeny_reconstruction"],
                                         signature_def_map={'predict': signature})

builder.save()
