import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import load_data_utils
import tree_parser
import tree_utils

hiddenUnits = 50

sequenceLength = 20

dnaNumLetters = 4

learningRate = 0.05

batchSize = 100

numTrainingIters = 1000


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


tree = tree_parser.parse('dataset/phylogeny.tree')

data_input = tf.placeholder(tf.float32, [batchSize, None, sequenceLength, dnaNumLetters], name="encoder_dataset_plc")
dna_sequence_input_1 = tf.placeholder(tf.float32, [batchSize, sequenceLength, dnaNumLetters],
                                      name="encoder_dna_seq_1_plc")
dna_sequence_input_2 = tf.placeholder(tf.float32, [batchSize, sequenceLength, dnaNumLetters],
                                      name="encoder_dna_seq_2_plc")
inputY = tf.placeholder(tf.float32, [batchSize, 1], name="together_plc")

# Encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenUnits)

encoded_dataset = tf.map_fn(lambda x: tf.nn.dynamic_rnn(cell=encoder_cell,
                                                        inputs=x,
                                                        dtype=tf.float32, time_major=False)[0][:, -1, :], data_input,
                            dtype=tf.float32)

encoded_dataset = tf.map_fn(lambda x: tf.reduce_mean(x, axis=0), encoded_dataset,
                            dtype=tf.float32)

encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
                                       inputs=dna_sequence_input_1, dtype=tf.float32)
encoded_dna_sequence_1 = encoder_outputs[:, -1, :]

encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
                                       inputs=dna_sequence_input_2, dtype=tf.float32)
encoded_dna_sequence_2 = encoder_outputs[:, -1, :]

# Classifier

feed_forward_inputX = tf.concat([encoded_dataset, encoded_dna_sequence_1, encoded_dna_sequence_2], 1)

w_1 = init_weights((3 * hiddenUnits, hiddenUnits))
b1 = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

w_2 = init_weights((hiddenUnits, 1))
b2 = tf.Variable(np.zeros((1, 1)), dtype=tf.float32)

h = tf.nn.tanh(tf.matmul(feed_forward_inputX, w_1) + b1)
output = tf.matmul(h, w_2) + b2

predictions = tf.nn.sigmoid(output, name="predictions")

losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputY, logits=output)
total_loss = tf.reduce_mean(losses, name="loss")

training_alg = tf.train.AdagradOptimizer(0.02).minimize(total_loss)

correct_pred = tf.equal(tf.round(predictions), inputY)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel50/')

signature = predict_signature_def(
    inputs={'encoder_dataset_plc': encoded_dataset, 'encoder_dna_seq_1_plc': dna_sequence_input_1,
            'encoder_dna_seq_2_plc': dna_sequence_input_2, 'together_plc':inputY},
    outputs={'predictions': predictions})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = load_data_utils.read_data("dataset/seq_20.txt")

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
                                         tags=["myTag"],
                                         signature_def_map={'predict': signature})

builder.save()
