import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import load_data_utils
import tree_parser
import tree_utils

tree_file = "dataset/20.2.tree"
dna_sequences_files = "dataset/internal_100.2.txt"
model_path = "./subroot-prediction-models/100/"

encoder_output_size = 100

sequenceLength = 100

feed_forward_hidden_units_2 = 1000
feed_forward_hidden_units_3 = 1000

dnaNumLetters = 4

learning_rate = 0.02

batchSize = 100

numTrainingIters = 1000


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def encode_sequence(sequence):
    enc_h1 = tf.nn.relu(tf.matmul(sequence, enc_w1) + enc_b1)
    return tf.nn.relu(enc_h1)


tree = tree_parser.parse(tree_file)
max_size_dataset = tree.get_number_of_leaves()
feed_forward_hidden_units_1 = 50

data_input = tf.placeholder(tf.float32, [batchSize, max_size_dataset * sequenceLength * dnaNumLetters],
                            name="encoder_dataset_plc")

subroot_sequences = tf.placeholder(tf.float32, [batchSize, sequenceLength * dnaNumLetters], name="subroot_sequences")

enc_w1 = init_weights((max_size_dataset * sequenceLength * dnaNumLetters, encoder_output_size))
enc_b1 = tf.Variable(np.zeros((1, encoder_output_size)), dtype=tf.float32)

# encoded_dataset = tf.map_fn(lambda x: encode_sequence(x), data_input,
#                             dtype=tf.float32)
encoded_dataset = encode_sequence(data_input)

w1 = init_weights((encoder_output_size, feed_forward_hidden_units_1))
b1 = tf.Variable(np.zeros((1, feed_forward_hidden_units_1)), dtype=tf.float32)
h1 = tf.nn.relu(tf.matmul(encoded_dataset, w1) + b1)

idx = 0
for network in range(100):
    w = init_weights((feed_forward_hidden_units_1, dnaNumLetters))
    b = tf.Variable(np.zeros((1, dnaNumLetters)), dtype=tf.float32)
    output = tf.matmul(h1, w) + b

    predictions = tf.nn.softmax(output, name="predictions")

    losses = tf.nn.softmax_cross_entropy_with_logits(labels=subroot_sequences[:, idx:idx + 4], logits=output)
    total_loss = tf.reduce_mean(losses, name="loss")

    training_alg = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    correct_pred = tf.equal(tf.round(predictions), subroot_sequences[:, idx:idx + 4])
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
    idx += 4

builder = tf.saved_model.builder.SavedModelBuilder(model_path)

signature = predict_signature_def(
    inputs={'encoder_dataset_plc': encoded_dataset, 'subroot_sequences': subroot_sequences},
    outputs={'predictions': predictions})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = load_data_utils.read_data(dna_sequences_files)

    for step in range(numTrainingIters + 1):

        subroots, dna_descendants, _, _, _ = tree_utils.get_subroot_and_nodes(tree, data, batchSize,
                                                                              max_size_dataset,
                                                                              sequence_length=sequenceLength)

        _encoded_dataset, _totalLoss, _training_alg, _predictions, _accuracy = sess.run(
            [encoded_dataset, total_loss, training_alg, predictions, accuracy],
            feed_dict={
                data_input: dna_descendants,
                subroot_sequences: subroots
            })
        if step % 50 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, _totalLoss, _accuracy))

    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=["subroot_prediction"],
                                         signature_def_map={'predict': signature})

builder.save()
