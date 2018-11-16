import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from utils import load_data_utils
from tree_files import tree_parser, tree_utils
from training_data_model import TrainingDataModel

tree_file = "dataset/20.2.tree"
dna_sequences_files = "dataset/internal_1.2.txt"
model_path = "./subroot-prediction-models/10/"

encoder_output_size = 100

sequence_length = 100

dna_num_letters = 4

learning_rate = 0.02

batch_size = 100

numTrainingIters = 10


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def encode_sequence(sequence):
    enc_h1 = tf.nn.relu(tf.matmul(sequence, enc_w1) + enc_b1)
    return tf.nn.relu(enc_h1)


tree = tree_parser.parse(tree_file)[0]
max_size_dataset = tree.get_number_of_leaves()
feed_forward_hidden_units_1 = 50

data_input = tf.placeholder(tf.float32, [batch_size, max_size_dataset * sequence_length * dna_num_letters],
                            name="encoder_dataset_plc")

subroot_sequences = tf.placeholder(tf.float32, [batch_size, sequence_length * dna_num_letters], name="subroot_sequences")

enc_w1 = init_weights((max_size_dataset * sequence_length * dna_num_letters, encoder_output_size))
enc_b1 = tf.Variable(np.zeros((1, encoder_output_size)), dtype=tf.float32)

# encoded_dataset = tf.map_fn(lambda x: encode_sequence(x), data_input,
#                             dtype=tf.float32)
encoded_dataset = encode_sequence(data_input)

w1 = init_weights((encoder_output_size, feed_forward_hidden_units_1))
b1 = tf.Variable(np.zeros((1, feed_forward_hidden_units_1)), dtype=tf.float32)
h1 = tf.nn.relu(tf.matmul(encoded_dataset, w1) + b1)

idx = 0
for network in range(100):
    w = init_weights((feed_forward_hidden_units_1, dna_num_letters))
    b = tf.Variable(np.zeros((1, dna_num_letters)), dtype=tf.float32)
    output = tf.matmul(h1, w) + b

    predictions = tf.nn.softmax(output, name="predictions")

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=subroot_sequences[:, idx:idx + 4], logits=output)
    total_loss = tf.reduce_mean(losses, name="loss")

    training_alg = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    aa = tf.argmax(predictions, axis=1)
    bb = tf.argmax(subroot_sequences[:, idx:idx + 4], axis=1)
    correct_pred = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(subroot_sequences[:, idx:idx + 4], axis=1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    tf_accuracy = tf.metrics.accuracy(subroot_sequences[:, idx:idx + 4], predictions, name="tf_accuracy")

    idx += 4

builder = tf.saved_model.builder.SavedModelBuilder(model_path)

signature = predict_signature_def(
    inputs={'encoder_dataset_plc': encoded_dataset, 'subroot_sequences': subroot_sequences},
    outputs={'predictions': predictions})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    data = load_data_utils.read_data(dna_sequences_files)

    for step in range(numTrainingIters + 1):

        training_data_model = TrainingDataModel(tree, data, sequence_length,
                                                dna_num_letters)

        tree_utils.get_batch_sized_data(batch_size, training_data_model)

        _encoded_dataset, _totalLoss, _training_alg, _predictions, _accuracy, a, b, _tf_accuracy = sess.run(
            [encoded_dataset, total_loss, training_alg, predictions, accuracy, aa, bb, tf_accuracy],
            feed_dict={
                data_input: training_data_model.descendants_dna_sequences,
                subroot_sequences: training_data_model.dna_subroots
            })

        if step % 1 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, _totalLoss, _accuracy))

    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=["subroot_prediction"],
                                         signature_def_map={'predict': signature})

builder.save()
