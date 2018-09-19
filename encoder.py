import tensorflow as tf
import numpy as np

import load_data_utils
import tree_parser


tree_file = "dataset/20.2.tree"
dna_sequences_file = "dataset/seq_20.2.txt"
model_path = "./models/2500/"

encoder_hidden_size_1 = 100

feed_forward_hidden_units_1 = 1000
feed_forward_hidden_units_2 = 1000

encoder_hidden_size_1 = 100
encoder_hidden_size_2 = 100
encoder_output_size = 100

# encoder_hidden_units = 256
feed_forward_hidden_units_1 = 700
feed_forward_hidden_units_2 = 1000
feed_forward_hidden_units_3 = 1000

sequenceLength = 20

dnaNumLetters = 4

learning_rate = 0.02

batchSize = 100

max_gradient_norm = 1

numTrainingIters = 10000


def calculate_accuracy(y, sequence_length, predictions):
    entSequence = 0
    correct = 0
    for i in range(len(y)):
        numCorrect = 0
        for j in range(sequence_length):
            maxPos = -1
            maxVal = 0.0
            for k in range(dnaNumLetters):
                if maxVal < predictions[i][j][k]:
                    maxVal = predictions[i][j][k]
                    maxPos = k
            if maxPos == np.argmax(y[i][j]):
                numCorrect += 1
        if numCorrect == sequence_length:
            entSequence += 1
        correct += numCorrect

    return numCorrect/(batchSize*sequenceLength*dnaNumLetters)

def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def encode_sequence(sequence):
    enc_h1 = tf.nn.relu(tf.matmul(sequence, enc_w1) + enc_b1)

    enc_h2 = tf.nn.relu(tf.matmul(enc_h1, enc_w2) + enc_b2)

    enc_output = tf.matmul(enc_h2, enc_w3) + enc_b3

    return tf.nn.relu(enc_output)


def decode_sequence(sequence):
    dec_h1 = tf.nn.relu(tf.matmul(sequence, dec_w1) + dec_b1)

    dec_h2 = tf.nn.relu(tf.matmul(dec_h1, dec_w2) + dec_b2)

    dec_output = tf.matmul(dec_h2, dec_w3) +dec_b3

    return dec_output


tree = tree_parser.parse(tree_file)
max_size_dataset = tree.get_number_of_leaves()

data_input = tf.placeholder(tf.float32, [batchSize, sequenceLength, dnaNumLetters], name="encoder_dataset_plc")

enc_w1 = init_weights((dnaNumLetters, encoder_hidden_size_1))
enc_b1 = tf.Variable(np.zeros((1, encoder_hidden_size_1)), dtype=tf.float32)

enc_w2 = init_weights((encoder_hidden_size_1, encoder_hidden_size_2))
enc_b2 = tf.Variable(np.zeros((1, encoder_hidden_size_2)), dtype=tf.float32)

enc_w3 = init_weights((encoder_hidden_size_2, encoder_output_size))
enc_b3 = tf.Variable(np.zeros((1, encoder_output_size)), dtype=tf.float32)

# encoded_dna_sequence_1 = encode_sequence(dna_sequence_input_1)
# encoded_dna_sequence_2 = encode_sequence(dna_sequence_input_2)

encoded_dataset = tf.map_fn(lambda x: encode_sequence(x), data_input,
                            dtype=tf.float32)

dec_w1 = init_weights((encoder_output_size, encoder_hidden_size_2))
dec_b1 = tf.Variable(np.zeros((1, encoder_hidden_size_2)), dtype=tf.float32)

dec_w2 = init_weights((encoder_hidden_size_2, encoder_hidden_size_1))
dec_b2 = tf.Variable(np.zeros((1, encoder_hidden_size_2)), dtype=tf.float32)

dec_w3 = init_weights((encoder_hidden_size_1, dnaNumLetters))
dec_b3 = tf.Variable(np.zeros((1, dnaNumLetters)), dtype=tf.float32)

decoded_outputs = tf.map_fn(lambda x: decode_sequence(x), encoded_dataset,
                            dtype=tf.float32)
decoded_predictions = tf.nn.softmax(decoded_outputs)

losses = tf.nn.softmax_cross_entropy_with_logits(labels=data_input, logits=decoded_outputs)
total_loss = tf.reduce_mean(losses, name="loss")

training_alg = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = load_data_utils.read_data("dataset/seq_20.2.txt")

    for step in range(numTrainingIters):
        #
        # get some data
        dna_sequences = load_data_utils.get_dna_sequences(data, batchSize)
        #
        # do the training epoch
        _total_loss, _decoded_outputs, _decoded_predictions = sess.run(
            [total_loss, decoded_outputs, decoded_predictions],
            feed_dict={
                data_input: dna_sequences
                # decoder_lengths: [len(i) for i in dna_sequences]
            })

        accuracy = calculate_accuracy(dna_sequences, sequenceLength, _decoded_predictions)

        if step % 50 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, _total_loss, accuracy))
