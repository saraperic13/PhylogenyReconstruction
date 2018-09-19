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




tree = tree_parser.parse(tree_file)
max_size_dataset = tree.get_number_of_leaves()

data_input = tf.placeholder(tf.float32, [batchSize, sequenceLength, dnaNumLetters], name="encoder_dataset_plc")

# Encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_size_1)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                   inputs=data_input, dtype=tf.float32, time_major=False)

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_size_1)

# Helper
decoder_lengths = tf.placeholder(dtype=tf.int32, shape=[None])

helper = tf.contrib.seq2seq.TrainingHelper(
    data_input, decoder_lengths, time_major=False)

projection_layer = tf.layers.Dense(dnaNumLetters, use_bias=True)

# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, initial_state=encoder_state,
    output_layer=projection_layer)

# Dynamic decoding
decode_outputs = tf.contrib.seq2seq.dynamic_decode(decoder)
outputs = decode_outputs[0].rnn_output

predictions = tf.nn.softmax(outputs)

losses = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=data_input, logits=outputs)

totalLoss = tf.reduce_mean(losses)

# Calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(totalLoss, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient_norm)

# Optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_step = optimizer.apply_gradients(
    zip(clipped_gradients, params))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = load_data_utils.read_data("dataset/seq_20.2.txt")

    for step in range(numTrainingIters):
        #
        # get some data
        dna_sequences = load_data_utils.get_dna_sequences(data, batchSize)
        #
        # do the training epoch
        _totalLoss, _outputs, _predictions = sess.run(
            [totalLoss, outputs, predictions],
            feed_dict={
                data_input: dna_sequences,
                decoder_lengths: [len(i) for i in dna_sequences]
            })

        accuracy = calculate_accuracy(dna_sequences, sequenceLength, _predictions)

        if step % 50 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, _totalLoss, accuracy))
