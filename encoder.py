import numpy as np
import tensorflow as tf

hiddenUnits = 200

sequenceLength = 20

dnaNumLetters = 4

learningRate = 0.05

batchSize = 100

numTrainingIters = 100


def sequence_to_one_hot_enc(seq):
    ltrdict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return [ltrdict[x] for x in seq if x in ltrdict]


def read_data(file_name):
    data = []
    with open(file_name) as f:
        content = f.readlines()
        for line in content:
            sequence = sequence_to_one_hot_enc(line)
            if len(sequence) > 0:
                data.append(np.array(sequence))
    return data


def get_dna_sequences(data):
    myInts = np.random.random_integers(0, len(data) - 1, batchSize)
    return np.stack(data[i] for i in myInts.flat)


data_input = tf.placeholder(tf.float32, [batchSize, sequenceLength, dnaNumLetters])

# Encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenUnits)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                   inputs=data_input, dtype=tf.float32, time_major=False)

the_last_output = encoder_outputs[:, -1, :]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = read_data("dataset/dna_seq_50x20.txt")

    for epoch in range(numTrainingIters):
        #
        # get some data
        dna_sequences = get_dna_sequences(data)

        _the_last_output = sess.run(
            [the_last_output],
            feed_dict={
                data_input: dna_sequences
            })

    print(_the_last_output)

    entSequence = 0
