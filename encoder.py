import numpy as np
import tensorflow as tf

hiddenUnits = 200

sequenceLength = 20

dnaNumLetters = 4

learningRate = 0.05

batchSize = 100

numTrainingIters = 500

# Used for gradient clipping by the global norm.
# The max value is often set to a value like 5 or 1.
max_gradient_norm = 2


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

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenUnits)

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
optimizer = tf.train.AdagradOptimizer(learningRate)
update_step = optimizer.apply_gradients(
    zip(clipped_gradients, params))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = read_data("dataset/dna_seq_50x20.txt")

    for epoch in range(numTrainingIters):
        #
        # get some data
        dna_sequences = get_dna_sequences(data)
        #
        # do the training epoch
        _totalLoss, _outputs, _predictions = sess.run(
            [totalLoss, outputs, predictions],
            feed_dict={
                data_input: dna_sequences,
                decoder_lengths: [len(i) for i in dna_sequences]
            })

    entSequence = 0
    correct = 0
    for i in range(len(dna_sequences)):
        numCorrect = 0
        for j in range(sequenceLength):
            maxPos = -1
            maxVal = 0.0
            for k in range(dnaNumLetters):
                if maxVal < _predictions[i][j][k]:
                    maxVal = _predictions[i][j][k]
                    maxPos = k
            if maxPos == np.argmax(dna_sequences[i][j]):
                numCorrect += 1
        if numCorrect == sequenceLength:
            entSequence += 1
        print(numCorrect)
        correct += numCorrect

    print("Step", epoch, "Loss", _totalLoss, "Correct", correct, "out of", batchSize * sequenceLength,
          " correct entire seq ", entSequence)
