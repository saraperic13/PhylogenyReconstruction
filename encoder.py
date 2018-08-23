import numpy as np
import tensorflow as tf
import tree_parser
import tree_utils

hiddenUnits = 200

sequenceLength = 20

dnaNumLetters = 4

learningRate = 0.05

batchSize = 100

numTrainingIters = 10

dna_letters = "ACTG"


def sequence_to_one_hot_enc(seq):
    letter_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    one_hot_seq = []
    species = []
    for i in seq:
        if i in letter_dict:
            one_hot_seq.append(letter_dict[i])
        else:
            species.append(i)
    species = "".join(species)
    species = species.strip()
    return species, one_hot_seq


def read_data(file_name):
    data = {}
    with open(file_name) as f:
        content = f.readlines()
        for line in content:
            if "A" not in line and "C" not in line and "G" not in line and "T" not in line:
                continue
            species, sequence = sequence_to_one_hot_enc(line)
            if len(sequence) > 0:

                if species not in data:
                    data[species] = []

                data[species].append(np.array(sequence))
    return data


def get_dna_sequences(data):
    myInts = np.random.random_integers(0, len(data) - 1, batchSize)
    return np.stack(data[i] for i in myInts.flat)


def get_subroot_and_nodes(tree, data):
    # for i in range(batchSize):
    subroot = tree.get_random_node()
    descendants = []
    tree_utils.get_all_node_descendant_leaves(subroot, descendants)

    leaves = tree_utils.get_random_descendants(descendants)
    together = tree_utils.are_together(leaves[0], leaves[1], subroot)
    print(leaves[0].name, ", ", leaves[1].name, " together: ", together, " subroot ", subroot.name)

    dna_descendants = []
    for child in descendants:
        dna_descendants.append(data[child.name][0])

    dna_child_1 = data[leaves[0].name][0]
    dna_child_2 = data[leaves[1].name][0]

    return subroot, dna_descendants, dna_child_1, dna_child_2, together


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


tree = tree_parser.parse('dataset/small_tree.tree')

data_input = tf.placeholder(tf.float32, [None, sequenceLength, dnaNumLetters])
# data_input = tf.placeholder(tf.float32, [batchSize, sequenceLength, dnaNumLetters])
dna_sequence_input_1 = tf.placeholder(tf.float32, [1, sequenceLength, dnaNumLetters])
dna_sequence_input_2 = tf.placeholder(tf.float32, [1, sequenceLength, dnaNumLetters])
inputY = tf.placeholder(tf.int32, [1])

# Encoder
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenUnits)

encoded_dataset = []

# brisi = tf.placeholder(tf.int32, [None, 4, 1])
#
# brisi2 = tf.map_fn(lambda x: x*2, brisi)
encoded_dataset = tf.map_fn(lambda x: tf.nn.dynamic_rnn(cell=encoder_cell,
                                                        inputs=tf.reshape(x, [1, dnaNumLetters, sequenceLength]),
                                                        dtype=tf.float32), data_input,
                            dtype=tf.float32)
# for node in data_input:
#     encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
#                                            inputs=node, dtype=tf.float32)
#
#     encoded_node = encoder_outputs[:, -1, :]
#     encoded_dataset.append(encoded_node)

# encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
#                                            inputs=data_input, dtype=tf.float32)
# encoded_dataset = encoder_outputs[:, -1, :]

encoded_dataset = np.mean(encoded_dataset, axis=1)

# encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
#                                        inputs=data_input, dtype=tf.float32)
# encoded_dataset = encoder_outputs[:, -1, :]

encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
                                       inputs=dna_sequence_input_1, dtype=tf.float32)
encoded_dna_sequence_1 = encoder_outputs[:, -1, :]

encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder_cell,
                                       inputs=dna_sequence_input_2, dtype=tf.float32)
encoded_dna_sequence_2 = encoder_outputs[:, -1, :]

# feed_forward_inputX = tf.stack([encoded_dataset[0], encoded_dna_sequence_1[0], encoded_dna_sequence_2[0]])
feed_forward_inputX = tf.concat([encoded_dataset, encoded_dna_sequence_1, encoded_dna_sequence_2], 1)
# Classifier
# Weight initializations
w_1 = init_weights((600, hiddenUnits))
b1 = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

w_2 = init_weights((hiddenUnits, 2))
b2 = tf.Variable(np.zeros((1, 2)), dtype=tf.float32)

h = tf.nn.tanh(tf.matmul(feed_forward_inputX, w_1) + b1)
output = tf.matmul(h, w_2) + b2

predictions = tf.nn.softmax(output)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputY, logits=output)
total_loss = tf.reduce_mean(losses)

training_alg = tf.train.AdagradOptimizer(0.02).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = read_data("dataset/small_tree_seq.txt")

    for epoch in range(numTrainingIters):
        #
        # get some data
        # dna_sequences = get_dna_sequences(data)

        subroot, dna_descendants, dna_child_1, dna_child_2, together = get_subroot_and_nodes(tree, data)

        _encoded_dataset, _totalLoss, _training_alg, _predictions = sess.run(
            [encoded_dataset, total_loss, training_alg, predictions],
            feed_dict={
                # brisi: [[[2], [3], [4], [5]]],
                data_input: dna_descendants,
                dna_sequence_input_1: [dna_child_1],
                dna_sequence_input_2: [dna_child_2],
                inputY: [together]
            })

        print(encoded_dataset)
        print(predictions)

    entSequence = 0
