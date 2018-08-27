import numpy as np
import tensorflow as tf
import tree_parser
import tree_utils

hiddenUnits = 200

sequenceLength = 20

dnaNumLetters = 4

learningRate = 0.05

batchSize = 10

numTrainingIters = 50


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


def get_subroot_and_nodes(tree, data):
    dna_children_1, dna_children_2, together, dna_descendants, some = [], [], [], [], []

    for i in range(batchSize):
        subroot = tree.get_random_node()
        descendants = []
        tree_utils.get_all_node_descendant_leaves(subroot, descendants)

        for child in descendants:
            # dna_descendants.append(np.reshape(data[child.name][0].astype(np.float32), (1, 20, 4)))
            dna_descendants.append(data[child.name][0].astype(np.float32))

        some.append(dna_descendants)

        leaves = tree_utils.get_random_descendants(descendants)
        together.append([tree_utils.are_together(leaves[0], leaves[1], subroot)])
        # print(leaves[0].name, ", ", leaves[1].name, " together: ", together, " subroot ", subroot.name)

        dna_children_1.append(data[leaves[0].name][0])
        dna_children_2.append(data[leaves[1].name][0])

    return some, dna_children_1, dna_children_2, together


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


tree = tree_parser.parse('dataset/phylogeny.tree')

data_input = tf.placeholder(tf.float32, [batchSize, None, sequenceLength, dnaNumLetters])
dna_sequence_input_1 = tf.placeholder(tf.float32, [batchSize, sequenceLength, dnaNumLetters])
dna_sequence_input_2 = tf.placeholder(tf.float32, [batchSize, sequenceLength, dnaNumLetters])
inputY = tf.placeholder(tf.float32, [batchSize, 1])

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

feed_forward_inputX = tf.concat([encoded_dataset, encoded_dna_sequence_1, encoded_dna_sequence_2], 1)
# Classifier
# Weight initializations
w_1 = init_weights((3 * hiddenUnits, hiddenUnits))
b1 = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

w_2 = init_weights((hiddenUnits, 1))
b2 = tf.Variable(np.zeros((1, 1)), dtype=tf.float32)

h = tf.nn.tanh(tf.matmul(feed_forward_inputX, w_1) + b1)
output = tf.matmul(h, w_2) + b2

predictions = tf.nn.sigmoid(output)

losses = tf.nn.sigmoid_cross_entropy_with_logits (labels=inputY, logits=output)
total_loss = tf.reduce_mean(losses)

training_alg = tf.train.AdagradOptimizer(0.02).minimize(total_loss)

correct_pred = tf.equal(tf.round(predictions), tf.cast(inputY, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = read_data("dataset/dna_sequences_20.txt")

    for step in range(numTrainingIters + 1):

        dna_descendants, dna_child_1, dna_child_2, together = get_subroot_and_nodes(tree, data)

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
