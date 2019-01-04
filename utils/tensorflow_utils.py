import tensorflow as tf


def create_and_append_matrix(input_size, output_size, list):
    weight_matrix = init_variable(shape=(input_size, output_size))
    list.append(weight_matrix)


def init_variable(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, validate_shape=False)


def make_placeholder(shape, name):
    return tf.placeholder(tf.float32, shape=shape, name=name)


def multiply_sequence_weight_matrices(sequence, weights_matrices, bias_matrices):
    for i in range(len(weights_matrices)):
        sequence = tf.matmul(sequence, weights_matrices[i]) + bias_matrices[i]
    return sequence


def multiply_sequence_weight_matrices_with_activation(sequence, weights_matrices, bias_matrices):
    for i in range(len(weights_matrices)):
        sequence = tf.matmul(sequence, weights_matrices[i]) + bias_matrices[i]

        if i == len(weights_matrices) - 1:
            prediction = tf.nn.softmax(sequence, name="predictions")
            return sequence, prediction
        else:
            sequence = tf.nn.relu(sequence)

    return sequence
