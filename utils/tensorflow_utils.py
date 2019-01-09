import tensorflow as tf


def create_and_append_matrix(input_size, output_size, list):
    weight_matrix = init_variable(shape=(input_size, output_size))
    list.append(weight_matrix)


def init_variable(shape,name=None):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, validate_shape=False, name=name)


def make_placeholder(shape, name):
    return tf.placeholder(tf.float32, shape=shape, name=name)


def make_constant(shape, name, value):
    return tf.constant(name=name, value=value, shape=shape)


def make_int_placeholder(shape, name):
    return tf.placeholder(tf.int32, shape=shape, name=name)


def make_fill(name, value):
    return tf.fill(tf.int32, name=name, value=value)


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


def convert_to_int_32(value):
    return tf.int32(value)


def create_and_append_matrix_dynamic_shape(shape, list):

    zero_fill = tf.fill(shape, 0.0)
    variable = tf.Variable(0.0, validate_shape=False)
    update_shape_variable = tf.assign(variable, zero_fill, validate_shape=False)

    list.append(update_shape_variable)