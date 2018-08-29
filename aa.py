import tensorflow as tf
import numpy as np

brisi = tf.placeholder(tf.int32, [None, 1, 4, 1])
#
brisi2 = tf.map_fn(lambda x: x * 2, brisi)

aa = tf.reduce_mean(brisi2, axis=1)

with tf.Session() as sess:
    _brisi2, _aa = sess.run(
        [brisi2, aa],
        feed_dict={
            brisi: [np.reshape([[2], [3], [4], [5]], (1, 4, 1))]
        })
    print(_brisi2)
    print(_aa)


def get_subroot_and_nodes(tree, data):
    subroot = tree.get_random_node()
    descendants = []
    tree_utils.get_all_node_descendant_leaves(subroot, descendants)

    dna_children_1, dna_children_2, together = [], [], []

    dna_descendants = []
    for child in descendants:
        dna_descendants.append(np.reshape(data[child.name][0].astype(np.float32), (1, 20, 4)))

    for i in range(batchSize):
        leaves = tree_utils.get_random_descendants(descendants)
        together.append(tree_utils.are_together(leaves[0], leaves[1], subroot))
        # print(leaves[0].name, ", ", leaves[1].name, " together: ", together, " subroot ", subroot.name)

        dna_children_1.append(data[leaves[0].name][0])
        dna_children_2.append(data[leaves[1].name][0])

    return subroot, dna_descendants, dna_children_1, dna_children_2, together


def get_subroot_and_nodes(tree, data):
    subroot = tree.get_random_node()
    descendants = []
    tree_utils.get_all_node_descendant_leaves(subroot, descendants)

    dna_children_1, dna_children_2, together = [], [], []

    dna_descendants = []
    for child in descendants:
        dna_descendants.append(np.reshape(data[child.name][0].astype(np.float32), (1, 20, 4)))

    for i in range(batchSize):
        leaves = tree_utils.get_random_descendants(descendants)
        together.append(tree_utils.are_together(leaves[0], leaves[1], subroot))
        # print(leaves[0].name, ", ", leaves[1].name, " together: ", together, " subroot ", subroot.name)

        dna_children_1.append(data[leaves[0].name][0])
        dna_children_2.append(data[leaves[1].name][0])

    return subroot, dna_descendants, dna_children_1, dna_children_2, together
