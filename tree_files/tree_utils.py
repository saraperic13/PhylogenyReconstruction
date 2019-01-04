import random
import itertools


def get_all_node_descendant_leaves(node, descendants):
    if not node.descendants:
        return
    for child in node.descendants:
        # only select leaves
        if not child.descendants:
            descendants.append(child)
    for child in node.descendants:
        get_all_node_descendant_leaves(child, descendants)


def get_random_descendants(descendants):
    return descendants if not descendants else random.sample(descendants, 2)


def get_first_ancestor_after_root(node, root):
    if node.ancestor == root:
        return node
    return get_first_ancestor_after_root(node.ancestor, root)


def are_together(node_1, node_2, root):
    ancestor_node_1 = get_first_ancestor_after_root(node_1, root)
    ancestor_node_2 = get_first_ancestor_after_root(node_2, root)

    if ancestor_node_1 == ancestor_node_2:
        return [1, 0]
    return [0, 1]


def get_batch_sized_data(batch_size, training_data_model):
    for i in range(batch_size):
        get_subroot_and_descendants(training_data_model)

    return training_data_model


def get_subroot_and_descendants(training_data_model):
    descendants = []
    subroot = training_data_model.select_random_subroot()

    get_all_node_descendant_leaves(subroot, descendants)

    training_data_model.process_dataset(descendants)

    leaves = get_random_descendants(descendants)
    together = are_together(leaves[0], leaves[1], subroot)
    training_data_model.process_leaves(leaves, together)


def get_all_node_pairs(tree):
    return list(itertools.combinations(tree.leaves, r=2))
