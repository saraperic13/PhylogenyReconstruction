import random
import numpy as np



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


def get_subroot_and_nodes(tree, data, batchSize, max_size_dataset, sequence_length, dna_num_letters=4):
    dna_children_1, dna_children_2, together, dataset = [], [], [], []

    for i in range(batchSize):
        subroot = tree.get_random_node()
        descendants, dnas = [], []
        get_all_node_descendant_leaves(subroot, descendants)

        random_dataset_idx = random.randint(0, len(data[descendants[0].name])-1)

        for child in descendants:
            dnas.append(data[child.name][random_dataset_idx])

        for i in range(max_size_dataset - len(descendants)):
            dnas.append(np.zeros(dna_num_letters*sequence_length, dtype=np.int64))

        dataset.append(dnas)

        leaves = get_random_descendants(descendants)
        together.append(are_together(leaves[0], leaves[1], subroot))

        dna_children_1.append(data[leaves[0].name][random_dataset_idx])
        dna_children_2.append(data[leaves[1].name][random_dataset_idx])

    return dataset, dna_children_1, dna_children_2, together