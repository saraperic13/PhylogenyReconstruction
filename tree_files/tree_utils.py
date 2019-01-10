import random
import itertools
from tree_files.node import  Node


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

    # together
    if ancestor_node_1 == ancestor_node_2:
        return [1, 0]
    return [0, 1]


def get_all_node_pairs(node_list):
    return list(itertools.combinations(node_list, r=2))


def create_nodes(nodes_name_list):
    nodes = []
    for name in nodes_name_list:
        nodes.append(Node(name))
    return nodes
