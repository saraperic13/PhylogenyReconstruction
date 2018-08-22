import random


def get_all_node_descendant_leaves(node, descendants):
    if not node.descendants:
        return
    for child in node.descendants:
        # only select leaves
        if not child.descendants:
            descendants.append(child)
    for child in node.descendants:
        get_all_node_descendant_leaves(child, descendants)


def get_random_descendant_leaves_from_subroot(subroot):
    descendants = []
    get_all_node_descendant_leaves(subroot, descendants)
    return descendants if not descendants else random.choice(descendants), random.choice(descendants)


def get_first_ancestor_after_root(node, root):
    if node.ancestor == root:
        return node
    return get_first_ancestor_after_root(node.ancestor, root)


def are_together(node_1, node_2, root):
    ancestor_node_1 = get_first_ancestor_after_root(node_1, root)
    ancestor_node_2 = get_first_ancestor_after_root(node_2, root)

    if ancestor_node_1 == ancestor_node_2:
        return True
    return False
