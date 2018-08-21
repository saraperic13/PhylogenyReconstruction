import random


def get_all_node_descendants(node, descendants):
    if not node.descendants:
        return
    for item in node.descendants:
        descendants.append(item)
    for child in node.descendants:
        get_all_node_descendants(child, descendants)


def get_random_descendants_from_subroot(subroot):
    descendants = []
    get_all_node_descendants(subroot, descendants)
    return descendants if not descendants else random.choice(descendants), random.choice(descendants)
