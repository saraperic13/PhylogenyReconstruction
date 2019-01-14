import io

from newick import load

from tree_files.tree import Tree
# from tree_files.node import Node


def parse(file_name):
    with io.open(file_name, encoding='utf8') as fp:
        loaded_trees = load(fp)
        trees = []
        for t in loaded_trees:
            tree = Tree(leaves=t.get_leaves())
            load_tree(t, tree)
            trees.append(tree)
    return trees


def load_tree(node, tree):
    if not node.name:
        tree.increment_number_of_named()
        node.name = str(tree.get_number_of_named())

    # tree_node = create_tree_node(newick_node=node)
    tree.add_node(node)

    for child in node.descendants:
        load_tree(child, tree)


# def create_tree_node(newick_node):
#     return Node(newick_node.name, newick_node.ancestor, newick_node.descendants)


# parse("../dataset/10.2.tree")
