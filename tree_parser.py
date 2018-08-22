from newick import load
import io
from tree import Tree


def load_tree(node, tree):
    if not node.name:
        tree.increment_number_of_named()
        node.name = tree.get_number_of_named()

    tree.add_node(node)

    for child in node.descendants:
        load_tree(child, tree)


def parse(file_name):
    with io.open(file_name, encoding='utf8') as fp:
        loaded_tree = load(fp)[0]
        tree = Tree(len(loaded_tree.get_leaves()))
        load_tree(loaded_tree, tree)
    return tree
