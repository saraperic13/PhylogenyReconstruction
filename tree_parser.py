from newick import load
import io
from tree import Tree
from tree_utils import get_random_descendant_leaves_from_subroot, are_together


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


def main():
    tree = parse('dataset/small_tree.tree')
    # print(tree)
    for i in range(10):
        subroot = tree.get_random_node()
        for j in range(10):
            children = get_random_descendant_leaves_from_subroot(subroot)
            print(children[0].name, ", ", children[1].name, " together: ",
                  are_together(children[0], children[1], subroot), " subroot ", subroot.name)


main()
