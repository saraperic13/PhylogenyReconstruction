import io

from newick import load

from tree_files.tree import Tree

def load_tree(node, tree):
    if not node.name:
        tree.increment_number_of_named()
        node.name = str(tree.get_number_of_named())

    tree.add_node(node)

    for child in node.descendants:
        load_tree(child, tree)


def parse(file_name):
    with io.open(file_name, encoding='utf8') as fp:
        loaded_trees = load(fp)
        trees = []
        for t in loaded_trees:
            tree = Tree(len(t.get_leaves()))
            load_tree(t, tree)
            trees.append(tree)
    return trees

# parse("../dataset/100-trees/100_20.2.tree")
